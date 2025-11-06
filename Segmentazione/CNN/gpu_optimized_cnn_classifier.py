"""
GPU ottimizzata versione di CNN + Classificatore.
Riduce il numero di campioni generati per i classificatori boosting
e abilita l'uso della GPU per XGBoost/LightGBM quando disponibile.
"""

import os
import glob
import pickle
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import optuna
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import lightgbm as lgb


class XGBBoosterWrapper:
    """Wrapper per rendere un booster XGBoost simile allo sklearn estimator."""

    def __init__(self, booster: xgb.Booster, num_classes: int):
        self.booster = booster
        self.num_classes = num_classes

    def predict(self, X: np.ndarray) -> np.ndarray:
        dmatrix = xgb.DMatrix(X)
        preds = self.booster.predict(dmatrix)
        if preds.ndim == 1:
            return np.rint(preds).astype(int)
        return np.argmax(preds, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        dmatrix = xgb.DMatrix(X)
        preds = self.booster.predict(dmatrix)
        if preds.ndim == 1:
            preds = np.vstack([1 - preds, preds]).T
        return preds


@dataclass
class GPUClassifierConfig:
    cnn_model: str = "convnext_tiny"
    classifier: str = "xgboost"  # "xgboost" o "lightgbm"
    images_dir: str = "../images/Immagini"
    masks_dir: str = "../images/Maschere"
    image_size: Tuple[int, int] = (1024, 1024)
    feature_map_size: Optional[Tuple[int, int] | int] = None  # None => usa la risoluzione dell'immagine
    batch_size: int = 2
    use_augmentation: bool = True
    max_pixels_per_image: int = 20000  # Limita il numero di sample per immagine
    use_gpu: bool = True
    num_classes: int = 5
    decoder_filters: int = 128


class GPUOptimizedCNNSegmentationClassifier:
    """
    Pipeline CNN -> Feature Map -> Classificatore boosting con Optuna.
    Estrae feature spaziali a risoluzione ridotta per mantenere info pixel-level
    senza esplodere in memoria, e abilita parametri GPU per i classificatori.
    """

    def __init__(self, config: Optional[GPUClassifierConfig] = None):
        self.config = config or GPUClassifierConfig()
        self.images_dir = self.config.images_dir
        self.masks_dir = self.config.masks_dir

        self.class_names = [
            "Resina",
            "Pori/Imperfezioni",
            "Fase Fusa",
            "Belite",
            "Alite",
        ]

        self._feature_extractor: Optional[tf.keras.Model] = None
        self.classifier = None
        self.best_params: Optional[Dict] = None
        self.class_weights: Optional[Dict[int, float]] = None
        self.best_num_boost_round: Optional[int] = None

        self.X_features: Optional[np.ndarray] = None
        self.y_labels: Optional[np.ndarray] = None
        self._feature_extractor_weights: Optional[list[np.ndarray]] = None
        self.current_image_size: Tuple[int, int] = tuple(self.config.image_size)

    # ------------------------------------------------------------------ #
    # Data loading
    # ------------------------------------------------------------------ #

    def load_data(self):
        """Carica immagini e maschere con augmentation opzionale."""
        img_paths = sorted(glob.glob(os.path.join(self.images_dir, "*.png")))
        mask_paths = sorted(glob.glob(os.path.join(self.masks_dir, "*.tif")))
        if not img_paths or not mask_paths:
            raise FileNotFoundError(
                f"Nessuna immagine trovata in {self.images_dir} "
                f"o nessuna maschera in {self.masks_dir}"
            )

        images = []
        masks = []

        for img_path, mask_path in zip(img_paths, mask_paths):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.config.image_size)
            images.append(img.astype(np.float32) / 255.0)

            mask = Image.open(mask_path)
            mask_np = np.array(mask)
            if mask_np.ndim == 3:
                mask_np = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)
            mask_np = cv2.resize(
                mask_np,
                self.config.image_size,
                interpolation=cv2.INTER_NEAREST,
            )
            masks.append(mask_np.astype(np.int32))

        self.images = np.asarray(images)
        self.masks = np.asarray(masks)
        if self.images.size == 0 or self.masks.size == 0:
            raise ValueError("Dataset vuoto dopo il caricamento.")

        if self.config.use_augmentation:
            self._apply_augmentation()

        self._compute_class_weights()

    def _apply_augmentation(self):
        """Augmentation leggera per evitare duplicati identici."""
        transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.15, contrast_limit=0.15, p=0.5
                ),
            ]
        )

        augmented_images = []
        augmented_masks = []
        for img, mask in zip(self.images, self.masks):
            augmented_images.append(img)
            augmented_masks.append(mask)

            aug = transform(image=img, mask=mask)
            augmented_images.append(aug["image"])
            augmented_masks.append(aug["mask"])

        self.images = np.asarray(augmented_images)
        self.masks = np.asarray(augmented_masks)

    def _compute_class_weights(self):
        """Calcola pesi di bilanciamento ignorando classe 0."""
        labels = self.masks.flatten()
        labels = labels[labels != 0] - 1
        if labels.size == 0:
            self.class_weights = None
            return

        unique = np.unique(labels)
        weights = compute_class_weight("balanced", classes=unique, y=labels)
        self.class_weights = dict(zip(unique.tolist(), weights.tolist()))

    # ------------------------------------------------------------------ #
    # Feature extraction
    # ------------------------------------------------------------------ #

    def extract_features(self):
        """Estrae feature spaziali ridotte dalla CNN."""
        if self._feature_extractor is None:
            self._feature_extractor = self._build_feature_extractor()
            self._feature_extractor_weights = self._feature_extractor.get_weights()
            self.current_image_size = tuple(self.config.image_size)

        features = []
        labels = []
        rng = np.random.default_rng(seed=42)

        for idx in range(0, len(self.images), self.config.batch_size):
            batch_imgs = self.images[idx : idx + self.config.batch_size]
            batch_masks = self.masks[idx : idx + self.config.batch_size]
            feats = self._feature_extractor.predict(batch_imgs, verbose=0)

            # feats shape: (B, Hf, Wf, C)
            for img_feat, mask in zip(feats, batch_masks):
                h_f, w_f = img_feat.shape[:2]
                mask_resized = cv2.resize(
                    mask,
                    (w_f, h_f),
                    interpolation=cv2.INTER_NEAREST,
                )

                valid = mask_resized != 0
                if not np.any(valid):
                    continue

                feat_flat = img_feat.reshape(-1, img_feat.shape[-1])
                label_flat = mask_resized.reshape(-1)
                feat_filtered = feat_flat[valid.ravel()]
                label_filtered = label_flat[valid.ravel()] - 1
                label_filtered = np.clip(label_filtered, 0, self.config.num_classes - 1)

                if (
                    self.config.max_pixels_per_image is not None
                    and len(label_filtered) > self.config.max_pixels_per_image
                ):
                    sample_idx = rng.choice(
                        len(label_filtered),
                        size=self.config.max_pixels_per_image,
                        replace=False,
                    )
                    feat_filtered = feat_filtered[sample_idx]
                    label_filtered = label_filtered[sample_idx]

                features.append(feat_filtered)
                labels.append(label_filtered)

        if not features:
            raise ValueError("Nessuna feature valida estratta.")

        self.X_features = np.vstack(features)
        self.y_labels = np.hstack(labels)

    def ensure_feature_extractor_size(self, image_size: Tuple[int, int]) -> None:
        """
        Ricostruisce il feature extractor per gestire immagini di dimensioni diverse.
        Mantiene i pesi addestrati, ma aggiorna l'input/output spatial size.
        """
        target_size = (int(image_size[0]), int(image_size[1]))
        if self._feature_extractor is None:
            raise RuntimeError("Feature extractor non inizializzato: carica o addestra il modello prima.")

        if tuple(self.current_image_size) == target_size:
            return

        if self._feature_extractor_weights is None:
            self._feature_extractor_weights = self._feature_extractor.get_weights()

        rebuilt = self._build_feature_extractor(override_image_size=target_size)
        rebuilt.set_weights(self._feature_extractor_weights)
        self._feature_extractor = rebuilt
        self.current_image_size = target_size
        self._feature_extractor_weights = self._feature_extractor.get_weights()

    def _build_feature_extractor(
        self, override_image_size: Optional[Tuple[int, int]] = None
    ) -> tf.keras.Model:
        """Crea la CNN di base e restituisce feature map ridimensionata."""
        image_size = override_image_size or self.config.image_size
        input_shape = (*image_size, 3)
        model_name = self.config.cnn_model.lower()

        if model_name == "resnet50":
            base = tf.keras.applications.ResNet50(
                weights="imagenet", include_top=False, input_shape=input_shape
            )
        elif model_name == "convnext_tiny":
            base = tf.keras.applications.ConvNeXtTiny(
                weights="imagenet", include_top=False, input_shape=input_shape
            )
        else:
            raise ValueError(
                f"Modello CNN non supportato: {self.config.cnn_model}. "
                "Usa 'resnet50' o 'convnext_tiny'."
            )

        for layer in base.layers:
            layer.trainable = False

        # Determina la dimensione target della feature map finale.
        target_config = self.config.feature_map_size
        if target_config is None:
            target_h, target_w = image_size
        elif isinstance(target_config, int):
            target_h = target_w = int(target_config)
        else:
            target_h, target_w = target_config

        decoder_filters = self.config.decoder_filters

        x = base.output
        current_h = x.shape[1]
        current_w = x.shape[2]

        # Se la dimensione è indefinita (None) lasciamo che la Resizing finale gestisca il target.
        while (
            (current_h is not None and current_h < target_h)
            or (current_w is not None and current_w < target_w)
        ):
            x = tf.keras.layers.Conv2D(
                decoder_filters, kernel_size=3, padding="same", activation="relu"
            )(x)
            x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
            if current_h is not None:
                current_h *= 2
            if current_w is not None:
                current_w *= 2

        x = tf.keras.layers.Conv2D(
            decoder_filters, kernel_size=3, padding="same", activation="relu"
        )(x)
        x = tf.keras.layers.Conv2D(
            decoder_filters // 2, kernel_size=3, padding="same", activation="relu"
        )(x)

        resized = tf.keras.layers.Resizing(
            target_h,
            target_w,
            interpolation="bilinear",
            name="feature_resizer",
        )(x)
        return tf.keras.Model(inputs=base.input, outputs=resized)

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train_classifier_optuna(
        self,
        n_trials: int = 20,
        timeout: Optional[int] = None,
        test_size: float = 0.2,
    ) -> float:
        """
        Esegue ottimizzazione Optuna e restituisce l'accuracy sul validation set.
        """
        if self.X_features is None or self.y_labels is None:
            raise RuntimeError("Devi chiamare extract_features() prima del training.")

        X_train, X_val, y_train, y_val = train_test_split(
            self.X_features,
            self.y_labels,
            test_size=test_size,
            random_state=42,
            stratify=self.y_labels,
        )

        sample_weights = None
        if self.class_weights is not None:
            sample_weights = np.array([self.class_weights.get(y, 1.0) for y in y_train])

        def objective(trial: optuna.Trial) -> float:
            if self.config.classifier == "lightgbm":
                params = self._suggest_lightgbm_params(trial)
                model = lgb.LGBMClassifier(**params)
                fit_kwargs = {
                    "X": X_train,
                    "y": y_train,
                    "eval_set": [(X_val, y_val)],
                    "callbacks": [lgb.early_stopping(20), lgb.log_evaluation(0)],
                }
                if sample_weights is not None:
                    fit_kwargs["sample_weight"] = sample_weights
                try:
                    model.fit(**fit_kwargs)
                except lgb.basic.LightGBMError as ex:
                    if self.config.use_gpu:
                        warnings.warn(
                            f"LightGBM GPU non disponibile ({ex}). "
                            "Riprovo in CPU.",
                            RuntimeWarning,
                        )
                        params_cpu = params.copy()
                        params_cpu.pop("device", None)
                        model = lgb.LGBMClassifier(**params_cpu)
                        model.fit(**fit_kwargs)
                    else:
                        raise
            else:
                params = self._suggest_xgboost_params(trial)
                num_boost_round = params["n_estimators"]
                train_params = self._build_xgb_train_params(params)
                dtrain = xgb.DMatrix(
                    X_train,
                    label=y_train,
                    weight=sample_weights if sample_weights is not None else None,
                )
                dval = xgb.DMatrix(X_val, label=y_val)
                try:
                    bst = xgb.train(
                        train_params,
                        dtrain,
                        num_boost_round=num_boost_round,
                        evals=[(dval, "validation")],
                        early_stopping_rounds=20,
                        verbose_eval=False,
                    )
                except xgb.core.XGBoostError as ex:
                    if self.config.use_gpu:
                        warnings.warn(
                            f"XGBoost GPU non disponibile ({ex}). "
                            "Riprovo in CPU.",
                            RuntimeWarning,
                        )
                        cpu_params = train_params.copy()
                        cpu_params["tree_method"] = "hist"
                        cpu_params.pop("predictor", None)
                        cpu_params.pop("device", None)
                        bst = xgb.train(
                            cpu_params,
                            dtrain,
                            num_boost_round=num_boost_round,
                            evals=[(dval, "validation")],
                            early_stopping_rounds=20,
                            verbose_eval=False,
                        )
                    else:
                        raise

                best_iteration = (
                    bst.best_iteration if bst.best_iteration is not None else num_boost_round - 1
                )
                trial.set_user_attr("best_iteration", int(best_iteration))
                preds = bst.predict(dval)
                preds = np.argmax(preds, axis=1)

            # LightGBM branch returns preds too
            if self.config.classifier == "lightgbm":
                preds = model.predict(X_val)

            return accuracy_score(y_val, preds)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        self.best_params = study.best_params
        best_iteration = study.best_trial.user_attrs.get(
            "best_iteration", self.best_params.get("n_estimators", 1) - 1
        )
        self.best_num_boost_round = int(best_iteration) + 1
        print(f"Migliori parametri: {self.best_params}")
        print(f"Migliore accuracy (val): {study.best_value:.4f}")

        # Training finale sul dataset completo
        if self.config.classifier == "lightgbm":
            final_params = self.best_params.copy()
            final_params.update(
                {
                    "objective": "multiclass",
                    "num_class": self.config.num_classes,
                    "random_state": 42,
                }
            )
            model = lgb.LGBMClassifier(**final_params)
            fit_args = {"X": self.X_features, "y": self.y_labels}
            if self.class_weights is not None:
                full_weights = np.array(
                    [self.class_weights.get(y, 1.0) for y in self.y_labels]
                )
                fit_args["sample_weight"] = full_weights
            model.fit(**fit_args)
        else:
            final_params = self.best_params.copy()
            train_params = self._build_xgb_train_params(final_params)
            num_boost_round = (
                self.best_num_boost_round
                if self.best_num_boost_round is not None
                else final_params["n_estimators"]
            )
            dtrain_full = xgb.DMatrix(
                self.X_features,
                label=self.y_labels,
                weight=(
                    np.array([self.class_weights.get(y, 1.0) for y in self.y_labels])
                    if self.class_weights is not None
                    else None
                ),
            )
            bst_final = xgb.train(
                train_params,
                dtrain_full,
                num_boost_round=num_boost_round,
                verbose_eval=False,
            )
            model = XGBBoosterWrapper(bst_final, self.config.num_classes)

        self.classifier = model
        return study.best_value

    # ------------------------------------------------------------------ #
    # Helper per parametri
    # ------------------------------------------------------------------ #

    def _suggest_xgboost_params(self, trial: optuna.Trial) -> Dict:
        params = {
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2),
            "n_estimators": trial.suggest_int("n_estimators", 80, 200),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            "gamma": trial.suggest_float("gamma", 0.0, 0.4),
            "scale_pos_weight": 1.0,
            "verbosity": 0,
            "eval_metric": "mlogloss",
        }

        if self.config.use_gpu:
            params.update(
                {
                    "tree_method": "hist",
                    "predictor": "gpu_predictor",
                    "gpu_id": 0,
                    "device": "cuda",
                }
            )
        else:
            params["tree_method"] = "hist"
            params["n_jobs"] = -1

        return params

    def _build_xgb_train_params(self, params: Dict) -> Dict:
        train_params = {
            "max_depth": params["max_depth"],
            "eta": params["learning_rate"],
            "subsample": params["subsample"],
            "colsample_bytree": params["colsample_bytree"],
            "min_child_weight": params["min_child_weight"],
            "reg_alpha": params["reg_alpha"],
            "reg_lambda": params["reg_lambda"],
            "gamma": params["gamma"],
            "objective": "multi:softprob",
            "num_class": self.config.num_classes,
            "eval_metric": "mlogloss",
            "verbosity": 0,
        }

        if self.config.use_gpu:
            train_params.update(
                {
                    "tree_method": "hist",
                    "predictor": "gpu_predictor",
                    "device": "cuda",
                }
            )
        else:
            train_params["tree_method"] = "hist"

        return train_params

    def _suggest_lightgbm_params(self, trial: optuna.Trial) -> Dict:
        params = {
            "boosting_type": "gbdt",
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2),
            "n_estimators": trial.suggest_int("n_estimators", 80, 220),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 120),
            "objective": "multiclass",
            "num_class": self.config.num_classes,
            "verbosity": -1,
            "random_state": 42,
        }

        if self.config.use_gpu:
            params.update(
                {
                    "device": "gpu",
                    "gpu_platform_id": 0,
                    "gpu_device_id": 0,
                }
            )

        return params

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path_prefix: str):
        """Salva feature extractor (+) classificatore su disco."""
        if self.classifier is None or self._feature_extractor is None:
            raise RuntimeError("Modello non addestrato, nulla da salvare.")

        feature_path = f"{path_prefix}_feature_extractor.keras"
        self._feature_extractor.save(feature_path)

        payload = {
            "classifier": self.classifier,
            "best_params": self.best_params,
            "class_weights": self.class_weights,
            "config": self.config,
            "best_num_boost_round": self.best_num_boost_round,
        }
        with open(f"{path_prefix}_classifier.pkl", "wb") as f:
            pickle.dump(payload, f)

        print(f"Modello salvato (feature extractor + classifier) con prefisso {path_prefix}")

    def load(self, path_prefix: str):
        """Carica modello precedentemente salvato."""
        feature_path = f"{path_prefix}_feature_extractor.keras"
        classifier_path = f"{path_prefix}_classifier.pkl"
        if not os.path.exists(feature_path) or not os.path.exists(classifier_path):
            raise FileNotFoundError("File del modello non trovati.")

        self._feature_extractor = tf.keras.models.load_model(feature_path)
        self._feature_extractor_weights = self._feature_extractor.get_weights()
        with open(classifier_path, "rb") as f:
            payload = pickle.load(f)

        self.classifier = payload["classifier"]
        self.best_params = payload.get("best_params")
        self.class_weights = payload.get("class_weights")
        self.config = payload.get("config", self.config)
        self.best_num_boost_round = payload.get("best_num_boost_round", self.best_num_boost_round)
        self.current_image_size = tuple(self.config.image_size)
