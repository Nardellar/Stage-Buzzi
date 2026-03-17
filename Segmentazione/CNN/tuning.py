"""
Utilities per il tuning basato su Optuna dei classificatori boosting.
"""
from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple, TYPE_CHECKING, Any

import lightgbm as lgb
import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import f1_score

if TYPE_CHECKING:
    from gpu_optimized_cnn_classifier import GPUClassifierConfig

#utile per creare un array dei pesi dei pixel parallelo all'array di label di ogni pixel -> da fornire poi al classifciatore
def calculate_pixels_weights(labels: np.ndarray, class_weights: Optional[Dict[int, float]]) -> np.ndarray:
    """
    Preso un elenco di labels (cioe' la label di ogni pixel) e i pesi relativi ad ogni label, 
    restituisce un array con stessa lunghezza di labels che contiene i pesi relativi ad ogni pixel.
    """
    if class_weights is None:
        raise ValueError("class_weights non può essere None. I pesi sono obbligatori per il tuning.")
    return np.array([class_weights[int(element)] for element in labels], dtype=float)


def _suggest_xgboost_params(config: "GPUClassifierConfig", trial: optuna.Trial) -> Dict:
    """
    Suggerisce i parametri per XGBoost usando Optuna.
    Optuna li usa durante i trial per installare un modello XGBoost
    """
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 12), #profodndita' alberi
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 400), #numero di alberi
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0), #percentuale di colonne da considerare per ogni albero
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10), #peso minimo del campione per ogni nodo
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True), #regolarizzazione L1
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),#regolarizzazione L2
        "gamma": trial.suggest_float("gamma", 0.0, 0.6),
        "scale_pos_weight": 1.0,
        "verbosity": 0,
        "eval_metric": "mlogloss", #metrica di valutazione per l'early stopping e sta per Multi-Class Log Loss (mlogloss) calcola la loss dalla differenza tra "probabilita' classe predetta" e la classe effettiva
    }

    return params


def _build_xgb_train_params(params: Dict, config: "GPUClassifierConfig") -> Dict:
    """
    Converte il dizionario sklearn di suggest_XGBoost_params in un formato accettabile dall'API di XGBoost. -> Ogni volta che optuna propone iper-parametri in un trial li convertiamo in un formato accettabile a "xgb.train"
    Costruisce i parametri di training per XGBoost usando i parametri suggeriti da Optuna.
    "params" sono i parametri suggeriti da Optuna
    "config" contiene la configurazione del classificatore
    """
    #creiamo un dizionario con i parametri di training
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
        "num_class": config.num_classes,
        "eval_metric": "mlogloss", #metrica di valutazione per l'early stopping e sta per Multi-Class Log Loss (mlogloss) calcola la loss dalla differenza tra "probabilita' classe predetta" e la classe effettiva
        "verbosity": 0,
    }

    #se config ci segnala che abbiamo una GPU
    if config.use_gpu:
        train_params.update(
            {
                "tree_method": "hist",
                "predictor": "gpu_predictor",
                "device": "cuda",
            }
        )
    #se abbiamo solo CPU
    else:
        train_params["tree_method"] = "hist"
        train_params["n_jobs"] = -1
    return train_params


def _suggest_lightgbm_params(config: "GPUClassifierConfig", trial: optuna.Trial) -> Dict:
    """
    Suggerisce i parametri per LightGBM usando Optuna.
    Optuna li usa durante i trial per installare un modello LightGBM
    (a differenza di XGBoost, non e' necessaria una funzione build poiche' usa API sklearn-like)
    """
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
        "num_class": config.num_classes,
        "verbosity": -1,
        "random_state": 42,
    }

    #se config ci segnala che abbiamo una GPU
    if config.use_gpu:
        params.update(
            {
                "device": "gpu",
                "gpu_platform_id": 0,
                "gpu_device_id": 0,
            }
        )

    return params


def tune_classifier(config: "GPUClassifierConfig", pixels_features: np.ndarray, pixels_labels: np.ndarray, class_weights: Optional[Dict[int, float]], validation_data: Tuple[np.ndarray, np.ndarray], n_trials: int = 20) -> Tuple[float, Any, Dict, Optional[int]]:
    """
    Esegue ottimizzazione Optuna per il classificatore specificato in config e restituisce
    (best_value, best_classifier, best_params, best_iteration).
    """
    
    
    if pixels_features is None or pixels_labels is None:
        raise RuntimeError("Feature o label non disponibili per il tuning.")

    val_features, val_labels = validation_data
    #calcoliamo il peso di ogni pixel e lo salviamo
    pixels_weights = calculate_pixels_weights(pixels_labels, class_weights)

    #chiamata ad ogni trial
    def train_and_evaluate_trial(trial: optuna.Trial) -> float:
        """
        Addestra un modello con i parametri suggeriti da Optuna per questo trial
        e restituisce l'accuracy del modello addestrato sul set di validazione.
        """
        #se il classificatore scelto e' lightbgm
        if config.classifier == "lightgbm":
            #impostiamo i parametri per LightGBM
            params = _suggest_lightgbm_params(config, trial)
            #creiamo il classificatore LightGBM con i parametri suggeriti
            classifier = lgb.LGBMClassifier(**params)
            training_params = {
                "X": pixels_features, #feature di training
                "y": pixels_labels, #labels di training
                "eval_set": [(val_features, val_labels)], #set di validazione per early stopping
                "callbacks": [lgb.early_stopping(20), lgb.log_evaluation(0)], #early stopping e rimuove logs
                "sample_weight": pixels_weights, #pesi di ogni pixel
                #TODO: GIOCA CON LA LOSS FUNCTION PER DARE PRIORITà A CLASSE PICCOLE (io ho usato la multi-class log loss, o mlogloss)
                #prova: focal loss, prova weighted log loss
                #per i pesi del sample ho usato "sample_weight=pixels_weights" prova a cambiare che tipo di sample weight
                #LEGGITI LETTERATURA SUI PESI

                #--------------------------------
                #NO CLASS WEIGHTS SU IMMAGINE DOMINANTE, TOGLILO è UNA CAGATA <------FATTO
                #--------------------------------

                #TESI:
                #BUTTA GIU TESTO SU METODOLOGIA DI COSA HAI FATTO (SALTA INTRODUZIONE, PARTIAMO DA METà)
                #NON SCRIVERE PIU DI 5/6 PAGINE

                #UNA PARTE DA: UNA PAGINA MAX-> DOVE DICO COME ORGANIZZATO IL LAVORO: HO FATTO 2 MODELLI:, UNO CON CLASSIFICIAZIONE ed uno su segmetnazione
                # DUE PAGINE DI CLASSIFICAZIONE: SULLA BASE DEI PARAMETRI DI COTTURA. C'E' UN MODELLO PER OGNI PARAMETRO + QUELLO CHE CLASSIFCIA SUGLI ESPERIMENTI + IL FATTO CHE USO IL VIT e che lo addestri
                #-Con l'obbiettivo di fare predizione (classificazione) sulla base dei parametri di cottura. spiega che parametri sono, come erano fatti gli esperimenti ed il fatto che ci sia un modello per ciascuno di questi parametri che classifica per quel parametro + quello che classifica sul riconoscere il numero esperimento , dire che l'ha fatto con un vit


                #LE ALTRE 3 PAGINE: SU SEGMENTAZIONE. SPIEGARE OBBIETTIVO: SERVIVA PER SUPPORTARE L'ESSERE UMANO A CLASSIFICARLI. scrivere l'APRROCCIO UTILIZZATO -> IL NOSTRO è SUGLI EMBEDDING. prendiamo uno strato DELLA VGG. E SPIEGARE COME FUNZIONA LA SEGMENTAZIONE

               #NON PARLARE DI RISULTATI

                #IN UN MESETTO SCRIVO TUTTO (CONSIDERANDO CHE NEL FRATTEMPO FACCIO ALTRO
                #RICONTATTALO CON LE PAGINE AD INIZIO GENNAIO O A RIDOSSO DELLE FESTE.
                #SITO DIR DOVE IN TEORIA C'è SCRITTO COME SCRIVERE TESI
            }
            
            try:
                #addestriamo il classificatore
                classifier.fit(**training_params)
            except lgb.basic.LightGBMError as ex:
                #se l'addestramento da' problemi, controllo se era prevista la GPU e se si riprovo usando solo CPU
                if config.use_gpu:
                    warnings.warn(
                        f"LightGBM GPU non disponibile ({ex}). Riprovo in CPU.",
                        RuntimeWarning,
                    )
                    
                    params_cpu = params.copy()
                    params_cpu.pop("device", None)
                    classifier = lgb.LGBMClassifier(**params_cpu)
                    classifier.fit(**training_params)
                else:
                    #altrimenti fallisco
                    raise

            #calcolo le predizioni del classificatore addestrato sul set di validazione
            preds = classifier.predict(val_features)
            #salvo il classificatore addestrato come attriubto del trial (cosi' possiamo recuperalrlo alla fine se si rivelasse il modello migliore)
            trial.set_user_attr("estimator", classifier)
            #salviamo la best_iteration_ se disponibile per riutilizzarla nel training finale
            best_iter = getattr(classifier, "best_iteration_", None)
            trial.set_user_attr("best_iteration", int(best_iter) if best_iter is not None else None)
        
        #se il classificatroe scelto è XGBoost:
        else:
            #impostiamo i parametri per XGBoost
            params = _suggest_xgboost_params(config, trial)
            num_boost_round = params["n_estimators"]
            #converto i parametri suggeriti da Optuna in un formato accettabile a "xgb.train"
            train_params = _build_xgb_train_params(params, config)
            #creo la struttura dati (DMatrix) usata da XGBoost per il training
            train_dMatrix = xgb.DMatrix(
                data = pixels_features,
                label = pixels_labels,
                weight = pixels_weights,
            )
            #creaimo un a DMatrix di validazione da usare durante l'addestramento
            val_dMatrix = xgb.DMatrix(data = val_features, label=val_labels)
            try:
                #addestriamo il booster (classificatore) XGBoost
                trained_booster = xgb.train(
                    params=train_params,  # iperparametri (proposti da Optuna) e configurazione GPU/CPU
                    dtrain=train_dMatrix,  # dati di training (features, labels, pesi)
                    num_boost_round=num_boost_round,  # numero massimo di alberi
                    evals=[(val_dMatrix, "validation")],  # set di validazione per early stopping
                    early_stopping_rounds=20,
                    verbose_eval=False,  # log disabilitati
                )
            #se l'addestramento fallisce:
            except xgb.core.XGBoostError as ex:
                #se era prevista la GPU emette un warning e fa un fallback in CPU
                if config.use_gpu:
                    warnings.warn(
                        f"XGBoost GPU non disponibile ({ex}). Riprovo in CPU.",
                        RuntimeWarning,
                    )
                    cpu_params = train_params.copy()
                    cpu_params["tree_method"] = "hist"
                    cpu_params.pop("predictor", None)
                    cpu_params.pop("device", None)
                    trained_booster = xgb.train(
                        params=cpu_params,
                        dtrain=train_dMatrix,
                        num_boost_round=num_boost_round,
                        evals=[(val_dMatrix, "validation")],
                        early_stopping_rounds=20,
                        verbose_eval=False,
                    )
                else:
                    raise

            #Salvo l'iterazione migliore
            best_iteration = trained_booster.best_iteration
            #Salvo il booster addestrato come attriubto del trial (cosi' possiamo recuperalrlo dopo il tuning se si rivelasse il modello migliore)
            trial.set_user_attr("estimator", trained_booster)
            #salvo l'iterazione migliore come attributo del trial
            trial.set_user_attr("best_iteration", int(best_iteration) if best_iteration is not None else None)
            #calcolo le predizioni del booster addestrato sul set di validazione
            preds_raw = trained_booster.predict(val_dMatrix)
            #converto le predizioni probabilistiche di ogni classe in un array che salva solo la classe piu' probabile (es: [[0.1, 0.3, 0.6], [0.8, 0.1, 0.1], ...] -> [2, 0, 1, ...])
            preds = np.argmax(preds_raw, axis=1)

        #restituiamo l'f1 macro del trial (metrica robusta allo sbilanciamento delle classi)
        return f1_score(val_labels, preds, average="macro")


    
    #creiamo uno study optuna per l'ottimizzazione dei parametri (con direzione massimizzazione dell'f1 macro)
    study = optuna.create_study(direction="maximize")
    #eseguiamo l'ottimizzazione eseguendo train_and_evaluate_trial n_trials volte
    study.optimize(train_and_evaluate_trial, n_trials=n_trials)

    #recupero il miglior trial, i suoi parametri, la sua accuracy, il modello relativo e la sua miglior iterazione (salvati come attributo del trial)
    best_trial = study.best_trial
    best_params = study.best_params
    best_value = study.best_value
    stored_estimator = best_trial.user_attrs.get("estimator")
    best_iteration = best_trial.user_attrs.get("best_iteration")

    if stored_estimator is None:
        raise RuntimeError("Il best trial non contiene un estimatore addestrato.")

    # Riusa il modello addestrato nel best trial per non perdere l'early stopping.
    if config.classifier == "lightgbm":
        classifier = stored_estimator
    else:
        from gpu_optimized_cnn_classifier import XGBBoosterWrapper
        classifier = XGBBoosterWrapper(stored_estimator, config.num_classes)

    final_iteration = int(best_iteration) + 1 if best_iteration is not None else None

    #restituiamo f1 macro, il classifciatore, i migliori parametri trovati ed il numero di iterazioni
    return best_value, classifier, best_params, final_iteration


__all__ = ["tune_classifier"]
