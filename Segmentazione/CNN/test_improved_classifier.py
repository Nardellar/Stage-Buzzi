import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image
import cv2 # Necessario per l'upsampling delle probabilità
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian

# Importa il modello CNN e il Dataset
from improved_cnn_classifier import VGGFeatureExtractor
from Dataset import SegmentationDataset 

# --- PARAMETRI ---
TEST_IMG_DIR = 'data/val/images'
TEST_MASK_DIR = 'data/val/masks'
RF_MODEL_PATH = 'rf_classifier_vgg_features.joblib'

# ATTENZIONE: Il CRF è lento e processa un'immagine alla volta
BATCH_SIZE = 1 
IMAGE_SIZE = 256
FEATURE_MAP_SIZE = IMAGE_SIZE // 32
# -----------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Funzione Helper per il CRF ---
# Questa funzione applica l'algoritmo CRF
def apply_crf(image, probs, n_classes):
    """
    Applica il CRF denso.
    :param image: Immagine originale (H, W, 3) - np.uint8
    :param probs: Probabilità dall'upsampling (H, W, N_CLASSES) - np.float32
    :return: Maschera predetta (H, W) - np.int
    """
    H, W = image.shape[:2]
    
    # 1. Converte le probabilità in "unary potentials"
    # pydensecrf vuole (N_CLASSES, H*W)
    unary = unary_from_softmax(probs.transpose(2, 0, 1).reshape(n_classes, -1))
    
    d = dcrf.DenseCRF2D(W, H, n_classes)
    d.setUnaryEnergy(unary)

    # 2. Add Pairwise Bilateral (termini di apparenza e prossimità)
    # "Pixel vicini con colori simili dovrebbero avere la stessa etichetta"
    pairwise_bilateral = create_pairwise_bilateral(
        sdims=(80, 80),  # std dev spaziale
        schan=(13, 13, 13), # std dev colore
        img=np.ascontiguousarray(image), # Immagine originale
        chdim=2
    )
    d.addPairwiseEnergy(pairwise_bilateral, compat=10) # Peso

    # 3. Add Pairwise Gaussian (termini di prossimità)
    # "Pixel vicini dovrebbero avere la stessa etichetta"
    pairwise_gaussian = create_pairwise_gaussian(
        sdims=(3, 3), # std dev spaziale
        shape=(H, W),
        chdim=-1
    )
    d.addPairwiseEnergy(pairwise_gaussian, compat=3) # Peso
    
    # 4. Esegui inferenza
    Q = d.inference(5) # 5 iterazioni
    
    # 5. Ottieni la mappa finale
    map_result = np.argmax(Q, axis=0).reshape((H, W))
    return map_result

# -------------------------------

# 1. Carica il Dataset di Test
# Trasformazioni per VGG (Normalizzate)
image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])
# Trasformazioni per Maschere (Ground Truth)
mask_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), 
                      interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor()
])

try:
    test_dataset = SegmentationDataset(
        TEST_IMG_DIR, TEST_MASK_DIR, 
        image_transform=image_transform, 
        mask_transform=mask_transform
    )
    # IMPORTANTE: batch_size=1
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Dataset di test caricato con {len(test_dataset)} immagini.")
    
    # Ci serve l'elenco dei percorsi delle immagini per caricarle
    # non normalizzate per il CRF
    image_paths = test_dataset.image_paths
    
except Exception as e:
    print(f"Errore caricamento dati: {e}. Controlla 'Dataset.py'.")
    exit()

# 2. Carica i Modelli
cnn_model = VGGFeatureExtractor(freeze_weights=True).to(device)
cnn_model.eval()
print("Modello CNN (VGG16) caricato.")

rf_classifier = joblib.load(RF_MODEL_PATH)
print(f"Modello Random Forest caricato da {RF_MODEL_PATH}.")
    
# 3. Valutazione con CRF
print("Inizio valutazione sul set di test (con CRF)...")
all_preds_crf = []
all_labels_full = []
n_classes = rf_classifier.n_classes_

with torch.no_grad():
    # Usiamo 'enumerate' per tracciare l'indice 'i'
    for i, (images_norm, masks_full_tensor) in enumerate(tqdm(test_loader)):
        
        images_norm = images_norm.to(device) # [1, 3, 256, 256]
        
        # Ground truth (maschera completa)
        labels_full = masks_full_tensor.squeeze().cpu().numpy().astype(int) # [256, 256]

        # --- FASE 1 & 2: CNN -> RF ---
        
        # 1. Estrai feature con la CNN
        features = cnn_model(images_norm) # [1, 512, 8, 8]
        
        # 2. Prepara feature per RF
        features_reshaped = features.permute(0, 2, 3, 1).reshape(-1, 512).cpu().numpy()
        
        # 3. Ottieni le PROBABILITA' dal Random Forest
        # probs_small -> [64, N_CLASSES] (64 = 8*8)
        probs_small = rf_classifier.predict_proba(features_reshaped)
        
        # --- FASE 3: CRF ---
        
        # 4. Rimodella e fai l'Upsample delle probabilità
        # [64, C] -> [8, 8, C]
        probs_map_small = probs_small.reshape(FEATURE_MAP_SIZE, FEATURE_MAP_SIZE, n_classes)
        
        # [8, 8, C] -> [256, 256, C]
        # Usiamo cv2.resize per un upsampling interpolato
        probs_map_full = cv2.resize(
            probs_map_small, 
            (IMAGE_SIZE, IMAGE_SIZE), 
            interpolation=cv2.INTER_LINEAR # Interpolazione lineare è la migliore per le probabilità
        ).astype(np.float32)

        # 5. Carica l'immagine originale (NON normalizzata) per il CRF
        original_img_path = image_paths[i]
        original_img = Image.open(original_img_path).resize((IMAGE_SIZE, IMAGE_SIZE))
        original_img_np = np.array(original_img).astype(np.uint8) # [256, 256, 3]

        # 6. Applica il CRF
        crf_pred = apply_crf(original_img_np, probs_map_full, n_classes) # [256, 256]
        
        all_preds_crf.append(crf_pred)
        all_labels_full.append(labels_full)

print("Valutazione completata.")

# 4. Calcolo Metriche
# Ora stiamo confrontando le maschere 256x256
y_pred = np.concatenate(all_preds_crf).ravel()
y_true = np.concatenate(all_labels_full).ravel()

accuracy = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred)

print("\n--- Risultati della Valutazione (con CRF) ---")
print(f"Accuratezza (per pixel su maschera {IMAGE_SIZE}x{IMAGE_SIZE}): {accuracy:.4f}")
print("\nClassification Report:")
print(report)