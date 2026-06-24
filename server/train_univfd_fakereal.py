#!/usr/bin/env python3
"""Entrena el detector FAKE/REAL "UnivFD": CLIP ViT-L/14 congelado + cabeza lineal.

Solo se entrena la capa Linear(768 -> 1) sobre el embedding CLIP (el backbone
queda congelado). Como CLIP no se toca, el entrenamiento es rapidísimo y casi no
sobreajusta aunque el dataset sea pequeño. Junto con Swin-B, reemplaza al COCO.

Para ir aún más rápido y estable, este trainer PRE-CALCULA los embeddings CLIP de
todo el dataset UNA vez (con la augmentación del paper aplicada a la vista de
train) y luego entrena la cabeza lineal sobre esos vectores. Es el enfoque
estándar de "linear probe" sobre features congeladas.

Uso (en WSL/Linux con torch+torchvision y el paquete `clip`):
    python server/train_univfd_fakereal.py
    python server/train_univfd_fakereal.py --epochs 50 --lr 5e-4

Estructura de dataset esperada (ImageFolder), insensible a mayúsculas:
    <data>/fake/*.png   (o FAKE/)
    <data>/real/*.png   (o REAL/)

El checkpoint guarda `model_state` (solo la cabeza), `class_to_idx`
({"FAKE": 0, "REAL": 1}) y `feat_dim`, para que `models_infer.py` reconstruya la
cabeza y la cargue directo. El logit es P(FAKE).
"""

import argparse
import io
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from PIL import Image

from univfd_model import UnivFDClassifier, load_clip, FEAT_DIM

# ===========================================================================
# HIPERPARÁMETROS  (edítalos aquí; cualquiera se puede sobreescribir por CLI)
# ===========================================================================
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Datos y rutas ---
DATA_DIR   = _PROJECT_ROOT / "dataset_new_model" / "train"  # split de train (excluye las 100 del experimento)
OUT_CKPT   = _PROJECT_ROOT / "server" / "univfd_fakereal.pth"  # checkpoint de salida

# --- Entrenamiento ---
# Solo se entrena la cabeza lineal sobre features CLIP congeladas: no hay riesgo
# real de overfitting del backbone, así que podemos usar LR alto y muchas épocas.
# El early stopping sobre val_acc elige el mejor punto.
EPOCHS       = 60       # tope; el early stopping suele cortar antes
BATCH_SIZE   = 32
LR           = 1e-3     # LR alto: solo entrenamos una capa lineal
WEIGHT_DECAY = 1e-3
VAL_SPLIT    = 0.15
PATIENCE     = 12       # early stopping (0=off); paciencia alta con LR alto
AUGMENT      = True     # augmentación del paper (Gaussian blur + JPEG) en train
SEED         = 42

# class_to_idx canónico; el logit de la cabeza es P(FAKE).
CLASS_TO_IDX = {"FAKE": 0, "REAL": 1}


def parse_args():
    p = argparse.ArgumentParser(description="Entrena FAKE/REAL (UnivFD: CLIP + lineal).")
    p.add_argument("--data", type=Path, default=DATA_DIR,
                   help="Carpeta con subcarpetas fake/ y real/.")
    p.add_argument("--out", type=Path, default=OUT_CKPT,
                   help="Ruta del checkpoint de salida.")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--val-split", type=float, default=VAL_SPLIT)
    p.add_argument("--patience", type=int, default=PATIENCE,
                   help="Early stopping: épocas sin mejora antes de parar (0=off).")
    p.add_argument("--no-augment", dest="augment", action="store_false",
                   default=AUGMENT, help="Desactiva blur+JPEG en train.")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


class RandomJPEG:
    """Recomprime a JPEG con calidad aleatoria (augmentación del paper UnivFD)."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() < self.p:
            quality = int(torch.randint(30, 95, (1,)).item())
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        return img


def remap_label(class_to_idx_folder, orig_label):
    """Mapea la etiqueta de ImageFolder a CLASS_TO_IDX por nombre (case-insensitive)."""
    idx_to_name = {v: k for k, v in class_to_idx_folder.items()}
    key = idx_to_name[orig_label].strip().upper()
    if key not in CLASS_TO_IDX:
        raise ValueError(f"Carpeta de clase no reconocida: '{key}'. Se esperan FAKE/REAL.")
    return CLASS_TO_IDX[key]


@torch.no_grad()
def precompute_features(folder, clip_model, clip_preprocess, device, augment, seed):
    """Codifica TODO el dataset a embeddings CLIP (B, 768). Devuelve
    (feats float32 CPU, labels long CPU) ya remapeados a CLASS_TO_IDX.

    Para train aplica blur+JPEG (augmentación del paper) ANTES del preprocess de
    CLIP; para val no. Las stats de label se imprimen para detectar desbalance."""
    ds = datasets.ImageFolder(str(folder))
    jpeg = RandomJPEG(p=0.5)
    blur = transforms.RandomApply(
        [transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))], p=0.5)

    feats_all, labels_all = [], []
    n = len(ds.samples)
    for i, (path, orig_label) in enumerate(ds.samples, 1):
        img = ds.loader(path)
        if augment:
            img = jpeg(img)
            img = blur(img)
        x = clip_preprocess(img).unsqueeze(0).to(device)
        feat = clip_model.encode_image(x).float().squeeze(0).cpu()
        feats_all.append(feat)
        labels_all.append(remap_label(ds.class_to_idx, orig_label))
        print(f"\r  Codificando CLIP [{i}/{n}]", end="", flush=True)
    print()
    return torch.stack(feats_all), torch.tensor(labels_all, dtype=torch.long)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print("=" * 60)
    if use_cuda:
        props = torch.cuda.get_device_properties(0)
        print(f"  ✅ GPU DETECTADA -> entrenando en CUDA")
        print(f"  GPU            : {props.name}")
        print(f"  VRAM total     : {props.total_memory / 1e9:.1f} GB")
        print(f"  torch / cuda   : {torch.__version__} / {torch.version.cuda}")
    else:
        print(f"  ⚠️  NO HAY GPU -> entrenando en CPU (será más lento)")
        print(f"  torch          : {torch.__version__} (sin soporte CUDA)")
    print("=" * 60)
    if not args.data.exists():
        raise FileNotFoundError(f"No existe el dataset: {args.data}")

    print(f"Cargando encoder CLIP {('ViT-L/14')} (congelado)...")
    clip_model, clip_preprocess = load_clip(device)

    # Pre-calcula los embeddings CLIP de TODO el dataset (una vista de train con
    # augmentación y otra de val sin ella), luego entrena solo la cabeza lineal.
    print("Pre-calculando embeddings CLIP (vista TRAIN, con augmentación)...")
    feats_t, labels_t = precompute_features(
        args.data, clip_model, clip_preprocess, device, args.augment, args.seed)
    print("Pre-calculando embeddings CLIP (vista VAL, sin augmentación)...")
    feats_v, labels_v = precompute_features(
        args.data, clip_model, clip_preprocess, device, False, args.seed)

    n_total = feats_t.size(0)
    n_val = max(1, int(round(n_total * args.val_split)))
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n_total, generator=gen)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    # Train usa la vista augmentada; val usa la vista limpia (mismos índices).
    train_ds = TensorDataset(feats_t[train_idx], labels_t[train_idx])
    val_ds = TensorDataset(feats_v[val_idx], labels_v[val_idx])

    print(f"class_to_idx: {CLASS_TO_IDX} (logit = P(FAKE))")
    print(f"Total: {n_total} | Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    classifier = UnivFDClassifier(feat_dim=FEAT_DIM).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss()

    fake_idx = CLASS_TO_IDX["FAKE"]

    best_val_acc = -1.0
    epochs_no_improve = 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        classifier.train()
        total_loss = correct = total = 0
        for feats, labels in train_loader:
            feats = feats.to(device)
            # Target BCE: 1 si FAKE, 0 si REAL (el logit es P(FAKE)).
            y = (labels == fake_idx).float().to(device)
            optimizer.zero_grad()
            logits = classifier(feats)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * feats.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
            total += feats.size(0)
        scheduler.step()

        train_loss = total_loss / total
        train_acc = correct / total

        # Validación
        classifier.eval()
        v_correct = v_total = 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats = feats.to(device)
                y = (labels == fake_idx).float().to(device)
                logits = classifier(feats)
                preds = (torch.sigmoid(logits) > 0.5).float()
                v_correct += (preds == y).sum().item()
                v_total += feats.size(0)
        val_acc = v_correct / max(1, v_total)

        cur_lr = optimizer.param_groups[-1]["lr"]
        print(f"  Epoch {epoch:02d}/{args.epochs} | loss {train_loss:.4f} | "
              f"acc {train_acc:.4f} | val_acc {val_acc:.4f} | lr {cur_lr:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state": classifier.state_dict(),
                "class_to_idx": CLASS_TO_IDX,
                "val_acc": val_acc,
                "train_loss": train_loss,
                "arch": "univfd_clip_vitl14_fakereal",
                "feat_dim": FEAT_DIM,
            }, args.out)
            print(f"  -> Mejor checkpoint guardado (val_acc={val_acc:.4f}) en {args.out}")
        else:
            epochs_no_improve += 1
            if args.patience > 0 and epochs_no_improve >= args.patience:
                print(f"  -> Early stopping: {epochs_no_improve} épocas sin mejorar "
                      f"(mejor val_acc={best_val_acc:.4f}).")
                break

    total_mm, total_ss = divmod(int(time.time() - t_start), 60)
    print(f"\nEntrenamiento completo. Mejor val_acc: {best_val_acc:.4f}")
    print(f"Tiempo total: {total_mm}m {total_ss}s")
    print(f"Checkpoint: {args.out}")


if __name__ == "__main__":
    main()
