#!/usr/bin/env python3
"""Entrena el modelo FAKE/REAL con backbone Swin-B (Swin Transformer Base).

Swin-B (~88M params) parte de pesos ImageNet y se fine-tunea como clasificador
FAKE/REAL sobre `dataset_new_model/`. Es el detector "transformer fuerte" que,
junto con UnivFD, reemplaza al baseline COCO.

Uso (en WSL/Linux con torch+torchvision):
    python server/train_swin_fakereal.py
    python server/train_swin_fakereal.py --epochs 20 --lr 1e-5
    python server/train_swin_fakereal.py --freeze-backbone   # linear probe

Estructura de dataset esperada (ImageFolder), insensible a mayúsculas:
    <data>/fake/*.png   (o FAKE/)
    <data>/real/*.png   (o REAL/)

El checkpoint resultante guarda `model_state`, `class_to_idx`
({"FAKE": 0, "REAL": 1}) y la config de la cabeza (`dropout`/`freeze_backbone`)
para que `models_infer.py` reconstruya el modelo idéntico y lo cargue directo.
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets

from swin_model import build_swin_model, build_transform

# ===========================================================================
# HIPERPARÁMETROS  (edítalos aquí; cualquiera se puede sobreescribir por CLI)
# ===========================================================================
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Datos y rutas ---
DATA_DIR   = _PROJECT_ROOT / "dataset_new_model" / "train"  # split de train (excluye las 100 del experimento)
OUT_CKPT   = _PROJECT_ROOT / "server" / "swin_fakereal.pth"  # checkpoint de salida
FROM_SCRATCH = False   # True: backbone aleatorio (sin pesos ImageNet)

# --- Entrenamiento ---
# Swin-B es enorme (~88M params) y el dataset es pequeño (~660 imgs): riesgo alto
# de overfitting. Config por defecto = fine-tuning COMPLETO pero MUY conservador:
#   - LR bajísimo (3e-5, como el Swin-B original) para no destruir las features.
#   - WEIGHT_DECAY alto (AdamW) + LABEL_SMOOTHING + DROPOUT regularizan.
#   - Early stopping con val_acc para cortar en cuanto empieza a memorizar.
# Alternativa: --freeze-backbone (linear probe) si el fine-tuning sobreajusta.
FREEZE_BACKBONE  = False  # default: fine-tuning completo con LR bajo
DROPOUT          = 0.2    # dropout antes de head (0=off)
LABEL_SMOOTHING  = 0.1    # suaviza etiquetas en CrossEntropy (0=off)

EPOCHS       = 25       # tope; el early stopping suele cortar antes
BATCH_SIZE   = 8        # Swin-B es pesado en VRAM
OPTIMIZER    = "adamw"  # "adamw", "adam" o "sgd"
LR           = 3e-5     # fine-tuning completo: LR bajo (como el Swin-B original)
WEIGHT_DECAY = 1e-2     # AdamW: weight decay alto regulariza el transformer
VAL_SPLIT    = 0.15
PATIENCE     = 6        # early stopping (0=off)
NUM_WORKERS  = 0
SEED         = 42

# class_to_idx canónico, consistente con los demás modelos.
CLASS_TO_IDX = {"FAKE": 0, "REAL": 1}


def parse_args():
    p = argparse.ArgumentParser(description="Entrena FAKE/REAL (backbone Swin-B).")
    p.add_argument("--data", type=Path, default=DATA_DIR,
                   help="Carpeta con subcarpetas fake/ y real/.")
    p.add_argument("--out", type=Path, default=OUT_CKPT,
                   help="Ruta del checkpoint de salida.")
    p.add_argument("--from-scratch", action="store_true", default=FROM_SCRATCH,
                   help="Backbone aleatorio (sin pesos ImageNet).")
    p.add_argument("--freeze-backbone", dest="freeze_backbone",
                   action="store_true", default=FREEZE_BACKBONE,
                   help="Linear probe: entrena solo la cabeza head.")
    p.add_argument("--no-freeze-backbone", dest="freeze_backbone",
                   action="store_false",
                   help="Fine-tuning completo del backbone (default).")
    p.add_argument("--dropout", type=float, default=DROPOUT,
                   help="Dropout antes de head (0=off).")
    p.add_argument("--label-smoothing", type=float, default=LABEL_SMOOTHING,
                   help="Label smoothing en CrossEntropy (0=off).")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--optimizer", choices=["adamw", "adam", "sgd"], default=OPTIMIZER)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--val-split", type=float, default=VAL_SPLIT)
    p.add_argument("--patience", type=int, default=PATIENCE,
                   help="Early stopping: épocas sin mejora antes de parar (0=off).")
    p.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


class RemappedFolder(Dataset):
    """ImageFolder cuyo etiquetado se fuerza a CLASS_TO_IDX por nombre de clase
    (insensible a mayúsculas): 'fake'/'FAKE'->0 y 'real'/'REAL'->1, sin depender
    del orden alfabético que asigna ImageFolder."""

    def __init__(self, root: Path, transform):
        self.ds = datasets.ImageFolder(str(root), transform=None)
        self.transform = transform
        self.remap = {}
        for cls_name, orig_idx in self.ds.class_to_idx.items():
            key = cls_name.strip().upper()
            if key not in CLASS_TO_IDX:
                raise ValueError(
                    f"Carpeta de clase no reconocida: '{cls_name}'. "
                    f"Se esperan 'fake'/'real' (o FAKE/REAL)."
                )
            self.remap[orig_idx] = CLASS_TO_IDX[key]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        path, orig_label = self.ds.samples[idx]
        img = self.ds.loader(path)
        return self.transform(img), self.remap[orig_label]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # ---- Banner de dispositivo: dejar MUY claro si entrena en GPU o CPU ----
    print("=" * 60)
    if use_cuda:
        props = torch.cuda.get_device_properties(0)
        print(f"  ✅ GPU DETECTADA -> entrenando en CUDA")
        print(f"  GPU            : {props.name}")
        print(f"  VRAM total     : {props.total_memory / 1e9:.1f} GB")
        print(f"  CUDA capability: {props.major}.{props.minor}")
        print(f"  torch / cuda   : {torch.__version__} / {torch.version.cuda}")
    else:
        print(f"  ⚠️  NO HAY GPU -> entrenando en CPU (será mucho más lento)")
        print(f"  torch          : {torch.__version__} (sin soporte CUDA)")
    print("=" * 60)
    if not args.data.exists():
        raise FileNotFoundError(f"No existe el dataset: {args.data}")

    # Dos vistas del mismo folder: train con augmentación, val sin ella.
    full_train = RemappedFolder(args.data, transform=build_transform(augment=True))
    full_val = RemappedFolder(args.data, transform=build_transform(augment=False))
    n_total = len(full_train)
    n_val = max(1, int(round(n_total * args.val_split)))
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(args.seed)
    train_idx, val_idx = random_split(range(n_total), [n_train, n_val], generator=gen)

    train_ds = torch.utils.data.Subset(full_train, list(train_idx))
    val_ds = torch.utils.data.Subset(full_val, list(val_idx))

    print(f"class_to_idx: {CLASS_TO_IDX}")
    print(f"Total: {n_total} | Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # ---- Modelo: backbone ImageNet (o aleatorio con --from-scratch) ----
    if args.from_scratch:
        print("Backbone Swin-B aleatorio (sin pesos ImageNet).")
        model = build_swin_model(pretrained=False,
                                 freeze_backbone=args.freeze_backbone,
                                 dropout=args.dropout)
    else:
        print("Inicializando Swin-B con pesos ImageNet (Swin_B_Weights.IMAGENET1K_V1).")
        model = build_swin_model(pretrained=True,
                                 freeze_backbone=args.freeze_backbone,
                                 dropout=args.dropout)
    model = model.to(device)

    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total_params = sum(p.numel() for p in model.parameters())
    mode = "LINEAR PROBE (solo head)" if args.freeze_backbone else "fine-tuning completo"
    print(f"Modo: {mode} | entrenables: {n_train_params:,} / {n_total_params:,} "
          f"| dropout: {args.dropout} | label_smoothing: {args.label_smoothing}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(trainable, lr=args.lr,
                                      weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(trainable, lr=args.lr,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(trainable, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    print(f"Optimizador: {args.optimizer} | LR: {args.lr}")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best_val_acc = -1.0
    epochs_no_improve = 0
    t_start = time.time()
    n_batches = len(train_loader)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()

        model.train()
        if args.freeze_backbone:
            # Backbone congelado: eval() para CONGELAR las stats de norm; solo la
            # cabeza head queda en modo train.
            model.eval()
            model.head.train()
        total_loss = correct = total = 0
        for batch_idx, (imgs, labels) in enumerate(train_loader, 1):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += imgs.size(0)

            # Progreso EN LA MISMA LÍNEA: loss/acc acumulados + velocidad y ETA.
            elapsed = time.time() - t0
            ips = total / elapsed if elapsed > 0 else 0.0
            eta = (n_batches - batch_idx) * (elapsed / batch_idx) if batch_idx else 0.0
            print(
                f"\r  Epoch {epoch:02d}/{args.epochs} "
                f"[{batch_idx:>{len(str(n_batches))}}/{n_batches}] "
                f"loss {total_loss / total:.4f} | acc {correct / total:.4f} | "
                f"{ips:5.1f} img/s | {elapsed:4.0f}s | ETA {eta:4.0f}s",
                end="", flush=True,
            )
        scheduler.step()

        train_loss = total_loss / total
        train_acc = correct / total

        # Validación
        model.eval()
        v_correct = v_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                v_correct += (logits.argmax(dim=1) == labels).sum().item()
                v_total += imgs.size(0)
        val_acc = v_correct / max(1, v_total)

        epoch_secs = time.time() - t0
        cur_lr = optimizer.param_groups[-1]["lr"]
        vram = (f" | VRAM pico {torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
                if use_cuda else "")
        print(
            f"\r  Epoch {epoch:02d}/{args.epochs} | "
            f"loss {train_loss:.4f} | acc {train_acc:.4f} | "
            f"val_acc {val_acc:.4f} | lr {cur_lr:.2e} | "
            f"{epoch_secs:.0f}s{vram}"
            + " " * 8
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "class_to_idx": CLASS_TO_IDX,
                "val_acc": val_acc,
                "train_loss": train_loss,
                "arch": "swin_b_fakereal",
                # Config de la cabeza: la inferencia reconstruye el modelo con el
                # MISMO dropout para que las claves del state_dict casen (con
                # dropout>0, head es un Sequential -> claves 'head.1.*').
                "dropout": args.dropout,
                "freeze_backbone": args.freeze_backbone,
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
