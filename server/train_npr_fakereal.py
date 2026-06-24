#!/usr/bin/env python3
"""Fine-tuning del modelo único FAKE/REAL (NPR) sobre un dataset nuevo.

Usa la arquitectura NPR (ResNet-50 sobre residuos de píxeles vecinos), diseñada
para detectar artefactos de generación/manipulación. Por defecto parte del
checkpoint NPR ya entrenado (`npr_fakereal.pth` en FakeImage) y lo fine-tunea con
`dataset_new_model/`; con --from-imagenet empieza solo desde pesos ImageNet.

Uso (en WSL/Linux con torch+torchvision):
    python server/train_npr_fakereal.py
    python server/train_npr_fakereal.py --epochs 20 --lr 1e-4
    python server/train_npr_fakereal.py --from-imagenet   # sin checkpoint base

Estructura de dataset esperada (ImageFolder), insensible a mayúsculas:
    <data>/fake/*.png   (o FAKE/)
    <data>/real/*.png   (o REAL/)

El checkpoint resultante guarda `model_state` y `class_to_idx`
({"FAKE": 0, "REAL": 1}), igual que los modelos anteriores, para que
`models_infer.py` lo cargue directamente.
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets

from npr_model import build_npr_model, build_transform

# ===========================================================================
# HIPERPARÁMETROS  (edítalos aquí; cualquiera se puede sobreescribir por CLI)
# ===========================================================================
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Datos y rutas ---
DATA_DIR   = _PROJECT_ROOT / "dataset_new_model" / "train"  # split de train (excluye las 100 del experimento)
OUT_CKPT   = _PROJECT_ROOT / "server" / "npr_fakereal.pth"  # checkpoint de salida (lo consume el server)
BASE_CKPT  = Path("/mnt/c/Users/mitsu/Desktop/FakeImage/npr_fakereal.pth")  # base a fine-tunear (si FROM_IMAGENET=False)
FROM_IMAGENET = True     # parte de pesos ImageNet (conv1 adaptada a 6 canales NPR); ignora BASE_CKPT

# --- Entrenamiento ---
# Config ANTI-OVERFITTING para un dataset pequeño (~560 train) con señal débil:
# LR muy bajo + pocas épocas + early stopping evitan que el ResNet (23M params)
# memorice el train (train acc ~1.0) en vez de aprender la señal generalizable.
EPOCHS       = 30       # tope; el early stopping suele cortar antes
BATCH_SIZE   = 16
OPTIMIZER    = "adam"    # "adam" o "sgd"
LR           = 5e-5      # fine-tuning desde el ckpt base; con crop la señal NPR
                         # sobrevive, así que el modelo puede aprender de verdad
                         # (con resize+lr 1e-5 antes solo memorizaba en 5 épocas)
WEIGHT_DECAY = 1e-3      # regularización fuerte (dataset pequeño)
VAL_SPLIT    = 0.15      # fracción de datos para validación
PATIENCE     = 6         # early stopping: épocas sin mejora de val_acc antes de parar (0=off)
NUM_WORKERS  = 0
SEED         = 42

# class_to_idx canónico, consistente con los modelos anteriores.
CLASS_TO_IDX = {"FAKE": 0, "REAL": 1}


def parse_args():
    # Los defaults vienen del bloque HIPERPARÁMETROS de arriba; el CLI permite
    # sobreescribir cualquiera puntualmente sin editar el archivo.
    p = argparse.ArgumentParser(description="Fine-tuning FAKE/REAL (NPR).")
    p.add_argument("--data", type=Path, default=DATA_DIR,
                   help="Carpeta con subcarpetas fake/ y real/.")
    p.add_argument("--out", type=Path, default=OUT_CKPT,
                   help="Ruta del checkpoint de salida.")
    p.add_argument("--base-ckpt", type=Path, default=BASE_CKPT,
                   help="Checkpoint NPR base del que partir el fine-tuning.")
    p.add_argument("--from-imagenet", action="store_true", default=FROM_IMAGENET,
                   help="Ignora --base-ckpt y parte solo de pesos ImageNet.")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--optimizer", choices=["adam", "sgd"], default=OPTIMIZER)
    p.add_argument("--lr", type=float, default=LR,
                   help="LR (bajo para fine-tuning desde un checkpoint).")
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

    # ---- Modelo: parte del checkpoint NPR base, o de ImageNet ----
    if args.from_imagenet:
        print("Partiendo de pesos ImageNet (conv1 adaptada a 6 canales NPR).")
        model = build_npr_model(pretrained=True)
    elif args.base_ckpt.exists():
        print(f"Fine-tuning desde checkpoint base: {args.base_ckpt}")
        model = build_npr_model(pretrained=False)
        base = torch.load(str(args.base_ckpt), map_location="cpu")
        model.load_state_dict(base["model_state"])
    else:
        print(f"⚠️  No se encontró --base-ckpt ({args.base_ckpt}); "
              f"partiendo de ImageNet.")
        model = build_npr_model(pretrained=True)
    model = model.to(device)

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    print(f"Optimizador: {args.optimizer} | LR: {args.lr}")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    epochs_no_improve = 0
    t_start = time.time()
    n_batches = len(train_loader)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()

        model.train()
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
                "arch": "npr_resnet50_fakereal",
            }, args.out)
            print(f"  -> Mejor checkpoint guardado (val_acc={val_acc:.4f}) en {args.out}")
        else:
            epochs_no_improve += 1
            # Early stopping: si val_acc no mejora en `patience` épocas, paramos.
            # Evita seguir entrenando mientras el modelo solo memoriza el train.
            if args.patience > 0 and epochs_no_improve >= args.patience:
                print(f"  -> Early stopping: {epochs_no_improve} épocas sin mejorar "
                      f"(mejor val_acc={best_val_acc:.4f}).")
                break

    total_mm, total_ss = divmod(int(time.time() - t_start), 60)
    print(f"\nFine-tuning completo. Mejor val_acc: {best_val_acc:.4f}")
    print(f"Tiempo total: {total_mm}m {total_ss}s")
    print(f"Checkpoint: {args.out}")


if __name__ == "__main__":
    main()
