#!/usr/bin/env python3
"""Entrena el clasificador SIMPLE FAKE/REAL (regresión logística sobre 6 features).

No usa torch ni GPU: extrae features con numpy/PIL y ajusta una regresión
logística por descenso de gradiente. Es rápido (segundos) y el checkpoint es un
JSON diminuto (pesos + normalización).

Uso:
    python server/train_simple_fakereal.py
    python server/train_simple_fakereal.py --data /ruta --out /ruta/simple.json

Estructura esperada (insensible a mayúsculas):
    <data>/fake/*   y   <data>/real/*
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from simple_model import extract_features, FEATURE_NAMES

CLASS_TO_IDX = {"FAKE": 0, "REAL": 1}
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- HIPERPARÁMETROS ---
DATA_DIR  = _PROJECT_ROOT / "dataset_new_model" / "train"  # split de train (excluye las 100 del experimento)
OUT_CKPT  = _PROJECT_ROOT / "server" / "simple_fakereal.json"
VAL_SPLIT = 0.15
LR        = 0.1
ITERS     = 5000
L2        = 1e-3     # regularización (weight decay) de la regresión
SEED      = 42


def parse_args():
    p = argparse.ArgumentParser(description="Entrena clasificador SIMPLE FAKE/REAL.")
    p.add_argument("--data", type=Path, default=DATA_DIR)
    p.add_argument("--out", type=Path, default=OUT_CKPT)
    p.add_argument("--val-split", type=float, default=VAL_SPLIT)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--iters", type=int, default=ITERS)
    p.add_argument("--l2", type=float, default=L2)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def load_dataset(data_dir):
    X, y = [], []
    for sub in sorted(data_dir.iterdir()):
        if not sub.is_dir():
            continue
        key = sub.name.strip().upper()
        if key not in CLASS_TO_IDX:
            continue
        label = CLASS_TO_IDX[key]
        files = [f for f in sorted(sub.iterdir())
                 if f.suffix.lower() in IMG_EXTS]
        print(f"  {sub.name}: {len(files)} imágenes")
        for f in files:
            X.append(extract_features(Image.open(f)))
            y.append(label)
    return np.array(X), np.array(y)


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)
    if not args.data.exists():
        raise FileNotFoundError(f"No existe el dataset: {args.data}")

    print(f"Extrayendo features de {args.data} ...")
    t0 = time.time()
    X, y = load_dataset(args.data)
    print(f"Total: {len(X)} imágenes | features: {FEATURE_NAMES} "
          f"| ({time.time() - t0:.1f}s)")

    # Split train/val
    idx = rng.permutation(len(X))
    n_val = max(1, round(len(X) * args.val_split))
    val, tr = idx[:n_val], idx[n_val:]
    print(f"Train: {len(tr)} | Val: {len(val)}")

    # Normalización con stats de train
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Xn = (X - mu) / sd

    # Regresión logística por descenso de gradiente (con L2)
    w = np.zeros(Xn.shape[1])
    b = 0.0
    for it in range(args.iters):
        z = Xn[tr] @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        gw = Xn[tr].T @ (p - y[tr]) / len(tr) + args.l2 * w
        gb = (p - y[tr]).mean()
        w -= args.lr * gw
        b -= args.lr * gb

    def acc(ii):
        p = 1.0 / (1.0 + np.exp(-(Xn[ii] @ w + b)))
        return float(((p > 0.5).astype(int) == y[ii]).mean())

    train_acc, val_acc = acc(tr), acc(val)
    print(f"\nResultado:  train_acc = {train_acc:.4f} | val_acc = {val_acc:.4f}")
    print("Pesos por feature:")
    for name, wi in zip(FEATURE_NAMES, w):
        print(f"  {name:12} {wi:+.4f}")

    # Guarda el checkpoint (JSON pequeño)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "w": w.tolist(),
        "b": float(b),
        "mu": mu.tolist(),
        "sd": sd.tolist(),
        "class_to_idx": CLASS_TO_IDX,
        "feature_names": FEATURE_NAMES,
        "val_acc": val_acc,
        "train_acc": train_acc,
        "arch": "simple_logreg_fakereal",
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, indent=2)
    print(f"\nCheckpoint guardado en: {args.out}")


if __name__ == "__main__":
    main()
