#!/usr/bin/env python3
"""Modelo FAKE/REAL "UnivFD" (Ojha et al., CVPR 2023): CLIP ViT-L/14 congelado
+ una cabeza lineal.

Es el segundo de los dos detectores que reemplazan al baseline COCO. La idea de
UnivFD ("Universal Fake Detection") es que las features de un CLIP grande,
entrenado con muchísimas imágenes naturales, son un espacio donde fakes y reales
se separan linealmente SIN tocar el backbone: solo se entrena una capa lineal
sobre el embedding de 768-d. Esto lo hace muy robusto y rapidísimo de entrenar.

Arquitectura (idéntica al FakeImage original):
  - Encoder: CLIP ViT-L/14 (768-d), CONGELADO (no se entrena).
  - Cabeza : Linear(768 -> 1) + BCEWithLogitsLoss; sigmoid -> P(FAKE).
  - Preprocesado: el `clip_preprocess` propio de CLIP (Resize 224 + normalize
    con la media/std de CLIP, no la de ImageNet).

Este módulo es COMPARTIDO por el entrenamiento (`train_univfd_fakereal.py`) y la
inferencia (`models_infer.py`). El encoder CLIP es pesado (~1.7 GB en VRAM/RAM):
se carga una sola vez y se cachea por dispositivo.

REQUISITOS extra: el paquete `clip` (OpenAI CLIP). El logit de la cabeza es
P(FAKE): class_to_idx canónico {"FAKE": 0, "REAL": 1}.
"""

import torch
import torch.nn as nn

IMAGE_SIZE = 224          # CLIP ViT-L/14 usa 224x224
FEAT_DIM = 768            # dimensión del embedding de imagen de ViT-L/14
CLIP_ARCH = "ViT-L/14"

# Cache del encoder CLIP por dispositivo: (model, preprocess). CLIP es caro de
# cargar, así que entrenamiento e inferencia comparten una sola instancia.
_CLIP_CACHE = {}


def load_clip(device):
    """Carga (y cachea) el encoder CLIP ViT-L/14 congelado. Devuelve
    (clip_model, clip_preprocess). Lazy: solo importa `clip` al llamarse."""
    key = str(device)
    if key not in _CLIP_CACHE:
        import clip  # lazy: solo se necesita para UnivFD
        clip_model, clip_preprocess = clip.load(CLIP_ARCH, device=device)
        # clip.load deja el modelo en fp16 sobre CUDA. Lo pasamos a fp32: el
        # encoder está congelado (no entrena), y en fp16 los gradientes internos
        # del ViT son tan pequeños que el Grad-CAM colapsa por underflow (el mapa
        # sale todo a cero). fp32 los mantiene bien condicionados. Entrenamiento e
        # inferencia comparten este loader, así que las features son consistentes.
        clip_model = clip_model.float()
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        _CLIP_CACHE[key] = (clip_model, clip_preprocess)
    return _CLIP_CACHE[key]


def build_transform(device, augment: bool = False):
    """Devuelve el preprocesado de CLIP (PIL -> tensor (3, 224, 224)).

    El preprocesado de UnivFD ES el `clip_preprocess` propio (su Resize/crop y su
    normalización específica), no el de ImageNet. La augmentación del paper
    (Gaussian blur + JPEG, p=0.5) se aplica ANTES, en el Dataset del trainer; por
    eso aquí `augment` no cambia el preprocess final."""
    _, clip_preprocess = load_clip(device)
    return clip_preprocess


class UnivFDClassifier(nn.Module):
    """Cabeza lineal sobre el embedding CLIP: Linear(768 -> 1). El logit es
    P(FAKE) tras sigmoid (BCEWithLogitsLoss en entrenamiento)."""

    def __init__(self, feat_dim: int = FEAT_DIM):
        super().__init__()
        self.fc = nn.Linear(feat_dim, 1)

    def forward(self, x):
        return self.fc(x).squeeze(1)   # (B,)


def encode_images(clip_model, imgs):
    """Embedding CLIP (B, 768) en float32. `imgs` ya preprocesadas por CLIP."""
    feats = clip_model.encode_image(imgs)
    return feats.float()
