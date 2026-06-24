#!/usr/bin/env python3
"""Modelo FAKE/REAL con backbone Swin-B (Swin Transformer Base).

Es uno de los dos detectores que reemplazan al baseline COCO. A diferencia del
NPR (residuos de píxeles, alta frecuencia), Swin-B es un transformer jerárquico
pre-entrenado en ImageNet que mira estructura local-global con ventanas
desplazadas: capta artefactos de textura/forma que delatan generación. Es el
detector "fuerte" del trío de transformers del proyecto FakeImage original.

A diferencia de NPR:
  - Entrada RGB normal (3 canales), no residuos.
  - Usa Resize estándar (224x224); la señal sobrevive al re-muestreo.
  - El backbone parte de pesos ImageNet (Swin_B_Weights.IMAGENET1K_V1).

Este módulo es COMPARTIDO por el entrenamiento (`train_swin_fakereal.py`) y la
inferencia (`models_infer.py`) para garantizar misma arquitectura y preprocesado.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import swin_b, Swin_B_Weights

IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _base_transform(augment: bool) -> transforms.Compose:
    # Swin-B mira estructura semántica (no alta frecuencia como NPR), así que
    # Resize bilineal a 224x224 es seguro. En train añadimos flips + ColorJitter
    # suave como augmentación barata (igual que el entrenamiento original Swin-B).
    if augment:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_transform(augment: bool = False):
    """Preprocesado imagen PIL -> tensor RGB (3, 224, 224)."""
    return _base_transform(augment)


def build_swin_model(pretrained: bool = True, freeze_backbone: bool = False,
                     dropout: float = 0.0) -> nn.Module:
    """Swin-B con cabeza `head` a 2 clases (FAKE/REAL).

    Si `pretrained`, parte de pesos ImageNet (Swin_B_Weights.IMAGENET1K_V1).

    `freeze_backbone`: congela todo menos la cabeza `head` (linear probe). Útil
    para dataset pequeño; mide la señal de las features Swin sin que el backbone
    (~88M params) memorice. `dropout`>0 inserta un Dropout antes de `head` para
    regularizar la cabeza (con dropout>0 `head` pasa a ser un Sequential, así que
    la inferencia debe reconstruir el modelo con el MISMO dropout).
    """
    weights = Swin_B_Weights.IMAGENET1K_V1 if pretrained else None
    model = swin_b(weights=weights)

    in_features = model.head.in_features
    if dropout > 0.0:
        model.head = nn.Sequential(nn.Dropout(p=dropout),
                                   nn.Linear(in_features, 2))
    else:
        model.head = nn.Linear(in_features, 2)

    # Gradient checkpointing del backbone para bajar el pico de VRAM (Swin-B es
    # pesado). No afecta a la salida, solo recomputa activaciones en el backward.
    for layer in model.features:
        if hasattr(layer, "use_checkpoint"):
            layer.use_checkpoint = True

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.head.parameters():
            p.requires_grad = True

    return model


def gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Capa objetivo para Grad-CAM: la última etapa conv del Swin
    (`model.features[-1][-1]`), igual que en el FakeImage original. Sus
    activaciones son channel-last (B, H, W, C), así que la inferencia debe
    permutarlas a (C, H, W) antes de ponderar por gradientes."""
    return model.features[-1][-1]
