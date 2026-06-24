#!/usr/bin/env python3
"""Modelo FAKE/REAL con backbone pre-entrenado en COCO (detección de objetos).

Es el contrapunto del NPR. Mientras NPR mira residuos de píxeles vecinos (alta
frecuencia) para cazar artefactos de generación, este modelo usa el backbone
ResNet-50 de un detector Faster R-CNN entrenado en COCO: representaciones
*semánticas* de objetos. Sirve como baseline para ver cuánta señal FAKE/REAL hay
en las features de "qué objetos hay en la imagen" (que, por diseño, ignoran los
artefactos de upsampling que NPR explota).

A diferencia de NPR:
  - Entrada RGB normal (3 canales), no residuos.
  - Usa Resize estándar (la señal semántica sobrevive al re-muestreo; no
    dependemos de alta frecuencia como NPR).
  - El backbone parte de pesos COCO (Faster R-CNN ResNet-50 FPN).

Este módulo es COMPARTIDO por el entrenamiento (`train_coco_fakereal.py`) y la
inferencia (`models_infer.py`) para garantizar misma arquitectura y preprocesado.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)

IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _base_transform(augment: bool) -> transforms.Compose:
    # Backbone semántico (COCO): la señal vive en estructuras de objeto, no en la
    # alta frecuencia. Por eso, a diferencia de NPR, SÍ usamos Resize estándar
    # (bilineal) sin miedo a borrar residuos. En train añadimos flip + un crop
    # ligeramente más grande para augmentación barata.
    if augment:
        return transforms.Compose([
            transforms.Resize(int(IMAGE_SIZE * 1.14)),   # ~256
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(int(IMAGE_SIZE * 1.14)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_transform(augment: bool = False):
    """Preprocesado imagen PIL -> tensor RGB (3, 224, 224)."""
    return _base_transform(augment)


def _load_coco_backbone_weights(model: nn.Module) -> bool:
    """Copia los pesos del backbone ResNet-50 de Faster R-CNN (COCO) al ResNet-50.

    Faster R-CNN guarda su backbone bajo `backbone.body.<layer>` con los mismos
    nombres internos que un ResNet-50 (conv1, bn1, layer1..layer4). Copiamos esos
    tensores al modelo de clasificación. `fc` y `conv1`-stem que no casen se dejan
    con sus pesos por defecto. Devuelve True si se cargó al menos una capa COCO.
    """
    det_weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    det = fasterrcnn_resnet50_fpn(weights=det_weights)
    body = det.backbone.body.state_dict()   # claves: conv1.weight, layer1.0...

    own = model.state_dict()
    copied = 0
    for k, v in body.items():
        if k in own and own[k].shape == v.shape:
            own[k] = v
            copied += 1
    model.load_state_dict(own)
    return copied > 0


def build_coco_model(pretrained: bool = True, freeze_backbone: bool = False,
                     dropout: float = 0.0) -> nn.Module:
    """ResNet-50 con backbone COCO y cabeza a 2 clases (FAKE/REAL).

    Si `pretrained`, inicializa el backbone con los pesos COCO de Faster R-CNN
    (detección de objetos). Si esos pesos no se pueden descargar, cae a ImageNet
    como respaldo razonable.

    `freeze_backbone`: congela TODO menos la cabeza `fc` (linear probe). Es la
    forma honesta de medir cuánta señal FAKE/REAL hay en las features COCO sin
    que el ResNet (23M params) memorice un dataset pequeño. `dropout`>0 inserta
    un Dropout antes de `fc` para regularizar la cabeza.
    """
    if pretrained:
        # Base ImageNet como respaldo; encima copiamos el backbone COCO.
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        try:
            _load_coco_backbone_weights(model)
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  No se pudieron cargar pesos COCO ({exc}); uso ImageNet.")
    else:
        model = resnet50(weights=None)

    in_features = model.fc.in_features
    if dropout > 0.0:
        model.fc = nn.Sequential(nn.Dropout(p=dropout),
                                 nn.Linear(in_features, 2))
    else:
        model.fc = nn.Linear(in_features, 2)

    if freeze_backbone:
        # Congela todo y reactiva solo la cabeza (fc). Así el optimizador solo
        # ajusta la última capa: el backbone COCO queda intacto (linear probe).
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True

    return model
