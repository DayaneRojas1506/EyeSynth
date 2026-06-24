#!/usr/bin/env python3
"""Modelo único FAKE/REAL basado en NPR (Neighboring Pixel Relationships).

Reemplaza a los 3 detectores anteriores (NPR / Swin-B / UnivFD) por un solo
modelo: la arquitectura NPR de Tan et al. (CVPR 2024), un ResNet-50 cuya entrada
son los residuos de píxeles vecinos (horizontal + vertical) en vez de RGB. Esos
residuos exponen los artefactos de upsampling que dejan los generadores, así que
detecta manipulaciones generativas que un backbone de detección de objetos
(COCO) ignora por diseño.

Este módulo es COMPARTIDO por el script de fine-tuning (`train_npr_fakereal.py`)
y por la inferencia del server (`models_infer.py`) para garantizar que
entrenamiento e inferencia usen EXACTAMENTE la misma arquitectura y el mismo
preprocesado.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

IMAGE_SIZE = 224
NPR_CHANNELS = 6   # 2 direcciones (horiz+vert) × 3 canales RGB

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Capa objetivo para Grad-CAM: el último bloque conv del ResNet (layer4[-1]),
# igual que en la inferencia original de NPR.


class NPRTransform:
    """Convierte un tensor (3, H, W) a su representación NPR (6, H, W).

    Para cada canal calcula el residuo con el vecino derecho (horizontal) y con
    el vecino inferior (vertical), y concatena ambos mapas de diferencias.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        horiz = x - torch.cat([x[:, :, 1:], x[:, :, -1:]], dim=2)
        vert = x - torch.cat([x[:, 1:, :], x[:, -1:, :]], dim=1)
        return torch.cat([horiz, vert], dim=0)   # (6, H, W)


_npr_tf = NPRTransform()


class ResizeIfSmaller:
    """Escala el lado corto a `size` SOLO si es menor que `size`.

    A diferencia de transforms.Resize(size), que siempre re-muestrea (y borraría
    los residuos de alta frecuencia que NPR necesita), esto es no-op cuando la
    imagen ya es lo bastante grande para un CenterCrop(size) — el caso del
    dataset 512x512. Solo escala imágenes pequeñas en inferencia para que el
    crop no falle.
    """

    def __init__(self, size: int):
        self.size = size
        self._resize = transforms.Resize(size)

    def __call__(self, img):
        w, h = img.size
        if min(w, h) < self.size:
            return self._resize(img)
        return img


def _base_transform(augment: bool) -> transforms.Compose:
    # IMPORTANTE: NPR mira residuos de píxeles vecinos (alta frecuencia). Un
    # Resize bilineal recalcula cada píxel promediando vecinos -> borra esos
    # residuos e inventa un patrón de interpolación uniforme en TODA la imagen,
    # destruyendo la señal que distingue fake/real. Por eso usamos CROP (no
    # resize): toma una ventana de 224 px nativos sin re-muestrear, dejando los
    # artefactos de upsampling intactos. El crop aleatorio en train aporta,
    # además, augmentación gratis que mitiga el overfitting con dataset pequeño.
    # No usamos ColorJitter porque altera la alta frecuencia (ruido para NPR).
    if augment:
        return transforms.Compose([
            transforms.RandomCrop(IMAGE_SIZE, pad_if_needed=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    # Val/inferencia: CenterCrop determinista (métrica estable). El Resize del
    # lado corto a IMAGE_SIZE es no-op para imágenes cuyo lado corto ya es
    # >=224 (como el dataset 512x512); solo evita que CenterCrop falle con
    # imágenes pequeñas en inferencia, escalando lo mínimo imprescindible.
    return transforms.Compose([
        ResizeIfSmaller(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_transform(augment: bool = False):
    """Preprocesado imagen PIL -> tensor NPR (6, 224, 224).

    Componemos el preprocesado base (resize/normalize/aug) con la transformada
    NPR que produce los 6 canales de residuos.
    """
    base = _base_transform(augment)

    def _tf(pil_img):
        return _npr_tf(base(pil_img))

    return _tf


def build_npr_model(pretrained: bool = True) -> nn.Module:
    """ResNet-50 con conv1 adaptada a 6 canales NPR y fc a 2 clases (FAKE/REAL).

    Si `pretrained`, parte de pesos ImageNet y replica los pesos de conv1 de 3
    a 6 canales (como en el entrenamiento original de NPR).
    """
    weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet50(weights=weights)
    old_conv = model.conv1
    model.conv1 = nn.Conv2d(
        NPR_CHANNELS, old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )
    if pretrained:
        with torch.no_grad():
            model.conv1.weight[:, :3] = old_conv.weight
            model.conv1.weight[:, 3:] = old_conv.weight
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model
