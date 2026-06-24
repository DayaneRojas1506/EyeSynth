#!/usr/bin/env python3
"""Clasificador SIMPLE FAKE/REAL basado en estadísticas globales de la imagen.

Las imágenes fake de este dataset se distinguen por ser globalmente más suaves
(menos detalle de alta frecuencia) que las reales. Un detector deep (NPR) destruye
esa señal al normalizar; en cambio una regresión logística sobre 6 estadísticas
simples la captura y generaliza (~87% val_acc en pruebas).

Este módulo es COMPARTIDO por el entrenamiento (`train_simple_fakereal.py`) y la
inferencia (`models_infer.py`). El checkpoint guarda los pesos de la regresión
logística más la media/std de normalización de las features (calculadas en train).

Además expone un heatmap local de "suavidad" (alta frecuencia local): zonas
oscuras = poco detalle (sospechosas de generación), zonas claras = mucho detalle.
"""

import numpy as np
from PIL import Image

IMAGE_SIZE = 224   # tamaño al que se mide todo (consistencia con NPR/overlays)

# Nombres de las 6 features, en orden. Útil para depurar / interpretar pesos.
FEATURE_NAMES = [
    "residuo_h", "residuo_v", "laplaciano", "std_gris", "std_rgb", "fft_mag",
]


def extract_features(pil_img):
    """Devuelve un vector (6,) de estadísticas globales de la imagen.

    IMPORTANTE: NO se redimensiona la imagen. La señal que distingue fake/real
    vive en la alta frecuencia (detalle fino); un resize a 224 la destruiría y
    el clasificador caería a ~azar. Se mide a resolución nativa.
    """
    im = np.asarray(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    g = im.mean(2)   # escala de grises
    h = np.abs(im[:, :-1] - im[:, 1:]).mean()
    v = np.abs(im[:-1, :] - im[1:, :]).mean()
    lap = np.abs(
        g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    ).mean()
    return np.array([h, v, lap, g.std(), im.std(),
                     float(np.abs(np.fft.fft2(g)).mean())], dtype=np.float64)


def smoothness_heatmap(pil_img):
    """Heatmap local de alta frecuencia (detalle) en [0,1], tamaño 224x224.

    Calcula la magnitud del laplaciano por píxel y la suaviza con una media móvil.
    Valores ALTOS = mucho detalle local; valores BAJOS = zona suave (lo que el
    modelo asocia a generación). Se normaliza a [0,1] para el overlay.
    """
    im = np.asarray(
        pil_img.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE)),
        dtype=np.float32,
    ) / 255.0
    lap = np.zeros_like(im)
    lap[1:-1, 1:-1] = np.abs(
        im[1:-1, 1:-1] * 4 - im[:-2, 1:-1] - im[2:, 1:-1]
        - im[1:-1, :-2] - im[1:-1, 2:]
    )
    # Suavizado por promedio en ventana (box blur) vía integral image simple.
    k = 9
    pad = np.pad(lap, k // 2, mode="edge")
    csum = pad.cumsum(0).cumsum(1)
    csum = np.pad(csum, ((1, 0), (1, 0)), mode="constant")
    H, W = lap.shape
    out = (csum[k:k + H, k:k + W] - csum[:H, k:k + W]
           - csum[k:k + H, :W] + csum[:H, :W]) / (k * k)
    m = out.max()
    if m > 0:
        out = out / m
    return out.astype(np.float32)


class SimpleClassifier:
    """Regresión logística sobre las 6 features, con normalización integrada.

    Guardar/cargar es solo (w, b, mu, sd) — un puñado de floats, trivial.
    """

    def __init__(self, w=None, b=0.0, mu=None, sd=None,
                 class_to_idx=None):
        n = len(FEATURE_NAMES)
        self.w = np.zeros(n) if w is None else np.asarray(w, dtype=np.float64)
        self.b = float(b)
        self.mu = np.zeros(n) if mu is None else np.asarray(mu, dtype=np.float64)
        self.sd = np.ones(n) if sd is None else np.asarray(sd, dtype=np.float64)
        self.class_to_idx = class_to_idx or {"FAKE": 0, "REAL": 1}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def prob_real(self, feats):
        """P(clase REAL) dada un vector de features (sin normalizar)."""
        z = ((feats - self.mu) / self.sd) @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-z))

    def predict(self, pil_img):
        """Devuelve (pred_label, confianza) para una imagen PIL."""
        p_real = float(self.prob_real(extract_features(pil_img)))
        real_idx = self.class_to_idx.get("REAL", 1)
        fake_idx = self.class_to_idx.get("FAKE", 0)
        pred = real_idx if p_real > 0.5 else fake_idx
        conf = p_real if pred == real_idx else 1.0 - p_real
        return self.idx_to_class[pred], float(conf)

    def state_dict(self):
        return {
            "w": self.w.tolist(),
            "b": self.b,
            "mu": self.mu.tolist(),
            "sd": self.sd.tolist(),
            "class_to_idx": self.class_to_idx,
            "feature_names": FEATURE_NAMES,
        }

    @classmethod
    def from_state(cls, state):
        return cls(w=state["w"], b=state["b"], mu=state["mu"], sd=state["sd"],
                   class_to_idx=state.get("class_to_idx"))
