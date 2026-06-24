#!/usr/bin/env python3
"""Inferencia de los detectores FAKE/REAL para la página "Tus respuestas".

Expone 4 modelos:
  - SIMPLE : regresión logística sobre 6 estadísticas globales de la imagen
             (detalle/suavidad). Es el que CLASIFICA bien (~87% val_acc); su
             heatmap es el mapa local de alta frecuencia. Ver simple_model.py.
  - NPR    : ResNet-50 sobre residuos de píxeles, fine-tuneado con
             `train_npr_fakereal.py`. Aporta el heatmap Grad-CAM (layer4[-1]).
  - SWIN   : Swin Transformer Base (swin_b) fine-tuneado con
             `train_swin_fakereal.py`. Heatmap Grad-CAM sobre features[-1][-1]
             (activaciones channel-last). Reemplaza al baseline COCO.
  - UNIVFD : CLIP ViT-L/14 congelado + cabeza lineal (Ojha et al., CVPR 2023),
             entrenado con `train_univfd_fakereal.py`. Heatmap Grad-CAM sobre el
             último bloque residual del ViT. Reemplaza al baseline COCO.

Cada modelo devuelve un overlay PNG, su predicción/confianza y las métricas de
similitud (CC/SIM/KL) frente al heatmap humano.

Expone `ModelBundle`, que carga el modelo una vez y, dada una imagen y las
muestras de mirada del participante, devuelve: un overlay PNG (imagen + mapa de
calor) en base64, la predicción/confianza, y las métricas de similitud
(CC/SIM/KL) frente al heatmap humano.

REQUISITOS: torch, torchvision, opencv-python, pillow, numpy y el paquete `clip`
(para UnivFD). Debe correrse en WSL/Linux con el entorno que tiene estos
paquetes; el import de torch falla en el venv ligero de Windows (por diseño: el
server "exige" estos paquetes).
"""

import base64
import io
import os
import threading
from pathlib import Path

import cv2
import numpy as np
import torch

from npr_model import build_npr_model, build_transform, IMAGE_SIZE
from swin_model import (
    build_swin_model,
    build_transform as build_swin_transform,
    gradcam_target_layer as swin_gradcam_target,
)
from univfd_model import (
    UnivFDClassifier,
    load_clip,
    build_transform as build_univfd_transform,
)
from simple_model import SimpleClassifier, smoothness_heatmap
from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# Config / rutas
# ---------------------------------------------------------------------------
SERVER_DIR = Path(__file__).resolve().parent

# Checkpoint del NPR fine-tuneado (heatmap Grad-CAM). Override con NPR_CKPT.
CKPT_NPR = Path(os.environ.get(
    "NPR_CKPT", str(SERVER_DIR / "npr_fakereal.pth")))

# Checkpoint del Swin-B (transformer + Grad-CAM). Override con SWIN_CKPT.
CKPT_SWIN = Path(os.environ.get(
    "SWIN_CKPT", str(SERVER_DIR / "swin_fakereal.pth")))

# Checkpoint del UnivFD (cabeza lineal sobre CLIP). Override con UNIVFD_CKPT.
CKPT_UNIVFD = Path(os.environ.get(
    "UNIVFD_CKPT", str(SERVER_DIR / "univfd_fakereal.pth")))

# Checkpoint del clasificador SIMPLE (JSON). Override con SIMPLE_CKPT.
CKPT_SIMPLE = Path(os.environ.get(
    "SIMPLE_CKPT", str(SERVER_DIR / "simple_fakereal.json")))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Heatmap humano (misma lógica que analysis/render_attention_heatmaps.py)
GRID = 200
SIGMA = 9.0


# ===========================================================================
# Heatmap HUMANO desde gaze_samples (u,v en [0,1])
# ===========================================================================
def gaussian_kernel(sigma):
    radius = max(1, int(round(3 * sigma)))
    ax = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def convolve2d_same(img, kernel):
    ih, iw = img.shape
    kh, kw = kernel.shape
    ph, pw = ih + kh - 1, iw + kw - 1
    fimg = np.fft.rfft2(img, s=(ph, pw))
    fker = np.fft.rfft2(kernel, s=(ph, pw))
    full = np.fft.irfft2(fimg * fker, s=(ph, pw))
    sy, sx = kh // 2, kw // 2
    return full[sy:sy + ih, sx:sx + iw]


def build_human_heatmap(samples):
    """Devuelve (heatmap 224x224 en [0,1], n_puntos). None si no hay datos."""
    grid = np.zeros((GRID, GRID), dtype=float)
    n = 0
    for s in samples:
        u = s.get("u")
        v = s.get("v")
        if u is None or v is None:
            u, v = s.get("x_norm"), s.get("y_norm")
        if u is None or v is None:
            continue
        if not (0.0 <= u < 1.0 and 0.0 <= v < 1.0):
            continue
        col = min(GRID - 1, int(u * GRID))
        row = min(GRID - 1, int(v * GRID))
        grid[row, col] += 1.0
        n += 1
    if n == 0:
        return None, 0
    heat = convolve2d_same(grid, gaussian_kernel(SIGMA))
    heat = np.clip(heat, 0.0, None)
    m = heat.max()
    if m > 0:
        heat = heat / m
    heat = cv2.resize(heat.astype(np.float32), (IMAGE_SIZE, IMAGE_SIZE))
    return heat, n


# ===========================================================================
# MODELO ÚNICO: NPR (ResNet-50 sobre residuos de píxeles) + Grad-CAM
# ===========================================================================
class NPRRunner:
    name = "npr"

    def __init__(self):
        ckpt = torch.load(str(CKPT_NPR), map_location=DEVICE)
        # pretrained=False: el state_dict del checkpoint trae todos los pesos.
        self.model = build_npr_model(pretrained=False).to(DEVICE)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.class_to_idx = ckpt.get("class_to_idx", {"FAKE": 0, "REAL": 1})
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.tf = build_transform(augment=False)

        # Grad-CAM sobre el último bloque conv del ResNet (layer4[-1]), el target
        # estándar para ResNet: atribuye la decisión FAKE/REAL espacialmente.
        self._acts = None
        self._grads = None
        self.model.layer4[-1].register_forward_hook(self._save_activation)

    def _save_activation(self, module, inp, output):
        self._acts = output
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self._grads = grad.detach()

    def _gradcam_map(self):
        # acts/grads: (B, C, h, w). Batch 0.
        acts = self._acts[0].detach()           # (C, h, w)
        grads = self._grads[0]                   # (C, h, w)
        weights = grads.mean(dim=(1, 2), keepdim=True)   # (C, 1, 1)
        cam = (weights * acts).sum(dim=0)        # (h, w)
        cam = torch.relu(cam).float().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        return cam

    def run(self, pil_img):
        x = self.tf(pil_img).unsqueeze(0).to(DEVICE)
        self._acts = None
        self._grads = None
        with torch.enable_grad():
            logits = self.model(x)               # (1, 2)
            probs = torch.softmax(logits, dim=1)[0]
            pred = int(logits.argmax(dim=1).item())
            self.model.zero_grad(set_to_none=True)
            logits[0, pred].backward()
        cam = self._gradcam_map()
        return cam, self.idx_to_class[pred], float(probs[pred].item())


# ===========================================================================
# MODELO SWIN: Swin Transformer Base (swin_b) + Grad-CAM
# ===========================================================================
class SwinRunner:
    name = "swin"

    def __init__(self):
        ckpt = torch.load(str(CKPT_SWIN), map_location=DEVICE)
        # pretrained=False: el state_dict del checkpoint trae todos los pesos.
        # Reconstruimos la cabeza con el MISMO dropout con que se entrenó para
        # que las claves casen (con dropout>0, head es un Sequential -> 'head.1.*').
        dropout = float(ckpt.get("dropout", 0.0))
        self.model = build_swin_model(pretrained=False, dropout=dropout).to(DEVICE)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.class_to_idx = ckpt.get("class_to_idx", {"FAKE": 0, "REAL": 1})
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.tf = build_swin_transform(augment=False)

        # Grad-CAM sobre la última etapa del Swin (features[-1][-1]). OJO: en Swin
        # las activaciones son channel-last (B, H, W, C), no (B, C, H, W) como en
        # un ResNet, así que hay que permutarlas antes de ponderar por gradientes.
        self._acts = None
        self._grads = None
        swin_gradcam_target(self.model).register_forward_hook(self._save_activation)

    def _save_activation(self, module, inp, output):
        self._acts = output
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self._grads = grad.detach()

    def _gradcam_map(self):
        # acts/grads en Swin: (B, H, W, C) -> tomamos batch 0 y pasamos a (C,H,W).
        acts = self._acts[0].detach().permute(2, 0, 1)    # (C, H, W)
        grads = self._grads[0].permute(2, 0, 1)           # (C, H, W)
        weights = grads.mean(dim=(1, 2), keepdim=True)    # (C, 1, 1)
        cam = (weights * acts).sum(dim=0)                 # (H, W)
        cam = torch.relu(cam).float().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        return cam

    def run(self, pil_img):
        x = self.tf(pil_img).unsqueeze(0).to(DEVICE)
        self._acts = None
        self._grads = None
        with torch.enable_grad():
            logits = self.model(x)               # (1, 2)
            probs = torch.softmax(logits, dim=1)[0]
            pred = int(logits.argmax(dim=1).item())
            self.model.zero_grad(set_to_none=True)
            logits[0, pred].backward()
        cam = self._gradcam_map()
        return cam, self.idx_to_class[pred], float(probs[pred].item())


# ===========================================================================
# MODELO UNIVFD: CLIP ViT-L/14 congelado + cabeza lineal + Grad-CAM (ViT)
# ===========================================================================
class UnivFDRunner:
    name = "univfd"

    def __init__(self):
        ckpt = torch.load(str(CKPT_UNIVFD), map_location=DEVICE)
        feat_dim = int(ckpt.get("feat_dim", 768))
        self.clip_model, _ = load_clip(DEVICE)
        self.classifier = UnivFDClassifier(feat_dim=feat_dim).to(DEVICE)
        self.classifier.load_state_dict(ckpt["model_state"])
        self.classifier.eval()
        self.class_to_idx = ckpt.get("class_to_idx", {"FAKE": 0, "REAL": 1})
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.fake_idx = self.class_to_idx.get("FAKE", 0)
        self.real_idx = self.class_to_idx.get("REAL", 1)
        self.tf = build_univfd_transform(DEVICE, augment=False)  # clip_preprocess

        # Grad-CAM sobre el PENÚLTIMO bloque residual del ViT de CLIP. Activaciones
        # (seq_len, B, C); el token 0 es CLS (se descarta). OJO: NO se puede hookear
        # el ÚLTIMO bloque: la feature de imagen de CLIP es solo el token CLS, así
        # que el gradiente sobre la salida del último bloque es no-nulo SOLO en CLS
        # y cero en todos los parches -> mapa degenerado. En el penúltimo bloque los
        # parches sí reciben gradiente (vía la atención del último bloque sobre
        # todos los tokens), dando un mapa espacial real.
        self._feat = None
        self._grad = None
        target = self.clip_model.visual.transformer.resblocks[-2]
        target.register_forward_hook(self._save_activation)
        target.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, output):
        self._feat = output                       # (seq_len, B, C)

    def _save_gradient(self, module, grad_input, grad_output):
        self._grad = grad_output[0].detach()      # (seq_len, B, C)

    def _gradcam_map(self):
        feat = self._feat[:, 0, :].float()         # (seq_len, C)
        grad = self._grad[:, 0, :].float()         # (seq_len, C)
        # En un ViT, el Grad-CAM clásico (promediar el gradiente sobre canales y
        # usarlo como peso) COLAPSA: por token, el gradiente está repartido casi
        # simétrico en torno a 0, así que su media por canal es ~0 y el mapa sale
        # nulo. Usamos la variante por-elemento (grad ⊙ activación, sumando sobre
        # canales): es la atribución estándar para tokens de transformer y no se
        # cancela. Tomamos |·| porque la relevancia del token va en ambos signos.
        cam = (grad * feat).sum(dim=1).abs()       # (seq_len,)
        cam = cam[1:]                              # quita el token CLS
        cam = cam.cpu().detach().numpy()
        n = int(round(cam.shape[0] ** 0.5))        # ViT-L/14 @224 -> 16x16 patches
        cam = cam[: n * n].reshape(n, n)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cv2.resize(cam.astype(np.float32), (IMAGE_SIZE, IMAGE_SIZE),
                         interpolation=cv2.INTER_LINEAR)
        return cam

    def run(self, pil_img):
        x = self.tf(pil_img).unsqueeze(0).to(DEVICE)
        # El encoder CLIP está congelado (ningún parámetro requiere grad). Para
        # que autograd retenga el grafo a través del último resblock y el backward
        # hook del Grad-CAM dispare, la ENTRADA debe requerir grad: así hay un
        # tensor upstream que necesita gradiente y las activaciones intermedias se
        # diferencian. Sin esto, _grad queda en None (grafo podado).
        x = x.requires_grad_(True)
        self._feat = None
        self._grad = None
        with torch.enable_grad():
            feats = self.clip_model.encode_image(x).float()   # (1, 768)
            logit = self.classifier(feats)                    # (1,) = P(FAKE) logit
            self.classifier.zero_grad(set_to_none=True)
            self.clip_model.zero_grad(set_to_none=True)
            prob_fake = float(torch.sigmoid(logit)[0].item())
            # Grad-CAM hacia la clase PREDICHA: el logit es P(FAKE), así que para
            # un fake (prob>0.5) propagamos +logit y para un real -logit. Sin esto,
            # al backpropagar siempre +logit, el mapa de un real sale todo negativo
            # y el ReLU lo deja en cero (CAM degenerado).
            target = logit.sum() if prob_fake > 0.5 else -logit.sum()
            target.backward()
        cam = self._gradcam_map()
        if prob_fake > 0.5:
            pred, conf = self.idx_to_class[self.fake_idx], prob_fake
        else:
            pred, conf = self.idx_to_class[self.real_idx], 1.0 - prob_fake
        return cam, pred, conf


# ===========================================================================
# MODELO 2: SIMPLE (regresión logística sobre 6 features) + heatmap de suavidad
# ===========================================================================
class SimpleRunner:
    name = "simple"

    def __init__(self):
        import json
        with open(str(CKPT_SIMPLE), "r", encoding="utf-8") as f:
            state = json.load(f)
        self.clf = SimpleClassifier.from_state(state)

    def run(self, pil_img):
        """Devuelve (heatmap_local, pred, conf). El heatmap es el mapa de detalle
        de alta frecuencia (suavidad): zonas suaves = sospechosas de generación.
        """
        pred, conf = self.clf.predict(pil_img)
        heat = smoothness_heatmap(pil_img)
        return heat, pred, conf


# ===========================================================================
# Métricas de similitud heatmap humano vs. mapa de modelo
# ===========================================================================
def _finite(x, default=0.0):
    """Devuelve x si es finito; si es NaN/inf, devuelve `default`. JSON no admite
    NaN/Infinity: si una métrica sale así, jsonify emite el literal `NaN`, que el
    navegador no puede parsear (res.json() falla -> 'respuesta no-JSON')."""
    return float(x) if np.isfinite(x) else float(default)


def similarity(h, m):
    """h, m: mapas 224x224 en [0,1]. Devuelve {cc, sim, kl} siempre finitos.

    Cuando un mapa es plano (heatmap humano vacío, o mapa de modelo casi todo
    cero), el corrcoef no está definido y el KL diverge (log de 0). Saneamos esos
    casos a valores finitos para que la respuesta sea JSON válido."""
    hf = h.astype(np.float64).ravel()
    mf = m.astype(np.float64).ravel()

    if hf.std() < 1e-8 or mf.std() < 1e-8:
        cc = 0.0
    else:
        cc = _finite(np.corrcoef(hf, mf)[0, 1])

    # Distribuciones de probabilidad: si la suma es ~0 (mapa vacío), repartimos
    # uniforme para evitar dividir por cero y que el KL diverja.
    n = hf.size
    hs, ms = hf.sum(), mf.sum()
    hp = hf / hs if hs > 1e-12 else np.full(n, 1.0 / n)
    mp = mf / ms if ms > 1e-12 else np.full(n, 1.0 / n)
    sim = _finite(np.minimum(hp, mp).sum())

    eps = 1e-12
    kl = _finite(np.sum(hp * np.log((hp + eps) / (mp + eps))))

    return {"cc": round(cc, 4), "sim": round(sim, 4), "kl": round(kl, 4)}


# ===========================================================================
# Overlay imagen + heatmap -> PNG base64
# ===========================================================================
def overlay(pil_img, heat01):
    """Mezcla imagen (224x224) + heatmap jet 50/50. Devuelve RGB float [0,1]."""
    img = np.asarray(pil_img.resize((IMAGE_SIZE, IMAGE_SIZE)).convert("RGB")) / 255.0
    heat = cv2.applyColorMap(np.uint8(255 * heat01), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB) / 255.0
    return np.clip(0.5 * img + 0.5 * heat, 0, 1)


def png_b64(rgb01):
    """RGB float [0,1] -> 'data:image/png;base64,...'."""
    arr = np.uint8(255 * np.clip(rgb01, 0, 1))
    buf = io.BytesIO()
    PILImage.fromarray(arr).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return "data:image/png;base64," + b64


# ===========================================================================
# Bundle: carga el modelo una vez y compara una imagen
# ===========================================================================
class ModelBundle:
    # Constructores de cada runner (instanciados según keep_in_memory).
    RUNNER_CLASSES = [SimpleRunner, NPRRunner, SwinRunner, UnivFDRunner]

    def __init__(self):
        self.loaded = False
        # Mantener los modelos en memoria entre peticiones SOLO si hay RAM de
        # sobra. En equipos con poca RAM (WSL) se cargan y descartan por petición.
        self.keep_in_memory = os.environ.get("EYESYNTH_KEEP_MODELS", "0") == "1"
        self._cache = {}
        # Los runners reutilizados entre peticiones guardan estado mutable (hooks
        # de Grad-CAM del NPR). Flask atiende peticiones en hilos concurrentes y
        # el frontend dispara varias /compare-models a la vez al abrir la
        # revisión, así que sin este lock las inferencias se entremezclan y los
        # mapas de unas imágenes "pisan" a los de otras. Una sola GPU/CPU no gana
        # nada ejecutándolas en paralelo, así que las serializamos.
        self._infer_lock = threading.Lock()

    def load_all(self):
        """Verifica pesos/entorno al arrancar. NO mantiene los modelos en
        memoria salvo que EYESYNTH_KEEP_MODELS=1, para evitar picos de RAM."""
        print(f"[models] Device: {DEVICE}")
        if DEVICE.type == "cuda":
            print(f"[models] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[models] Checkpoint SIMPLE: {CKPT_SIMPLE}")
        print(f"[models] Checkpoint NPR:    {CKPT_NPR}")
        print(f"[models] Checkpoint SWIN:   {CKPT_SWIN}")
        print(f"[models] Checkpoint UNIVFD: {CKPT_UNIVFD}")
        if not CKPT_SIMPLE.exists():
            raise FileNotFoundError(
                f"No se encontró el clasificador simple: {CKPT_SIMPLE}. "
                f"Entrena primero con: python server/train_simple_fakereal.py"
            )
        if not CKPT_NPR.exists():
            raise FileNotFoundError(
                f"No se encontró el peso del NPR: {CKPT_NPR}. "
                f"Entrena/fine-tunea primero con: python server/train_npr_fakereal.py"
            )
        if not CKPT_SWIN.exists():
            raise FileNotFoundError(
                f"No se encontró el peso del Swin-B: {CKPT_SWIN}. "
                f"Entrena primero con: python server/train_swin_fakereal.py"
            )
        if not CKPT_UNIVFD.exists():
            raise FileNotFoundError(
                f"No se encontró el peso del UnivFD: {CKPT_UNIVFD}. "
                f"Entrena primero con: python server/train_univfd_fakereal.py"
            )
        mode = "en memoria (rápido)" if self.keep_in_memory else "de a uno (bajo consumo de RAM)"
        print(f"[models] Modo de carga: {mode}")
        if self.keep_in_memory:
            print("[models] Cargando SIMPLE (regresión logística)...")
            self._cache["SimpleRunner"] = SimpleRunner()
            print("[models] Cargando NPR (Grad-CAM sobre layer4)...")
            self._cache["NPRRunner"] = NPRRunner()
            print("[models] Cargando SWIN (Swin-B + Grad-CAM)...")
            self._cache["SwinRunner"] = SwinRunner()
            print("[models] Cargando UNIVFD (CLIP ViT-L/14 + cabeza lineal)...")
            self._cache["UnivFDRunner"] = UnivFDRunner()
            print("[models] Listo: 4 modelos cargados en memoria.")
        else:
            print("[models] Verificando entorno (carga de prueba de los modelos)...")
            for cls in self.RUNNER_CLASSES:
                _ = cls()
                del _
            self._gc()
            print("[models] Entorno OK. Los modelos se cargan al pedir una comparación.")
        self.loaded = True

    def _gc(self):
        import gc
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    def _get_runner(self, cls):
        """Devuelve un runner cargado. Lo cachea solo si keep_in_memory."""
        if self.keep_in_memory:
            if cls.__name__ not in self._cache:
                self._cache[cls.__name__] = cls()
            return self._cache[cls.__name__], False   # (runner, descartar?)
        return cls(), True   # nuevo cada vez; se descarta tras usarlo

    def compare(self, pil_img, gaze_samples):
        """Devuelve dict con heatmap humano + overlay/predicción/métricas de cada
        modelo. Carga cada modelo de a uno y lo libera (salvo keep_in_memory)
        para minimizar el pico de RAM."""
        if not self.loaded:
            raise RuntimeError("Entorno no verificado (llama a load_all() primero).")

        human_heat, n_pts = build_human_heatmap(gaze_samples or [])
        if human_heat is None:
            human_heat = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
            n_pts = 0

        out = {
            "n_points": n_pts,
            "human_png": png_b64(overlay(pil_img, human_heat)),
            "models": {},
        }
        # Serializar la inferencia: los runners comparten estado mutable entre
        # peticiones concurrentes (ver _infer_lock en __init__).
        with self._infer_lock:
            for cls in self.RUNNER_CLASSES:
                runner, disposable = self._get_runner(cls)
                try:
                    mmap, pred, conf = runner.run(pil_img)
                    metrics = similarity(human_heat, mmap)
                    out["models"][runner.name] = {
                        "png": png_b64(overlay(pil_img, mmap)),
                        "pred": pred,
                        "conf": round(conf, 4),
                        **metrics,
                    }
                finally:
                    if disposable:
                        del runner
                        self._gc()
        return out
