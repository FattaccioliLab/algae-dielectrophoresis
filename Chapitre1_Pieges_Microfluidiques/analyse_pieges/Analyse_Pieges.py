"""
=============================================================================
MICROFLUIDIC TRAP ANALYZER  v2.0
=============================================================================
Analyse de vidéos AVI de puces microfluidiques.

NOUVEAUTÉS v2.0
---------------
1. SOUSTRACTION DU FOND par image de référence (routine ImageJ) :
       Résultat = Référence(pièges vides) - Image courante
   Réplique de :
       imageCalculator("Subtract create", "Empty", "Full");
   Corrige l'éclairage non uniforme : les particules ressortent en BLANC
   sur un fond noir, quelle que soit l'inhomogénéité de la lampe.
   L'utilisateur choisit la frame de référence dans une fenêtre dédiée,
   avec aperçu en direct sur une frame de contrôle contenant des particules.

2. SEUIL EXPRIMÉ COMME UNE PLAGE  setThreshold(lo, hi)  au lieu d'un seul
   seuil bas, afin de couvrir les deux polarités (particules sombres sans
   soustraction, particules claires après soustraction).

3. DEUX MODES DE COMPTAGE :
       • BINAIRE   : piège vide (0) ou plein (1)      → robuste
       • COMPTAGE  : 0, 1, 2, … particules par piège  → informatif

Pipeline complet :
  0. (option) Soustraction de la référence      → particules claires
  1. Conversion 8-bit
  2. Remap LUT  [lut_min, lut_max] → [0, 255]
  3. Threshold  [thr_lo, thr_hi]
  4. Fill Holes + nettoyage morphologique
  5. Mode BINAIRE  : aire du plus gros blob ≥ min_area  → 0 / 1
     Mode COMPTAGE : blobs + filtre taille/solidité/circularité,
                     amas → HoughCircles local

Dépendances :
    pip install opencv-python numpy matplotlib scipy Pillow openpyxl

Usage : Ouvrir dans Pyzo et exécuter (F5)
=============================================================================
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONSTANTES
# =============================================================================
APP_TITLE  = "Microfluidic Trap Analyzer"
VERSION    = "2.0"
ROI_EXPAND = 2

MODE_COUNT  = "count"
MODE_BINARY = "binary"

# Couleurs RGB fond blanc — 0=vide (gris), 1=bleu, 2=vert, 3=orange, 4+=rouge
COLORS_RGB = {
    0: (180, 180, 180),
    1: (30,  100, 220),
    2: (30,  160,  50),
    3: (220, 120,   0),
    4: (200,  20,  20),
}
COLORS_BGR = {k: (v[2], v[1], v[0]) for k, v in COLORS_RGB.items()}


def legend_label(k, mode):
    """Libellé de légende adapté au mode de comptage."""
    if mode == MODE_BINARY:
        return "vide" if k == 0 else "plein"
    return f'{k}{"+" if k == 4 else ""}'


# =============================================================================
# DÉTECTION HORS-FOCUS  (Laplacien + luminosité combinés)
# =============================================================================

def frame_sharpness(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def frame_brightness(gray):
    return float(np.mean(gray))

def detect_oof(sharpness, brightness, win=15, lap_thr=0.40, bri_thr=0.30):
    """
    Frame marquée OOF si :
      - son Laplacien local chute sous lap_thr × mediane locale, OU
      - sa luminosité s'écarte de plus de bri_thr (relatif) de la médiane locale.

    NOTE : toujours calculé sur l'image BRUTE, jamais sur l'image soustraite,
    sinon la soustraction masquerait justement les dérives que l'on cherche.
    """
    s = np.array(sharpness,  dtype=float)
    b = np.array(brightness, dtype=float)
    n = len(s)
    oof = np.zeros(n, dtype=bool)
    med_s_g = np.median(s[s > 0]) if np.any(s > 0) else 1.0
    med_b_g = np.median(b[b > 0]) if np.any(b > 0) else 1.0
    for i in range(n):
        lo = max(0, i - win);  hi = min(n, i + win + 1)
        ms = np.median(s[lo:hi]) or med_s_g
        mb = np.median(b[lo:hi]) or med_b_g
        lap_oof = (s[i] / ms) < lap_thr  if ms > 0 else False
        bri_oof = (abs(b[i] - mb) / mb) > bri_thr if mb > 0 else False
        oof[i]  = lap_oof or bri_oof
    return oof


# =============================================================================
# SOUSTRACTION DU FOND  (routine ImageJ « Subtract create »)
# =============================================================================

def subtract_reference(gray, ref_gray, invert=False):
    """
    Réplique de :  imageCalculator("Subtract create", "Empty", "Full")

    Résultat = Référence - Image courante, saturé à 0 (pas de valeurs
    négatives, comme en 8-bit ImageJ).

    Une particule sombre dans l'image courante donne donc une valeur
    POSITIVE et ÉLEVÉE : les particules ressortent en BLANC sur fond noir.
    Toute inhomogénéité d'éclairage présente dans les deux images
    s'annule par construction.

    invert=True reproduit le  run("Invert")  du script d'origine et
    ramène les particules en SOMBRE sur fond clair. Laisser à False est
    en général préférable : les particules claires sur fond noir donnent
    un bien meilleur contraste après soustraction.
    """
    if ref_gray is None:
        return gray
    if ref_gray.shape != gray.shape:
        ref_gray = cv2.resize(ref_gray, (gray.shape[1], gray.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
    diff = cv2.subtract(ref_gray, gray)      # saturation à 0 incluse
    if invert:
        diff = cv2.bitwise_not(diff)
    return diff


# =============================================================================
# DÉTECTION DE PARTICULES  (pipeline ImageJ-like + hybride)
# =============================================================================

def imagej_preprocess(gray, lut_min=40, lut_max=255, thr_lo=0, thr_hi=155):
    """
    Réplique du pipeline ImageJ :
      1. Cast uint8
      2. Remap LUT  [lut_min, lut_max] → [0, 255]      (setMinAndMax + Apply)
      3. Threshold  setThreshold(thr_lo, thr_hi)
         → pixels DANS la plage = objets = 255 dans le masque

    La plage explicite permet de couvrir les deux polarités :
      • sans soustraction, particules sombres  → [0, 155] environ
      • après soustraction, particules claires → [64, 255] environ

    Retourne : (mask uint8, remapped uint8)
    """
    gray8    = np.clip(gray, 0, 255).astype(np.uint8) if gray.dtype != np.uint8 else gray.copy()
    lut_rng  = float(max(lut_max - lut_min, 1))
    remapped = np.clip((gray8.astype(np.float32) - lut_min) * (255.0 / lut_rng),
                       0.0, 255.0).astype(np.uint8)
    lo, hi   = int(min(thr_lo, thr_hi)), int(max(thr_lo, thr_hi))
    mask     = np.where((remapped >= lo) & (remapped <= hi),
                        np.uint8(255), np.uint8(0))
    return mask, remapped


def fill_holes(mask):
    """
    Équivalent du 'Fill Holes' d'ImageJ.
    RETR_CCOMP : niveau 0 = contours externes, niveau 1 = trous internes.
    On remplit tous les trous internes → les amas à cœur clair deviennent
    des disques pleins.
    """
    filled = mask.copy()
    cnts, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP,
                                       cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return filled
    hierarchy = hierarchy[0]
    for i, cnt in enumerate(cnts):
        if hierarchy[i][3] >= 0:
            cv2.drawContours(filled, [cnt], -1, 255, thickness=cv2.FILLED)
    return filled


def build_mask(roi_gray, lut_min, lut_max, thr_lo, thr_hi):
    """Étapes communes aux deux modes : LUT → threshold → fill holes → morpho."""
    mask, remapped = imagej_preprocess(roi_gray, lut_min, lut_max, thr_lo, thr_hi)
    mask   = fill_holes(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask, remapped


def _blob_solidity(cnt):
    """Solidité = aire_contour / aire_convexe_hull.  Bille ≈ 1.0, arc < 0.75."""
    area = cv2.contourArea(cnt)
    if area == 0:
        return 0.0
    hull   = cv2.convexHull(cnt)
    h_area = cv2.contourArea(hull)
    return area / h_area if h_area > 0 else 0.0


def _remove_concentrics(circles, min_particle_r):
    """Supprime les cercles concentriques détectés par Hough (anneaux du même blob)."""
    if not circles:
        return []
    sorted_c = sorted(circles, key=lambda c: c[2], reverse=True)
    kept = []
    for cand in sorted_c:
        cx_c, cy_c, r_c = cand
        too_close = False
        for cx_k, cy_k, r_k in kept:
            if np.hypot(cx_c - cx_k, cy_c - cy_k) < min_particle_r:
                too_close = True
                break
        if not too_close:
            kept.append(cand)
    return kept


def _hough_on_blob(blob_mask_patch, x1b, y1b, particle_r, hough_sensitivity=12):
    """HoughCircles sur le masque REMPLI du blob recadré."""
    if blob_mask_patch.size == 0:
        return []

    blurred  = cv2.GaussianBlur(blob_mask_patch, (5, 5), 1.2)
    r_min    = max(2, int(particle_r * 0.60))
    r_max    = max(r_min + 2, int(particle_r * 1.45))
    min_dist = max(int(particle_r * 2 * 0.80), r_min * 2)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.0, minDist=min_dist,
        param1=40, param2=hough_sensitivity,
        minRadius=r_min, maxRadius=r_max,
    )
    if circles is None:
        return []

    ph, pw = blob_mask_patch.shape
    raw = []
    for (cx, cy, r) in np.round(circles[0]).astype(int):
        if 0 <= cy < ph and 0 <= cx < pw and blob_mask_patch[cy, cx] > 0:
            raw.append((int(cx), int(cy), int(r)))

    clean = _remove_concentrics(raw, particle_r)
    return [(cx + x1b, cy + y1b, r) for (cx, cy, r) in clean]


def detect_particles(roi_gray,
                     lut_min=40, lut_max=255,
                     thr_lo=0, thr_hi=155,
                     min_area=20, max_area=2000,
                     circularity_min=0.55,
                     use_watershed=True,
                     min_particle_r=5,
                     mode=MODE_COUNT):
    """
    Détecte les particules dans une ROI déjà prétraitée (soustraction du fond
    éventuellement déjà appliquée en amont, sur l'image entière).

    mode = MODE_BINARY :
        On ne cherche pas à compter. Un piège est déclaré PLEIN dès qu'il
        contient un blob d'aire ≥ min_area. Beaucoup plus robuste sur des
        objets non sphériques ou accolés, typiquement des algues.
        Retourne (0 ou 1, [(cx, cy, r)] du plus gros blob, mask).

    mode = MODE_COUNT :
        Pipeline complet : filtre aire, solidité, circularité, puis Hough
        local sur les amas.
        Retourne (n, [(cx, cy, r), …], mask).
    """
    if roi_gray is None or roi_gray.size == 0:
        return 0, [], None
    h, w = roi_gray.shape
    if h < 6 or w < 6:
        return 0, [], None

    mask, _remapped = build_mask(roi_gray, lut_min, lut_max, thr_lo, thr_hi)
    mask_debug = mask.copy()

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ── MODE BINAIRE ─────────────────────────────────────────────────────────
    if mode == MODE_BINARY:
        best = None
        best_area = 0.0
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            if area > best_area:
                best_area = area
                best = cnt
        if best is None:
            return 0, [], mask_debug
        M_cnt = cv2.moments(best)
        if M_cnt['m00'] == 0:
            return 0, [], mask_debug
        cx0 = int(M_cnt['m10'] / M_cnt['m00'])
        cy0 = int(M_cnt['m01'] / M_cnt['m00'])
        r   = max(int(np.sqrt(best_area / np.pi)), 1)
        return 1, [(cx0, cy0, r)], mask_debug

    # ── MODE COMPTAGE ────────────────────────────────────────────────────────
    single_area   = np.pi * (min_particle_r ** 2)
    blob_area_thr = single_area * 1.25
    hough_sens    = max(8, min(20, int(min_particle_r * 1.4)))
    particles     = []

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        if _blob_solidity(cnt) < 0.75:
            continue

        M_cnt = cv2.moments(cnt)
        if M_cnt['m00'] == 0:
            continue
        cx0 = int(M_cnt['m10'] / M_cnt['m00'])
        cy0 = int(M_cnt['m01'] / M_cnt['m00'])

        if area <= blob_area_thr:
            peri = cv2.arcLength(cnt, True)
            circ = (4.0 * np.pi * area / peri ** 2) if peri > 0 else 0.0
            if circ >= circularity_min:
                r = max(int(np.sqrt(area / np.pi)), 1)
                particles.append((cx0, cy0, r))

        elif use_watershed:
            blob_mask = np.zeros_like(mask)
            cv2.drawContours(blob_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            margin = min_particle_r + 2
            ys, xs = np.where(blob_mask > 0)
            x1b = max(0, int(xs.min()) - margin)
            y1b = max(0, int(ys.min()) - margin)
            x2b = min(w, int(xs.max()) + margin + 1)
            y2b = min(h, int(ys.max()) + margin + 1)
            patch_mask = blob_mask[y1b:y2b, x1b:x2b]

            found = _hough_on_blob(patch_mask, x1b, y1b,
                                   particle_r=min_particle_r,
                                   hough_sensitivity=hough_sens)
            if found:
                particles.extend(found)
            else:
                n_est = max(1, round(area / single_area))
                for _ in range(n_est):
                    particles.append((cx0, cy0, max(int(min_particle_r), 1)))

    return len(particles), particles, mask_debug


# Alias de compatibilité avec la v1.x
detect_spherical_particles = detect_particles


# =============================================================================
# UTILITAIRES IMAGE
# =============================================================================

def rotate_image(image, angle_deg):
    if angle_deg == 0.0:
        return image
    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2
    M  = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos_, sin_ = abs(M[0, 0]), abs(M[0, 1])
    nw = int(h * sin_ + w * cos_)
    nh = int(h * cos_ + w * sin_)
    M[0, 2] += nw / 2 - cx
    M[1, 2] += nh / 2 - cy
    return cv2.warpAffine(image, M, (nw, nh),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def label_panel(img_gray_or_rgb, text):
    """Convertit en RGB si besoin et écrit un titre en haut à gauche."""
    if img_gray_or_rgb.ndim == 2:
        out = cv2.cvtColor(img_gray_or_rgb, cv2.COLOR_GRAY2RGB)
    else:
        out = img_gray_or_rgb.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# =============================================================================
# MODÈLE
# =============================================================================

class MicrofluidicAnalyzer:

    def __init__(self):
        self.video_path      = None
        self.cap             = None
        self.total_frames    = 0
        self.fps             = 25.0
        self.rotation_angle  = 0.0
        self.crop_rect       = None        # (x1, y1, x2, y2)

        # Points de calibration
        self.trap_tl         = None
        self.trap_br         = None
        self.next_col_tl     = None
        self.next_row_tl     = None
        self.corner_tr       = None
        self.corner_bl       = None
        self.last_trap_tl    = None
        self.roi_expand      = ROI_EXPAND

        # Grille générée
        self.n_cols   = 0
        self.n_rows   = 0
        self.rois     = []
        self.trap_w   = 0
        self.trap_h   = 0
        self.dx_col   = 0
        self.dy_col   = 0
        self.dy_row   = 0

        # Paramètres d'analyse
        self.frame_start    = 0
        self.frame_end      = 0
        self.analysis_step  = 1

        # ── Soustraction du fond ────────────────────────────────────────────
        self.bg_subtract   = False   # activer la soustraction
        self.bg_ref_frame  = 0       # index de la frame de référence (pièges vides)
        self.bg_ref_navg   = 1       # moyenner N frames autour pour réduire le bruit
        self.bg_invert     = False   # reproduire le run("Invert") du script ImageJ
        self._bg_ref_gray  = None    # cache (non sérialisé)

        # ── Mode de comptage ────────────────────────────────────────────────
        self.count_mode = MODE_COUNT     # MODE_COUNT ou MODE_BINARY

        # Paramètres détection (pipeline ImageJ-like)
        self.detect_lut_min       = 19
        self.detect_lut_max       = 109
        self.detect_thr_lo        = 0      # setThreshold bas
        self.detect_thr_hi        = 155    # setThreshold haut
        self.detect_min_area      = 20
        self.detect_max_area      = 2000
        self.detect_circularity   = 0.55
        self.detect_watershed     = True
        self.detect_particle_r    = 5

        # Résultats
        self.analyzed_frames = []
        self.counts_matrix   = None
        self.sharpness_arr   = []
        self.brightness_arr  = []
        self.oof_flags       = []

    # ------------------------------------------------------------------ IO
    def open_video(self, path):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"Impossible d'ouvrir : {path}")
        self.video_path   = path
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps          = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_start  = 0
        self.frame_end    = self.total_frames - 1
        self._bg_ref_gray = None

    def read_frame_raw(self, idx):
        """Frame brute avec rotation + crop (BGR)."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = self.cap.read()
        if not ret:
            return None
        frame = rotate_image(frame, self.rotation_angle)
        if self.crop_rect:
            x1, y1, x2, y2 = self.crop_rect
            frame = frame[y1:y2, x1:x2]
        return frame

    def read_gray_raw(self, idx):
        """Frame en niveaux de gris, avec rotation + crop, SANS soustraction."""
        f = self.read_frame_raw(idx)
        return None if f is None else cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

    def get_preview_frame(self, idx=0):
        """Frame avec rotation uniquement (pour le wizard avant crop)."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = self.cap.read()
        if not ret:
            return None
        return rotate_image(frame, self.rotation_angle)

    # ------------------------------------------------------ Soustraction fond
    def build_reference(self, idx=None, navg=None):
        """
        Construit et met en cache l'image de référence (pièges vides).

        navg > 1 : moyenne des navg frames centrées sur idx, ce qui réduit le
        bruit de capteur sans coût notable. La référence subit exactement la
        même rotation et le même crop que les images analysées.
        """
        idx  = self.bg_ref_frame if idx  is None else int(idx)
        navg = self.bg_ref_navg  if navg is None else max(1, int(navg))

        half   = navg // 2
        lo     = max(0, idx - half)
        hi     = min(self.total_frames - 1, idx + half)
        stack  = []
        for i in range(lo, hi + 1):
            g = self.read_gray_raw(i)
            if g is not None:
                stack.append(g.astype(np.float32))
        if not stack:
            raise ValueError(f"Impossible de lire la frame de référence {idx}.")

        ref = np.clip(np.mean(stack, axis=0), 0, 255).astype(np.uint8)
        self.bg_ref_frame = idx
        self.bg_ref_navg  = navg
        self._bg_ref_gray = ref
        return ref

    def get_reference(self):
        """Référence en cache, reconstruite à la demande."""
        if self._bg_ref_gray is None and self.bg_subtract:
            self.build_reference()
        return self._bg_ref_gray

    def preprocess_gray(self, gray):
        """
        Applique la soustraction du fond si elle est activée.
        À appeler UNE FOIS sur l'image entière, avant le découpage en ROIs.
        """
        if not self.bg_subtract:
            return gray
        ref = self.get_reference()
        if ref is None:
            return gray
        return subtract_reference(gray, ref, invert=self.bg_invert)

    def read_gray_processed(self, idx):
        """Gris + rotation + crop + soustraction éventuelle."""
        g = self.read_gray_raw(idx)
        return None if g is None else self.preprocess_gray(g)

    # ------------------------------------------------------------------ ROI
    def compute_rois(self):
        """
        Calcule les ROIs de toute la grille par interpolation bilinéaire
        à partir des 4 coins cliqués (A=TL, B=TR, C=BL, D=BR).
            P(u,v) = (1-u)(1-v)·A + u(1-v)·B + (1-u)v·C + uv·D
        """
        required = [("trap_tl",      self.trap_tl),
                    ("trap_br",      self.trap_br),
                    ("next_col_tl",  self.next_col_tl),
                    ("next_row_tl",  self.next_row_tl),
                    ("corner_tr",    self.corner_tr),
                    ("corner_bl",    self.corner_bl),
                    ("last_trap_tl", self.last_trap_tl)]
        for name, v in required:
            if v is None:
                raise ValueError(f"Point de calibration manquant : {name}")

        x0, y0 = self.trap_tl
        xb, yb = self.trap_br
        self.trap_w = xb - x0
        self.trap_h = yb - y0

        xc, yc      = self.next_col_tl
        self.dx_col = xc - x0
        self.dy_col = yc - y0

        xr, yr      = self.next_row_tl
        self.dy_row = yr - y0

        if self.dx_col <= 0:
            raise ValueError("dx_col ≤ 0 — vérifiez le point 'Col+1'.")
        if self.dy_row <= 0:
            raise ValueError("dy_row ≤ 0 — vérifiez le point 'Row+1'.")

        xtr, ytr = self.corner_tr
        xbl, ybl = self.corner_bl
        xbr, ybr = self.last_trap_tl

        self.n_cols = max(2, round(np.hypot(xtr - x0, ytr - y0) / self.dx_col) + 1)
        self.n_rows = max(2, round(np.hypot(xbl - x0, ybl - y0) / self.dy_row) + 1)

        A = np.array([x0,  y0],  dtype=float)
        B = np.array([xtr, ytr], dtype=float)
        C = np.array([xbl, ybl], dtype=float)
        D = np.array([xbr, ybr], dtype=float)

        exp = self.roi_expand
        self.rois = []
        nc, nr = self.n_cols, self.n_rows

        for row in range(nr):
            for col in range(nc):
                u = col / (nc - 1) if nc > 1 else 0.0
                v = row / (nr - 1) if nr > 1 else 0.0
                pos = ((1-u)*(1-v) * A + u*(1-v) * B +
                       (1-u)*v     * C + u*v     * D)
                cx = int(round(pos[0]))
                cy = int(round(pos[1]))
                cy += int(self.dy_col * (col % 2))
                self.rois.append((max(0, cx - exp),
                                  max(0, cy - exp),
                                  cx + self.trap_w + exp,
                                  cy + self.trap_h + exp))

    # ------------------------------------------------------------------ Config
    def save_config(self, path):
        cfg = dict(
            version=VERSION,
            video_path=self.video_path,
            rotation_angle=self.rotation_angle,
            crop_rect=self.crop_rect,
            trap_tl=self.trap_tl, trap_br=self.trap_br,
            next_col_tl=self.next_col_tl, next_row_tl=self.next_row_tl,
            corner_tr=self.corner_tr,
            corner_bl=self.corner_bl,
            last_trap_tl=self.last_trap_tl,
            roi_expand=self.roi_expand,
            n_cols=self.n_cols, n_rows=self.n_rows,
            bg_subtract=self.bg_subtract,
            bg_ref_frame=self.bg_ref_frame,
            bg_ref_navg=self.bg_ref_navg,
            bg_invert=self.bg_invert,
            count_mode=self.count_mode,
            detect_lut_min=self.detect_lut_min,
            detect_lut_max=self.detect_lut_max,
            detect_thr_lo=self.detect_thr_lo,
            detect_thr_hi=self.detect_thr_hi,
            detect_min_area=self.detect_min_area,
            detect_max_area=self.detect_max_area,
            detect_circularity=self.detect_circularity,
            detect_watershed=self.detect_watershed,
            detect_particle_r=self.detect_particle_r,
        )
        with open(path, 'w') as f:
            json.dump(cfg, f, indent=2)

    def load_config(self, path):
        with open(path) as f:
            cfg = json.load(f)

        # Migration des configs v1.x : detect_thr_low → plage [0, thr_low-1]
        if 'detect_thr_low' in cfg and 'detect_thr_hi' not in cfg:
            old = int(cfg.pop('detect_thr_low'))
            cfg['detect_thr_lo'] = 0
            cfg['detect_thr_hi'] = max(0, old - 1)
            print(f"[config] v1.x détectée : seuil {old} → plage [0, {old-1}]")

        for k, v in cfg.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self._bg_ref_gray = None      # la référence sera reconstruite

    # ------------------------------------------------------------------ Analyse
    def count_in_roi(self, proc_gray, roi):
        """Compte (ou binarise) dans une ROI d'une image DÉJÀ prétraitée."""
        x1, y1, x2, y2 = roi
        sub = proc_gray[y1:y2, x1:x2]
        cnt, particles, _ = detect_particles(
            sub,
            lut_min=self.detect_lut_min,
            lut_max=self.detect_lut_max,
            thr_lo=self.detect_thr_lo,
            thr_hi=self.detect_thr_hi,
            min_area=self.detect_min_area,
            max_area=self.detect_max_area,
            circularity_min=self.detect_circularity,
            use_watershed=self.detect_watershed,
            min_particle_r=self.detect_particle_r,
            mode=self.count_mode,
        )
        return cnt, particles

    def analyze_all_frames(self, progress_cb=None):
        indices = list(range(self.frame_start,
                             self.frame_end + 1,
                             max(1, self.analysis_step)))
        n  = len(indices)
        nt = len(self.rois)

        # Construire la référence une seule fois, avant la boucle
        if self.bg_subtract:
            self.build_reference()

        raw    = np.zeros((n, nt), dtype=np.int32)
        sharp  = np.zeros(n)
        bright = np.zeros(n)
        done   = 0

        for i, fi in enumerate(indices):
            frame = self.read_frame_raw(fi)
            if frame is None:
                break
            gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # OOF toujours calculé sur l'image BRUTE
            sharp[i]  = frame_sharpness(gray)
            bright[i] = frame_brightness(gray)

            # Soustraction du fond une seule fois par frame
            proc = self.preprocess_gray(gray)

            for ti, roi in enumerate(self.rois):
                cnt, _p = self.count_in_roi(proc, roi)
                raw[i, ti] = cnt

            done = i + 1
            if progress_cb:
                progress_cb(done, n)

        oof = detect_oof(sharp[:done], bright[:done])
        self.sharpness_arr   = sharp[:done]
        self.brightness_arr  = bright[:done]
        self.oof_flags       = oof
        self.analyzed_frames = indices[:done]

        # Gel pendant les épisodes OOF
        clean = raw[:done].copy()
        last  = raw[0].copy()
        for j in range(done):
            if oof[j]:
                clean[j] = last
            else:
                last = raw[j].copy()

        self.counts_matrix = clean.reshape(done, self.n_rows, self.n_cols)
        return self.counts_matrix


# =============================================================================
# FENÊTRE — CHOIX DE L'IMAGE DE RÉFÉRENCE
# =============================================================================

class ReferenceFrameWindow:
    """
    Permet de parcourir la vidéo et de désigner la frame de référence,
    c'est-à-dire une frame où TOUS LES PIÈGES SONT VIDES.

    Deux curseurs :
      • Référence  : la frame vide qui servira de fond
      • Contrôle   : une frame contenant des particules, pour vérifier
                     visuellement le résultat de la soustraction

    Trois panneaux affichés en direct :
      ① Référence (vide)   ② Contrôle (avec particules)   ③ Référence − Contrôle
    """

    def __init__(self, master, analyzer):
        self.analyzer = analyzer
        self.ok       = False
        self._tk_img  = None

        win = tk.Toplevel(master)
        win.title("Choix de l'image de référence (pièges vides)")
        win.geometry("1180x680")
        win.configure(bg="#1e1e2e")
        win.grab_set()
        self.win = win

        tk.Label(win, text="🖼   Image de référence pour la soustraction du fond",
                 font=("Helvetica", 13, "bold"),
                 bg="#1e1e2e", fg="#89b4fa").pack(pady=(10, 2))
        tk.Label(win,
                 text="Choisissez une frame où AUCUN piège n'est occupé. "
                      "Le panneau de droite montre le résultat de la soustraction "
                      "sur la frame de contrôle : les particules doivent apparaître "
                      "en blanc sur fond noir.",
                 font=("Helvetica", 9), bg="#1e1e2e", fg="#cdd6f4",
                 wraplength=1120, justify="left").pack(padx=14, pady=(0, 6))

        self.canvas = tk.Canvas(win, bg="#0d0d14", highlightthickness=0, height=360)
        self.canvas.pack(fill="both", expand=True, padx=10, pady=4)
        self.canvas.bind("<Configure>", lambda e: self._refresh())

        ctrl = tk.Frame(win, bg="#1e1e2e")
        ctrl.pack(fill="x", padx=16, pady=4)

        last = max(0, analyzer.total_frames - 1)

        # Curseur référence
        r1 = tk.Frame(ctrl, bg="#1e1e2e"); r1.pack(fill="x", pady=2)
        tk.Label(r1, text="Frame de référence (pièges vides)", width=30, anchor="w",
                 bg="#1e1e2e", fg="#a6e3a1",
                 font=("Helvetica", 10, "bold")).pack(side="left")
        self.v_ref = tk.IntVar(value=min(analyzer.bg_ref_frame, last))
        self.sc_ref = tk.Scale(r1, from_=0, to=last, orient="horizontal",
                               variable=self.v_ref, showvalue=True,
                               bg="#1e1e2e", fg="#cdd6f4",
                               troughcolor="#313244", highlightthickness=0,
                               command=lambda _v: self._refresh())
        self.sc_ref.pack(side="left", fill="x", expand=True, padx=8)
        tk.Spinbox(r1, from_=0, to=last, textvariable=self.v_ref, width=7,
                   bg="#313244", fg="white",
                   command=self._refresh).pack(side="left")

        # Curseur contrôle
        r2 = tk.Frame(ctrl, bg="#1e1e2e"); r2.pack(fill="x", pady=2)
        tk.Label(r2, text="Frame de contrôle (avec particules)", width=30, anchor="w",
                 bg="#1e1e2e", fg="#f9e2af",
                 font=("Helvetica", 10, "bold")).pack(side="left")
        self.v_tst = tk.IntVar(value=min(last, max(0, last // 2)))
        self.sc_tst = tk.Scale(r2, from_=0, to=last, orient="horizontal",
                               variable=self.v_tst, showvalue=True,
                               bg="#1e1e2e", fg="#cdd6f4",
                               troughcolor="#313244", highlightthickness=0,
                               command=lambda _v: self._refresh())
        self.sc_tst.pack(side="left", fill="x", expand=True, padx=8)
        tk.Spinbox(r2, from_=0, to=last, textvariable=self.v_tst, width=7,
                   bg="#313244", fg="white",
                   command=self._refresh).pack(side="left")

        # Options
        r3 = tk.Frame(ctrl, bg="#1e1e2e"); r3.pack(fill="x", pady=6)
        tk.Label(r3, text="Moyenner N frames autour de la référence",
                 bg="#1e1e2e", fg="#cdd6f4", font=("Helvetica", 9)).pack(side="left")
        self.v_navg = tk.IntVar(value=analyzer.bg_ref_navg)
        tk.Spinbox(r3, from_=1, to=51, textvariable=self.v_navg, width=5,
                   bg="#313244", fg="white",
                   command=self._refresh).pack(side="left", padx=6)
        tk.Label(r3, text="(réduit le bruit du capteur ; 1 = frame unique)",
                 bg="#1e1e2e", fg="#585b70", font=("Helvetica", 8)).pack(side="left")

        self.v_inv = tk.BooleanVar(value=analyzer.bg_invert)
        tk.Checkbutton(r3, text="Inverser après soustraction  (run(\"Invert\"))",
                       variable=self.v_inv, command=self._refresh,
                       bg="#1e1e2e", fg="#cdd6f4", selectcolor="#313244",
                       activebackground="#1e1e2e",
                       font=("Helvetica", 9)).pack(side="right")

        self.lbl_stat = tk.Label(win, text="", bg="#1e1e2e", fg="#f9e2af",
                                 font=("Helvetica", 9))
        self.lbl_stat.pack()

        bf = tk.Frame(win, bg="#1e1e2e"); bf.pack(pady=10)
        tk.Button(bf, text="✓   Utiliser cette référence",
                  command=self._confirm,
                  font=("Helvetica", 11, "bold"),
                  bg="#a6e3a1", fg="#1e1e2e",
                  padx=14, pady=7, relief="flat").pack(side="left", padx=8)
        tk.Button(bf, text="Annuler", command=win.destroy,
                  bg="#313244", fg="#cdd6f4",
                  padx=12, pady=7, relief="flat").pack(side="left", padx=4)

        self._refresh()
        master.wait_window(win)

    # ------------------------------------------------------------------
    def _refresh(self, *_):
        try:
            i_ref = int(self.v_ref.get())
            i_tst = int(self.v_tst.get())
            navg  = max(1, int(self.v_navg.get()))
            inv   = bool(self.v_inv.get())
        except Exception:
            return

        ref = self.analyzer.build_reference(i_ref, navg)
        tst = self.analyzer.read_gray_raw(i_tst)
        if ref is None or tst is None:
            return
        diff = subtract_reference(tst, ref, invert=inv)

        # Statistiques utiles pour juger de la qualité de la référence
        self.lbl_stat.config(
            text=f"Référence : frame {i_ref} (moyenne de {navg})   |   "
                 f"Contrôle : frame {i_tst}   |   "
                 f"Soustraction : min={int(diff.min())}  max={int(diff.max())}  "
                 f"moyenne={diff.mean():.1f}")

        panels = [label_panel(ref,  f"1. Reference (vide) - frame {i_ref}"),
                  label_panel(tst,  f"2. Controle - frame {i_tst}"),
                  label_panel(diff, "3. Reference - Controle" + (" + Invert" if inv else ""))]
        sep    = np.full((panels[0].shape[0], 6, 3), 40, dtype=np.uint8)
        comp   = np.hstack([panels[0], sep, panels[1], sep, panels[2]])
        self._blit(comp)

    def _blit(self, rgb):
        cw = self.canvas.winfo_width()  or 1150
        ch = self.canvas.winfo_height() or 350
        h, w = rgb.shape[:2]
        scale = min(cw / w, ch / h, 1.0)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        small = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        self._tk_img = ImageTk.PhotoImage(Image.fromarray(small))
        self.canvas.delete("all")
        self.canvas.create_image((cw - nw) // 2, (ch - nh) // 2,
                                 anchor="nw", image=self._tk_img)

    def _confirm(self):
        self.analyzer.bg_ref_frame = int(self.v_ref.get())
        self.analyzer.bg_ref_navg  = max(1, int(self.v_navg.get()))
        self.analyzer.bg_invert    = bool(self.v_inv.get())
        self.analyzer.build_reference()
        self.ok = True
        self.win.destroy()


# =============================================================================
# FENÊTRE — PARAMÈTRES D'ANALYSE
# =============================================================================

class AnalysisParamsWindow:
    """
    Modale pour définir :
      • plage de frames (start / end)
      • sous-échantillonnage (1 frame sur N)
      • soustraction du fond + frame de référence
      • mode de comptage (binaire / comptage)
      • paramètres du pipeline ImageJ-like (LUT + threshold + filtres blobs)
    """

    def __init__(self, master, analyzer):
        self.ok       = False
        self.analyzer = analyzer
        win = tk.Toplevel(master)
        win.title("Paramètres d'analyse")
        win.geometry("560x900")
        win.resizable(False, True)
        win.configure(bg="#1e1e2e")
        win.grab_set()
        self.win = win

        # Zone défilante
        outer = tk.Frame(win, bg="#1e1e2e"); outer.pack(fill="both", expand=True)
        cv_scroll = tk.Canvas(outer, bg="#1e1e2e", highlightthickness=0)
        vs = tk.Scrollbar(outer, orient="vertical", command=cv_scroll.yview)
        body = tk.Frame(cv_scroll, bg="#1e1e2e")
        body.bind("<Configure>",
                  lambda e: cv_scroll.configure(scrollregion=cv_scroll.bbox("all")))
        cv_scroll.create_window((0, 0), window=body, anchor="nw", width=530)
        cv_scroll.configure(yscrollcommand=vs.set)
        cv_scroll.pack(side="left", fill="both", expand=True)
        vs.pack(side="right", fill="y")

        def _row(parent, label, var, from_, to_, hint="", inc=1, fmt=None):
            f = tk.Frame(parent, bg="#1e1e2e")
            f.pack(fill="x", padx=22, pady=3)
            tk.Label(f, text=label, bg="#1e1e2e", fg="#cdd6f4",
                     font=("Helvetica", 10), width=32, anchor="w").pack(side="left")
            kw = dict(from_=from_, to=to_, textvariable=var, width=7,
                      bg="#313244", fg="white", font=("Helvetica", 10),
                      increment=inc)
            if fmt:
                kw["format"] = fmt
            tk.Spinbox(f, **kw).pack(side="left")
            if hint:
                tk.Label(f, text=hint, bg="#1e1e2e", fg="#585b70",
                         font=("Helvetica", 8)).pack(side="left", padx=6)

        def _sect(txt):
            tk.Label(body, text=f"\n── {txt} ──", bg="#1e1e2e", fg="#a6adc8",
                     font=("Helvetica", 9, "italic")).pack()

        tk.Label(body, text="⚙   Paramètres d'analyse",
                 font=("Helvetica", 14, "bold"),
                 bg="#1e1e2e", fg="#89b4fa").pack(pady=12)

        # ── Plage de frames ──────────────────────────────────────────────────
        _sect("Plage de frames")
        self.v_start = tk.IntVar(value=analyzer.frame_start)
        self.v_end   = tk.IntVar(value=analyzer.frame_end)
        _row(body, f"Frame de début  (0 – {analyzer.total_frames-1})",
             self.v_start, 0, analyzer.total_frames - 1)
        _row(body, f"Frame de fin    (0 – {analyzer.total_frames-1})",
             self.v_end,   0, analyzer.total_frames - 1)

        _sect("Sous-échantillonnage")
        self.v_step = tk.IntVar(value=analyzer.analysis_step)
        _row(body, "Analyser 1 frame sur N  (1 = toutes)",
             self.v_step, 1, max(100, analyzer.total_frames))

        self.lbl_est = tk.Label(body, text="", bg="#1e1e2e", fg="#f9e2af",
                                font=("Helvetica", 9))
        self.lbl_est.pack()
        for v in (self.v_start, self.v_end, self.v_step):
            v.trace_add("write", lambda *_: self._estimate(analyzer))
        self._estimate(analyzer)

        # ── Soustraction du fond ─────────────────────────────────────────────
        _sect("Soustraction du fond (éclairage non uniforme)")
        tk.Label(body,
                 text="Résultat = Référence(vide) − Image courante   →   particules en blanc",
                 bg="#1e1e2e", fg="#585b70",
                 font=("Helvetica", 8, "italic")).pack()

        self.v_bg = tk.BooleanVar(value=analyzer.bg_subtract)
        f_bg = tk.Frame(body, bg="#1e1e2e"); f_bg.pack(fill="x", padx=22, pady=4)
        tk.Checkbutton(f_bg, text="Activer la soustraction du fond",
                       variable=self.v_bg, command=self._on_bg_toggle,
                       bg="#1e1e2e", fg="#a6e3a1", selectcolor="#313244",
                       activebackground="#1e1e2e",
                       font=("Helvetica", 10, "bold")).pack(side="left")

        f_ref = tk.Frame(body, bg="#1e1e2e"); f_ref.pack(fill="x", padx=22, pady=3)
        self.btn_ref = tk.Button(f_ref, text="🖼   Choisir la frame de référence…",
                                 command=self._choose_ref,
                                 bg="#a6e3a1", fg="#1e1e2e",
                                 font=("Helvetica", 10, "bold"),
                                 padx=10, pady=5, relief="flat")
        self.btn_ref.pack(side="left")
        self.lbl_ref = tk.Label(f_ref, text="", bg="#1e1e2e", fg="#f9e2af",
                                font=("Helvetica", 9))
        self.lbl_ref.pack(side="left", padx=10)

        # ── Mode de comptage ─────────────────────────────────────────────────
        _sect("Mode de comptage")
        self.v_mode = tk.StringVar(value=analyzer.count_mode)
        f_md = tk.Frame(body, bg="#1e1e2e"); f_md.pack(fill="x", padx=22, pady=3)
        tk.Radiobutton(f_md, text="Binaire  (vide / plein)",
                       variable=self.v_mode, value=MODE_BINARY,
                       command=self._on_mode_toggle,
                       bg="#1e1e2e", fg="#cdd6f4", selectcolor="#313244",
                       activebackground="#1e1e2e",
                       font=("Helvetica", 10)).pack(side="left", padx=(0, 16))
        tk.Radiobutton(f_md, text="Comptage  (0, 1, 2, …)",
                       variable=self.v_mode, value=MODE_COUNT,
                       command=self._on_mode_toggle,
                       bg="#1e1e2e", fg="#cdd6f4", selectcolor="#313244",
                       activebackground="#1e1e2e",
                       font=("Helvetica", 10)).pack(side="left")
        self.lbl_mode = tk.Label(body, text="", bg="#1e1e2e", fg="#585b70",
                                 font=("Helvetica", 8, "italic"),
                                 wraplength=480, justify="left")
        self.lbl_mode.pack(padx=22)

        # ── Détection ────────────────────────────────────────────────────────
        _sect("Détection ImageJ-like")
        tk.Label(body, text="setMinAndMax + Apply LUT  →  setThreshold(lo, hi)  →  Blobs",
                 bg="#1e1e2e", fg="#585b70",
                 font=("Helvetica", 8, "italic")).pack()

        self.v_lut_min  = tk.IntVar(value=analyzer.detect_lut_min)
        self.v_lut_max  = tk.IntVar(value=analyzer.detect_lut_max)
        self.v_thr_lo   = tk.IntVar(value=analyzer.detect_thr_lo)
        self.v_thr_hi   = tk.IntVar(value=analyzer.detect_thr_hi)
        self.v_min_area = tk.IntVar(value=analyzer.detect_min_area)
        self.v_max_area = tk.IntVar(value=analyzer.detect_max_area)
        self.v_circ     = tk.DoubleVar(value=analyzer.detect_circularity)

        _row(body, "LUT min  (setMinAndMax bas)",  self.v_lut_min, 0, 255)
        _row(body, "LUT max  (setMinAndMax haut)", self.v_lut_max, 1, 255)
        _row(body, "Seuil BAS   (setThreshold lo)", self.v_thr_lo, 0, 255,
             "objets = pixels DANS la plage")
        _row(body, "Seuil HAUT  (setThreshold hi)", self.v_thr_hi, 0, 255)

        f_pre = tk.Frame(body, bg="#1e1e2e"); f_pre.pack(fill="x", padx=22, pady=4)
        tk.Button(f_pre, text="↺  Préréglage sans soustraction (objets sombres)",
                  command=lambda: self._apply_preset(False),
                  bg="#45475a", fg="#cdd6f4", font=("Helvetica", 8),
                  padx=6, pady=3, relief="flat").pack(side="left", padx=2)
        tk.Button(f_pre, text="↺  Préréglage avec soustraction (objets clairs)",
                  command=lambda: self._apply_preset(True),
                  bg="#45475a", fg="#cdd6f4", font=("Helvetica", 8),
                  padx=6, pady=3, relief="flat").pack(side="left", padx=2)

        _row(body, "Aire min blob (px²)", self.v_min_area, 1, 5000, "filtre bruit")
        _row(body, "Aire max blob (px²)", self.v_max_area, 10, 50000, "filtre artefacts")

        f_circ = tk.Frame(body, bg="#1e1e2e"); f_circ.pack(fill="x", padx=22, pady=3)
        tk.Label(f_circ, text="Circularité min  (0–1)", bg="#1e1e2e", fg="#cdd6f4",
                 font=("Helvetica", 10), width=32, anchor="w").pack(side="left")
        self.sp_circ = tk.Spinbox(f_circ, from_=0.0, to=1.0, textvariable=self.v_circ,
                                  width=7, bg="#313244", fg="white",
                                  font=("Helvetica", 10),
                                  increment=0.05, format="%.2f")
        self.sp_circ.pack(side="left")
        tk.Label(f_circ, text="ignoré en mode binaire", bg="#1e1e2e", fg="#585b70",
                 font=("Helvetica", 8)).pack(side="left", padx=6)

        # ── Amas ─────────────────────────────────────────────────────────────
        _sect("Détection d'amas (mode comptage uniquement)")
        self.v_watershed  = tk.BooleanVar(value=analyzer.detect_watershed)
        self.v_particle_r = tk.IntVar(value=analyzer.detect_particle_r)

        f_ws = tk.Frame(body, bg="#1e1e2e"); f_ws.pack(fill="x", padx=22, pady=3)
        self.cb_ws = tk.Checkbutton(f_ws, text="Activer la détection d'amas (Hough local)",
                                    variable=self.v_watershed,
                                    bg="#1e1e2e", fg="#cdd6f4", selectcolor="#313244",
                                    activebackground="#1e1e2e",
                                    font=("Helvetica", 10))
        self.cb_ws.pack(side="left")
        _row(body, "Rayon nominal d'une particule (px)",
             self.v_particle_r, 1, 100, "→ seuil amas = 1.25 × πr²")

        # ── Boutons ──────────────────────────────────────────────────────────
        bf = tk.Frame(win, bg="#181825", pady=8)
        bf.pack(fill="x", side="bottom")
        tk.Button(bf, text="✓   Lancer l'analyse",
                  command=lambda: self._confirm(analyzer),
                  font=("Helvetica", 11, "bold"),
                  bg="#89b4fa", fg="#1e1e2e",
                  padx=14, pady=7, relief="flat").pack(side="left", padx=8)
        tk.Button(bf, text="🔍  Prévisualiser pipeline",
                  command=lambda: self._preview(analyzer),
                  font=("Helvetica", 10),
                  bg="#cba6f7", fg="#1e1e2e",
                  padx=10, pady=7, relief="flat").pack(side="left", padx=4)
        tk.Button(bf, text="Annuler", command=win.destroy,
                  bg="#313244", fg="#cdd6f4",
                  padx=10, pady=7, relief="flat").pack(side="left", padx=4)

        self._on_bg_toggle()
        self._on_mode_toggle()
        master.wait_window(win)

    # ------------------------------------------------------------------
    def _apply_preset(self, with_subtraction):
        """Valeurs de départ raisonnables selon la polarité des objets."""
        if with_subtraction:
            # Après soustraction : particules CLAIRES sur fond noir.
            # Équivalent du setMinAndMax(0,40) + seuil raw 10 du script ImageJ :
            # 10 en brut ↔ 10 × 255/40 ≈ 64 après remap.
            self.v_lut_min.set(0)
            self.v_lut_max.set(40)
            self.v_thr_lo.set(64)
            self.v_thr_hi.set(255)
        else:
            # Sans soustraction : particules SOMBRES sur fond clair.
            self.v_lut_min.set(19)
            self.v_lut_max.set(109)
            self.v_thr_lo.set(0)
            self.v_thr_hi.set(155)

    def _on_bg_toggle(self):
        on = bool(self.v_bg.get())
        self.btn_ref.config(state="normal" if on else "disabled")
        if on:
            self.lbl_ref.config(
                text=f"référence : frame {self.analyzer.bg_ref_frame} "
                     f"(moy. {self.analyzer.bg_ref_navg})")
        else:
            self.lbl_ref.config(text="désactivée")

    def _on_mode_toggle(self):
        binary = (self.v_mode.get() == MODE_BINARY)
        state  = "disabled" if binary else "normal"
        try:
            self.sp_circ.config(state=state)
            self.cb_ws.config(state=state)
        except Exception:
            pass
        self.lbl_mode.config(
            text=("Mode binaire : un piège est déclaré plein dès qu'il contient un objet "
                  "d'aire ≥ « aire min ». Circularité et détection d'amas sont ignorées. "
                  "C'est le mode à privilégier pour des objets non sphériques ou accolés."
                  if binary else
                  "Mode comptage : filtre de circularité puis séparation des amas par "
                  "Hough local. Plus informatif, mais plus sensible aux réglages."))

    def _choose_ref(self):
        w = ReferenceFrameWindow(self.win, self.analyzer)
        if w.ok:
            self.lbl_ref.config(
                text=f"référence : frame {self.analyzer.bg_ref_frame} "
                     f"(moy. {self.analyzer.bg_ref_navg})"
                     + ("  + invert" if self.analyzer.bg_invert else ""))
            if messagebox.askyesno(
                    "Préréglage",
                    "Appliquer le préréglage de seuils adapté à la soustraction "
                    "(objets clairs sur fond noir) ?"):
                self._apply_preset(True)

    def _push_to_analyzer(self, analyzer):
        """Recopie les widgets dans l'analyseur (utilisé par preview et confirm)."""
        analyzer.bg_subtract        = bool(self.v_bg.get())
        analyzer.count_mode         = self.v_mode.get()
        analyzer.detect_lut_min     = self.v_lut_min.get()
        analyzer.detect_lut_max     = self.v_lut_max.get()
        analyzer.detect_thr_lo      = self.v_thr_lo.get()
        analyzer.detect_thr_hi      = self.v_thr_hi.get()
        analyzer.detect_min_area    = self.v_min_area.get()
        analyzer.detect_max_area    = self.v_max_area.get()
        analyzer.detect_circularity = round(self.v_circ.get(), 3)
        analyzer.detect_watershed   = self.v_watershed.get()
        analyzer.detect_particle_r  = self.v_particle_r.get()

    def _preview(self, analyzer):
        """
        Prévisualisation — 6 panneaux :
          ① Original     ② Après soustraction    ③ Après LUT
          ④ Threshold + Fill Holes               ⑤ Blobs retenus
          ⑥ Overlay des détections
        """
        try:
            self._push_to_analyzer(analyzer)

            mid_fi = (self.v_start.get() + self.v_end.get()) // 2
            gray0  = analyzer.read_gray_raw(mid_fi)
            if gray0 is None:
                messagebox.showwarning("Prévisualisation", "Impossible de lire la frame.")
                return

            if analyzer.bg_subtract:
                analyzer.build_reference()
            gray = analyzer.preprocess_gray(gray0)

            lut_min = analyzer.detect_lut_min
            lut_max = analyzer.detect_lut_max
            thr_lo  = analyzer.detect_thr_lo
            thr_hi  = analyzer.detect_thr_hi
            mode    = analyzer.count_mode

            mask_thr, remapped = imagej_preprocess(gray, lut_min, lut_max, thr_lo, thr_hi)
            mask_morph, _      = build_mask(gray, lut_min, lut_max, thr_lo, thr_hi)

            # Blobs retenus après filtre d'aire
            mask_blobs = np.zeros_like(mask_morph)
            cnts_p, _  = cv2.findContours(mask_morph, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
            n_kept = 0
            for cnt_p in cnts_p:
                a = cv2.contourArea(cnt_p)
                if analyzer.detect_min_area <= a <= analyzer.detect_max_area:
                    cv2.drawContours(mask_blobs, [cnt_p], -1, 255, cv2.FILLED)
                    n_kept += 1

            # Overlay final (détections par ROI)
            frame_bgr   = analyzer.read_frame_raw(mid_fi)
            overlay_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).copy()
            n_occ = 0
            for (x1, y1, x2, y2) in analyzer.rois:
                cnt, particles = analyzer.count_in_roi(gray, (x1, y1, x2, y2))
                if cnt > 0:
                    n_occ += 1
                col = COLORS_RGB.get(min(cnt, 4), (200, 20, 20))
                cv2.rectangle(overlay_rgb, (x1, y1), (x2, y2), col, 2)
                for (px, py, pr) in particles:
                    cv2.circle(overlay_rgb, (x1+px, y1+py), max(2, pr), col, 1)
                if cnt > 0:
                    cx_t = (x1+x2)//2;  cy_t = (y1+y2)//2
                    cv2.putText(overlay_rgb, str(cnt), (cx_t-4, cy_t+4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1)

            sub_txt = (f"② Soustraction\nréf. frame {analyzer.bg_ref_frame}"
                       + (" + invert" if analyzer.bg_invert else "")
                       if analyzer.bg_subtract else "② Soustraction\n(désactivée)")

            fig, axes = plt.subplots(2, 3, figsize=(18, 9), facecolor='white')
            axes = axes.ravel()
            titles = [
                "① Original (8-bit)",
                sub_txt,
                f"③ Apply LUT\n[{lut_min}–{lut_max}]→[0–255]",
                f"④ Threshold [{thr_lo}–{thr_hi}]\n+ Fill Holes + morpho",
                f"⑤ Blobs retenus\naire ∈ [{analyzer.detect_min_area}–"
                f"{analyzer.detect_max_area}]  →  {n_kept} blobs",
                f"⑥ Overlay — {n_occ}/{len(analyzer.rois)} pièges occupés",
            ]
            images = [gray0, gray, remapped, mask_morph, mask_blobs, overlay_rgb]
            cmaps  = ['gray', 'gray', 'gray', 'gray', 'gray', None]

            for ax, img, title, cmap in zip(axes, images, titles, cmaps):
                ax.set_facecolor('white')
                ax.spines[:].set_color('#cccccc')
                if cmap:
                    ax.imshow(img, cmap=cmap, vmin=0, vmax=255)
                else:
                    ax.imshow(img)
                ax.set_title(title, color='black', fontsize=9)
                ax.axis('off')

            mode_txt = "BINAIRE (vide/plein)" if mode == MODE_BINARY else "COMPTAGE"
            fig.suptitle(f"Prévisualisation — frame {mid_fi}  |  mode {mode_txt}",
                         color='black', fontsize=11)
            fig.tight_layout()
            plt.show(block=False)

        except Exception as e:
            messagebox.showerror("Erreur prévisualisation", str(e))

    def _estimate(self, analyzer):
        try:
            s    = self.v_start.get()
            e    = self.v_end.get()
            step = max(1, self.v_step.get())
            n    = len(range(s, e + 1, step))
            nt   = len(analyzer.rois) if analyzer.rois else "?"
            self.lbl_est.config(text=f"→  {n} frames × {nt} pièges à analyser")
        except Exception:
            pass

    def _confirm(self, analyzer):
        try:
            s = self.v_start.get()
            e = self.v_end.get()
            if s > e:
                messagebox.showwarning("", "Frame de début > frame de fin.")
                return
            if self.v_thr_lo.get() > self.v_thr_hi.get():
                messagebox.showwarning("", "Seuil bas > seuil haut.")
                return
            if self.v_bg.get() and analyzer._bg_ref_gray is None:
                if not messagebox.askyesno(
                        "Référence",
                        "Aucune frame de référence n'a été choisie explicitement.\n"
                        f"Utiliser la frame {analyzer.bg_ref_frame} par défaut ?"):
                    return
            analyzer.frame_start   = s
            analyzer.frame_end     = e
            analyzer.analysis_step = max(1, self.v_step.get())
            self._push_to_analyzer(analyzer)
            self.ok = True
            self.win.destroy()
        except Exception as ex:
            messagebox.showerror("Erreur", str(ex))


# =============================================================================
# WIZARD DE CONFIGURATION
# =============================================================================

class SetupWizard:

    STEPS = [
        ("Correction de tilt",
         "Clic GAUCHE sur un piège  →  puis clic DROIT sur un autre piège sur la MÊME LIGNE.\n"
         "Le programme calcule l'angle et redresse la vidéo automatiquement."),
        ("Zone d'analyse (Crop)",
         "Clic GAUCHE = coin HAUT-GAUCHE de la zone à analyser.\n"
         "Clic DROIT  = coin BAS-DROIT   de la zone.  (Rectangle vert = sélection)"),
        ("Calibration 1/7 — Piège haut-gauche TL",
         "Clic GAUCHE sur le coin HAUT-GAUCHE du piège le plus en HAUT-GAUCHE de la matrice.  [Coin A]"),
        ("Calibration 2/7 — Piège haut-gauche BR",
         "Clic GAUCHE sur le coin BAS-DROIT du MÊME piège  (donne la taille d'un piège)."),
        ("Calibration 3/7 — Colonne suivante",
         "Clic GAUCHE sur le coin HAUT-GAUCHE du piège sur la COLONNE SUIVANTE (même ligne que A)."),
        ("Calibration 4/7 — Ligne suivante",
         "Clic GAUCHE sur le coin HAUT-GAUCHE du piège sur la LIGNE SUIVANTE (même colonne que A)."),
        ("Calibration 5/7 — Coin HAUT-DROIT de la matrice",
         "Clic GAUCHE sur le coin HAUT-GAUCHE du piège le plus en HAUT-DROITE de la matrice.  [Coin B]"),
        ("Calibration 6/7 — Coin BAS-GAUCHE de la matrice",
         "Clic GAUCHE sur le coin HAUT-GAUCHE du piège le plus en BAS-GAUCHE de la matrice.  [Coin C]"),
        ("Calibration 7/7 — Coin BAS-DROIT de la matrice",
         "Clic GAUCHE sur le coin HAUT-GAUCHE du piège le plus en BAS-DROITE de la matrice.  [Coin D]"),
        ("Confirmation",
         "Calibration terminée ! Les ROIs interpolées sont affichées en cyan.\n"
         "Cliquez sur 'Suivant' pour définir les paramètres d'analyse et lancer."),
    ]

    def __init__(self, master, analyzer):
        self.master          = master
        self.analyzer        = analyzer
        self.step            = 0
        self._tk_img         = None
        self._img_offset     = (0, 0)
        self._display_scale  = 1.0
        self._tilt_pt1       = None
        self._crop_pt1       = None
        self._calib_pts      = []

        win = tk.Toplevel(master)
        win.title(f"{APP_TITLE}  —  Configuration")
        win.geometry("1160x800")
        win.configure(bg="#1e1e2e")
        win.protocol("WM_DELETE_WINDOW", self._abort)
        self.win = win

        self.lbl_step = tk.Label(win, text="", font=("Helvetica", 12, "bold"),
                                 bg="#1e1e2e", fg="#89b4fa")
        self.lbl_step.pack(padx=12, pady=(10, 2))

        self.lbl_info = tk.Label(win, text="", font=("Helvetica", 10),
                                 bg="#1e1e2e", fg="#cdd6f4",
                                 wraplength=1130, justify="left")
        self.lbl_info.pack(padx=12, pady=(0, 6))

        self.canvas = tk.Canvas(win, bg="#0d0d14", cursor="crosshair",
                                highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=4)
        self.canvas.bind("<Button-1>",  self._left_click)
        self.canvas.bind("<Button-3>",  self._right_click)
        self.canvas.bind("<Configure>", lambda e: self._refresh_display())

        bar = tk.Frame(win, bg="#181825", pady=6)
        bar.pack(fill="x", padx=8)

        tk.Button(bar, text="⬆  Importer config", command=self._import,
                  bg="#a6e3a1", fg="#1e1e2e", font=("Helvetica", 9, "bold"),
                  padx=8, pady=4, relief="flat").pack(side="left", padx=4)

        self.btn_back = tk.Button(bar, text="← Retour", command=self._back,
                                  bg="#45475a", fg="white",
                                  padx=10, pady=4, relief="flat")
        self.btn_back.pack(side="left", padx=4)

        self.btn_next = tk.Button(bar, text="Suivant →", command=self._next,
                                  bg="#89b4fa", fg="#1e1e2e",
                                  font=("Helvetica", 10, "bold"),
                                  padx=14, pady=4, relief="flat")
        self.btn_next.pack(side="right", padx=4)

        self.lbl_status = tk.Label(bar, text="", bg="#181825", fg="#f9e2af",
                                   font=("Helvetica", 9))
        self.lbl_status.pack(side="right", padx=12)

        self._load_step()

    # ── Navigation ──────────────────────────────────────────────────────────
    def _load_step(self):
        title, info = self.STEPS[self.step]
        self.lbl_step.config(text=f"Étape {self.step+1}/{len(self.STEPS)}  —  {title}")
        self.lbl_info.config(text=info)
        self.btn_back.config(state="normal" if self.step > 0 else "disabled")
        self._refresh_display()

    def _refresh_display(self):
        if self.step == 0:
            frame = self.analyzer.get_preview_frame(0)
        else:
            frame = self.analyzer.read_frame_raw(0)
        if frame is None:
            return
        if self.step == 9 and self.analyzer.rois:
            frame = frame.copy()
            for (x1, y1, x2, y2) in self.analyzer.rois:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 1)
        self._blit(frame)
        self._redraw_marks()

    def _blit(self, frame_bgr):
        cw = self.canvas.winfo_width()  or 1130
        ch = self.canvas.winfo_height() or 700
        h, w = frame_bgr.shape[:2]
        scale = min(cw / w, ch / h, 1.0)
        self._display_scale = scale
        nw = int(w * scale);  nh = int(h * scale)
        ox = (cw - nw) // 2;  oy = (ch - nh) // 2
        self._img_offset = (ox, oy)
        rgb = cv2.cvtColor(cv2.resize(frame_bgr, (nw, nh)), cv2.COLOR_BGR2RGB)
        self._tk_img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete("all")
        self.canvas.create_image(ox, oy, anchor="nw", image=self._tk_img)

    def _img_coords(self, cx, cy):
        ox, oy = self._img_offset
        sc = self._display_scale
        return int((cx - ox) / sc), int((cy - oy) / sc)

    def _canvas_xy(self, ix, iy):
        ox, oy = self._img_offset
        sc = self._display_scale
        return int(ix * sc + ox), int(iy * sc + oy)

    def _dot(self, ix, iy, color="yellow", label="", r=6):
        cx, cy = self._canvas_xy(ix, iy)
        self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r,
                                outline=color, width=2, tags="mk")
        if label:
            self.canvas.create_text(cx + r + 5, cy, text=label, fill=color,
                                    tags="mk", font=("Helvetica", 9, "bold"))

    def _rect(self, ix1, iy1, ix2, iy2, color="lime"):
        x1, y1 = self._canvas_xy(ix1, iy1)
        x2, y2 = self._canvas_xy(ix2, iy2)
        self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=1,
                                     dash=(5, 3), tags="mk")

    def _redraw_marks(self):
        self.canvas.delete("mk")
        calib_labels = ["TL", "BR", "Col+1", "Row+1", "TR", "BL", "BR-last"]
        calib_colors = ["#fab387", "#f38ba8", "#a6e3a1", "#89dceb",
                        "#cba6f7", "#f9e2af", "#ff6b6b"]

        if self._tilt_pt1:
            self._dot(*self._tilt_pt1, "#89dceb", "L1")
        if self._crop_pt1:
            self._dot(*self._crop_pt1, "#a6e3a1", "↖")
        if self.analyzer.crop_rect:
            x1, y1, x2, y2 = self.analyzer.crop_rect
            self._rect(x1, y1, x2, y2, "#a6e3a1")
        for i, pt in enumerate(self._calib_pts):
            if pt:
                self._dot(*pt, calib_colors[i], calib_labels[i])

    # ── Clics ───────────────────────────────────────────────────────────────
    def _left_click(self, event):
        ix, iy = self._img_coords(event.x, event.y)

        if self.step == 0:
            self._tilt_pt1 = (ix, iy)
            self.lbl_status.config(text=f"Point L1 ({ix},{iy}) — clic DROIT pour L2")
        elif self.step == 1:
            self._crop_pt1 = (ix, iy)
            self.analyzer.crop_rect = None
            self.lbl_status.config(text=f"Coin ↖ ({ix},{iy}) — clic DROIT pour ↘")
        elif 2 <= self.step <= 8:
            n = self.step - 2
            while len(self._calib_pts) <= n:
                self._calib_pts.append(None)
            self._calib_pts[n] = (ix, iy)
            labels = ["TL", "BR", "Col+1", "Row+1", "TR", "BL", "BR-last"]
            self.lbl_status.config(text=f"{labels[n]} → ({ix},{iy})")

        self._redraw_marks()

    def _right_click(self, event):
        ix, iy = self._img_coords(event.x, event.y)

        if self.step == 0:
            if self._tilt_pt1 is None:
                messagebox.showwarning("Tilt", "Faites d'abord un clic gauche.")
                return
            dx = ix - self._tilt_pt1[0]
            dy = iy - self._tilt_pt1[1]
            angle = np.degrees(np.arctan2(dy, dx))
            self.analyzer.rotation_angle = angle
            self.analyzer._bg_ref_gray = None      # géométrie changée
            self.lbl_status.config(
                text=f"Ligne inclinée de {angle:.2f}°  →  correction appliquée  —  appuyez Suivant")
            self._refresh_display()

        elif self.step == 1:
            if self._crop_pt1 is None:
                messagebox.showwarning("Crop", "Faites d'abord un clic gauche.")
                return
            x1, y1 = self._crop_pt1
            self.analyzer.crop_rect = (min(x1,ix), min(y1,iy), max(x1,ix), max(y1,iy))
            self.analyzer._bg_ref_gray = None      # géométrie changée
            self.lbl_status.config(
                text=f"Zone : ({min(x1,ix)},{min(y1,iy)}) → ({max(x1,ix)},{max(y1,iy)})")
            self._refresh_display()

    # ── Boutons ─────────────────────────────────────────────────────────────
    def _next(self):
        if self.step == 0 and self._tilt_pt1 is None:
            if not messagebox.askyesno("Tilt",
                    "Aucun tilt défini. Continuer sans correction d'angle ?"):
                return

        if self.step == 1 and self.analyzer.crop_rect is None:
            if not messagebox.askyesno("Crop",
                    "Aucun crop défini. Utiliser l'image entière ?"):
                return

        if 2 <= self.step <= 8:
            n = self.step - 2
            if len(self._calib_pts) <= n or self._calib_pts[n] is None:
                messagebox.showwarning("Calibration", "Cliquez d'abord sur le point demandé.")
                return

        if self.step == 8:
            pts = self._calib_pts
            self.analyzer.trap_tl      = pts[0]
            self.analyzer.trap_br      = pts[1]
            self.analyzer.next_col_tl  = pts[2]
            self.analyzer.next_row_tl  = pts[3]
            self.analyzer.corner_tr    = pts[4]
            self.analyzer.corner_bl    = pts[5]
            self.analyzer.last_trap_tl = pts[6]
            try:
                self.analyzer.compute_rois()
            except Exception as e:
                messagebox.showerror("Erreur ROI", str(e));  return
            base = os.path.splitext(self.analyzer.video_path)[0]
            self.analyzer.save_config(base + "_config.json")
            self.lbl_status.config(
                text=f"✓  {self.analyzer.n_rows}×{self.analyzer.n_cols}"
                     f" = {len(self.analyzer.rois)} pièges  |  config sauvegardée")

        if self.step >= len(self.STEPS) - 1:
            self._finish()
            return

        self.step += 1
        self._load_step()

    def _back(self):
        if self.step > 0:
            self.step -= 1
            self._load_step()

    def _import(self):
        path = filedialog.askopenfilename(title="Importer config JSON",
                                          filetypes=[("JSON", "*.json"), ("Tous", "*.*")])
        if not path:
            return
        try:
            self.analyzer.load_config(path)
            self.analyzer.compute_rois()
            messagebox.showinfo("Import OK",
                f"{self.analyzer.n_rows}×{self.analyzer.n_cols} pièges chargés.")
            self._finish()
        except Exception as e:
            messagebox.showerror("Erreur", str(e))

    def _finish(self):
        self.win.destroy()
        self.master.event_generate("<<SetupDone>>")

    def _abort(self):
        if messagebox.askyesno("Quitter", "Abandonner la configuration ?"):
            self.win.destroy()
            self.master.destroy()


# =============================================================================
# BARRE DE PROGRESSION
# =============================================================================

class ProgressBar:
    def __init__(self, master, total):
        win = tk.Toplevel(master)
        win.title("Analyse en cours…")
        win.geometry("420x130")
        win.resizable(False, False)
        win.configure(bg="#1e1e2e")
        win.grab_set()
        self.win = win
        tk.Label(win, text="Analyse des frames en cours…",
                 bg="#1e1e2e", fg="#cdd6f4",
                 font=("Helvetica", 11)).pack(pady=12)
        self.bar = ttk.Progressbar(win, maximum=total, length=380, mode="determinate")
        self.bar.pack(pady=4)
        self.lbl = tk.Label(win, text="0 / 0", bg="#1e1e2e", fg="#a6adc8")
        self.lbl.pack()
        win.update()

    def update(self, cur, total):
        self.bar['value'] = cur
        self.lbl.config(text=f"{cur} / {total}")
        self.win.update()

    def close(self):
        self.win.destroy()


# =============================================================================
# FENÊTRE DE VISUALISATION
# =============================================================================

def white_fig(*args, **kwargs):
    fig, axes = plt.subplots(*args, facecolor='white', **kwargs)
    axlist = axes if hasattr(axes, '__iter__') else [axes]
    for ax in axlist:
        ax.set_facecolor('white')
        ax.spines[:].set_color('#cccccc')
        ax.tick_params(colors='black')
    return fig, axes


class VisualizationWindow:

    def __init__(self, master, analyzer):
        self.master   = master
        self.analyzer = analyzer
        self.n_an     = len(analyzer.analyzed_frames)
        self.cur_idx  = 0
        self.playing  = False
        self.binary   = (analyzer.count_mode == MODE_BINARY)

        self.fig = plt.figure(figsize=(16, 9), facecolor='white')
        gs = GridSpec(3, 2, figure=self.fig, height_ratios=[5, 5, 3],
                      hspace=0.38, wspace=0.28)
        self.ax_vid = self.fig.add_subplot(gs[0:2, 0])
        self.ax_mat = self.fig.add_subplot(gs[0:2, 1])
        self.ax_gr  = self.fig.add_subplot(gs[2, :])
        for ax in (self.ax_vid, self.ax_mat, self.ax_gr):
            ax.set_facecolor('white')
            ax.spines[:].set_color('#cccccc')
            ax.tick_params(colors='black')

        mode_txt = "mode binaire (vide/plein)" if self.binary else "mode comptage"
        bg_txt   = (f"soustraction réf. frame {analyzer.bg_ref_frame}"
                    if analyzer.bg_subtract else "sans soustraction")
        self.fig.suptitle(
            f"{APP_TITLE}  —  {os.path.basename(analyzer.video_path)}"
            f"   [{mode_txt}, {bg_txt}]",
            color='black', fontsize=12, y=0.99)

        total = analyzer.counts_matrix.sum(axis=(1, 2))
        times = np.array(analyzer.analyzed_frames) / analyzer.fps
        ylab  = "Pièges occupés" if self.binary else "Particules"
        self.ax_gr.plot(times, total, color='steelblue', lw=2,
                        label='Pièges occupés' if self.binary else 'Particules totales')
        dt = (times[1] - times[0]) / 2 if len(times) > 1 else 0.5
        for i, oof in enumerate(analyzer.oof_flags):
            if oof and i < len(times):
                self.ax_gr.axvspan(times[i]-dt, times[i]+dt, color='salmon', alpha=0.30)
        self.vline = self.ax_gr.axvline(times[0] if len(times) else 0,
                                        color='red', lw=1.5, ls='--',
                                        label='Frame courante')
        self.ax_gr.set_xlabel("Temps (s)", color='black', fontsize=9)
        self.ax_gr.set_ylabel(ylab, color='black', fontsize=9)
        self.ax_gr.legend(fontsize=8, loc='upper left')

        self.ax_sl = self.fig.add_axes([0.12, 0.012, 0.68, 0.018], facecolor='#eeeeee')
        self.slider = Slider(self.ax_sl, 'Frame', 0, max(self.n_an - 1, 1),
                             valinit=0, valstep=1, color='steelblue')
        self.slider.label.set_color('black')
        self.slider.valtext.set_color('black')
        self.slider.on_changed(lambda v: self._update(int(v)))

        from matplotlib.widgets import Button as MplBtn
        self.ax_play = self.fig.add_axes([0.82, 0.005, 0.08, 0.030], facecolor='#eeeeee')
        self.btn_play = MplBtn(self.ax_play, '▶ Play', color='#eeeeee', hovercolor='#dddddd')
        self.btn_play.label.set_color('black')
        self.btn_play.on_clicked(self._toggle_play)

        self.fig.canvas.mpl_connect('close_event', self._on_close)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        self.fig.text(0.01, 0.01,
                      "Appuyez sur  G  pour ouvrir le dialogue de sauvegarde",
                      fontsize=8, color='gray', ha='left', va='bottom')

        self._update(0)
        plt.show(block=False)

    # ── Légende adaptée au mode ───────────────────────────────────────────
    def _legend_patches(self):
        keys = [0, 1] if self.binary else sorted(COLORS_RGB)
        return [mpatches.Patch(color=np.array(COLORS_RGB[k])/255,
                               label=legend_label(k, self.analyzer.count_mode))
                for k in keys]

    def _update(self, idx):
        idx = int(np.clip(idx, 0, self.n_an - 1))
        self.cur_idx = idx
        fi  = self.analyzer.analyzed_frames[idx]
        counts_flat = self.analyzer.counts_matrix[idx].ravel()

        frame = self.analyzer.read_frame_raw(fi)
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()
            for ti, (x1, y1, x2, y2) in enumerate(self.analyzer.rois):
                cnt = int(counts_flat[ti])
                col = COLORS_RGB.get(min(cnt, 4), (200, 20, 20))
                cv2.rectangle(rgb, (x1, y1), (x2, y2), col, 2)
                if cnt > 0 and not self.binary:
                    cx_t = (x1+x2)//2;  cy_t = (y1+y2)//2
                    cv2.putText(rgb, str(cnt), (cx_t-4, cy_t+4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
            if (idx < len(self.analyzer.oof_flags) and self.analyzer.oof_flags[idx]):
                cv2.putText(rgb, "! OUT OF FOCUS", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.70, (200, 20, 20), 2)
            self.ax_vid.clear()
            self.ax_vid.imshow(rgb)
            self.ax_vid.set_title(f"Frame {fi}  (analysée {idx+1}/{self.n_an})",
                                  color='black', fontsize=9)
            self.ax_vid.axis('off')

        mat  = self.analyzer.counts_matrix[idx]
        cimg = np.zeros((*mat.shape, 3), dtype=np.uint8)
        for r in range(self.analyzer.n_rows):
            for c in range(self.analyzer.n_cols):
                cimg[r, c] = COLORS_RGB.get(min(int(mat[r, c]), 4), (200, 20, 20))
        self.ax_mat.clear()
        self.ax_mat.set_facecolor('white')
        self.ax_mat.imshow(cimg, interpolation='nearest', aspect='auto')
        fs = max(5, min(9, 60 // max(self.analyzer.n_rows, self.analyzer.n_cols, 1)))
        if not self.binary:
            for r in range(self.analyzer.n_rows):
                for c in range(self.analyzer.n_cols):
                    self.ax_mat.text(c, r, str(int(mat[r, c])),
                                     ha='center', va='center', color='white',
                                     fontsize=fs, fontweight='bold')
        self.ax_mat.legend(handles=self._legend_patches(), loc='upper right',
                           fontsize=7,
                           title="état" if self.binary else "particules",
                           title_fontsize=7)
        self.ax_mat.set_title("Matrice des pièges", color='black', fontsize=9)
        self.ax_mat.tick_params(colors='black', labelsize=7)

        t = fi / self.analyzer.fps
        self.vline.set_xdata([t, t])
        self.fig.canvas.draw_idle()

    # ── Play / Pause ──────────────────────────────────────────────────────
    def _toggle_play(self, event=None):
        self.playing = not self.playing
        self.btn_play.label.set_text('⏸ Pause' if self.playing else '▶ Play')
        if self.playing:
            self._play_next()

    def _play_next(self):
        if not self.playing:
            return
        nxt = (self.cur_idx + 1) % self.n_an
        self.slider.set_val(nxt)
        delay = max(20, int(1000 / self.analyzer.fps))
        try:
            self.fig.canvas.get_tk_widget().after(delay, self._play_next)
        except Exception:
            self.playing = False

    def _on_close(self, event=None):
        self.playing = False

    def _on_key(self, event):
        if event.key in ('g', 'G'):
            self._interactive_save()

    # ── Sauvegarde interactive ────────────────────────────────────────────
    def _interactive_save(self):
        base = os.path.splitext(self.analyzer.video_path)[0]
        saved_frames = []

        print("\n[Sauvegarde automatique : timeline…]")
        self._fig_timeline(base)

        while True:
            win = tk.Tk() if not self.master.winfo_exists() else tk.Toplevel(self.master)
            win.title("Sauvegarder une frame")
            win.geometry("420x220")
            win.configure(bg="white")
            win.grab_set()
            win.lift()

            tk.Label(win, text="📸  Sauvegarder une frame",
                     font=("Helvetica", 13, "bold"),
                     bg="white", fg="black").pack(pady=12)

            fi_max = self.analyzer.analyzed_frames[-1]
            fi_min = self.analyzer.analyzed_frames[0]
            tk.Label(win, text=f"Frames analysées : {fi_min} → {fi_max}",
                     bg="white", fg="#555555", font=("Helvetica", 9)).pack()

            row_f = tk.Frame(win, bg="white"); row_f.pack(pady=8)
            tk.Label(row_f, text="Numéro de frame :", bg="white",
                     font=("Helvetica", 10)).pack(side="left", padx=6)
            v_fi = tk.IntVar(value=self.analyzer.analyzed_frames[self.cur_idx])
            tk.Spinbox(row_f, from_=fi_min, to=fi_max, textvariable=v_fi,
                       width=7, font=("Helvetica", 10)).pack(side="left")

            choice = tk.StringVar(value="none")
            bf = tk.Frame(win, bg="white"); bf.pack(pady=10)
            tk.Button(bf, text="💾  Sauvegarder cette frame",
                      command=lambda: choice.set("save"),
                      bg="#4a90d9", fg="white", font=("Helvetica", 10, "bold"),
                      padx=10, pady=5, relief="flat").pack(side="left", padx=6)
            tk.Button(bf, text="✓  Terminer", command=lambda: choice.set("done"),
                      bg="#5cb85c", fg="white", font=("Helvetica", 10),
                      padx=10, pady=5, relief="flat").pack(side="left", padx=6)

            win.wait_variable(choice)
            fi_chosen = v_fi.get()
            win.destroy()

            if choice.get() == "done":
                break

            ai = int(np.argmin(np.abs(np.array(self.analyzer.analyzed_frames) - fi_chosen)))
            fi_actual = self.analyzer.analyzed_frames[ai]
            tag = f"frame{fi_actual:05d}"
            print(f"  → Sauvegarde frame {fi_actual}…")
            self._save_frame_figures(base, ai, fi_actual, tag)
            saved_frames.append((fi_actual, ai))

        if saved_frames:
            self._export_excel(base, saved_frames)

        out_dir = os.path.dirname(base) or os.getcwd()
        print(f"[OK] Répertoire : {out_dir}")
        try:
            msg = "Timeline sauvegardée.\n"
            if saved_frames:
                msg += f"{len(saved_frames)} frame(s) exportée(s) + Excel.\n"
            msg += f"\nDossier : {out_dir}"
            messagebox.showinfo("Sauvegarde terminée", msg)
        except Exception:
            pass

    def _save_frame_figures(self, base, ai, fi, tag):
        counts_flat = self.analyzer.counts_matrix[ai].ravel()
        mat         = self.analyzer.counts_matrix[ai]
        n_traps     = self.analyzer.n_rows * self.analyzer.n_cols
        n_filled    = int(np.sum(counts_flat > 0))
        total       = int(counts_flat.sum())

        frame = self.analyzer.read_frame_raw(fi)
        if frame is not None:
            gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            proc     = self.analyzer.preprocess_gray(gray_raw)
            overlay  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()
            for ti, (x1, y1, x2, y2) in enumerate(self.analyzer.rois):
                cnt = int(counts_flat[ti])
                col = COLORS_RGB.get(min(cnt, 4), (200, 20, 20))
                cv2.rectangle(overlay, (x1, y1), (x2, y2), col, 2)
                if cnt == 0:
                    continue
                _c, particles = self.analyzer.count_in_roi(proc, (x1, y1, x2, y2))
                for (px, py, pr) in particles:
                    cv2.circle(overlay, (x1+px, y1+py), max(2, int(pr)), col, 1)
                if not self.binary:
                    cx_t = (x1+x2)//2; cy_t = (y1+y2)//2
                    cv2.putText(overlay, str(cnt), (cx_t-4, cy_t+4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

            fig_img, ax_img = plt.subplots(figsize=(12, 8), facecolor='white')
            ax_img.set_facecolor('white')
            ax_img.imshow(overlay)
            head = (f"Frame {fi}  —  {n_filled}/{n_traps} pièges occupés"
                    if self.binary else
                    f"Frame {fi}  —  {total} particules  ({n_filled}/{n_traps} pièges remplis)")
            ax_img.set_title(head, fontsize=11, color='black')
            ax_img.axis('off')
            fig_img.tight_layout()
            p = f"{base}_{tag}_overlay.svg"
            fig_img.savefig(p, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig_img)
            print(f"    • {p}")

        colors_l = [np.array(COLORS_RGB.get(k, (128,128,128)))/255 for k in range(5)]
        fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5), facecolor='white')

        ax_hist = axes2[0]
        ax_hist.set_facecolor('white')
        ax_hist.spines[:].set_color('#cccccc')
        vals, cnts_v = np.unique(counts_flat, return_counts=True)
        ax_hist.bar(vals, cnts_v, color=[colors_l[min(int(v), 4)] for v in vals],
                    edgecolor='black', linewidth=0.7)
        if self.binary:
            ax_hist.set_xticks([0, 1])
            ax_hist.set_xticklabels(["vide", "plein"])
            ax_hist.set_xlabel("État du piège", color='black')
        else:
            ax_hist.set_xlabel("Nb particules / piège", color='black')
        ax_hist.set_ylabel("Nb de pièges", color='black')
        ax_hist.set_title(f"Distribution — frame {fi}", color='black')
        ax_hist.tick_params(colors='black')

        ax_mat2 = axes2[1]
        ax_mat2.set_facecolor('white')
        cimg = np.zeros((*mat.shape, 3), dtype=np.uint8)
        for r in range(self.analyzer.n_rows):
            for c in range(self.analyzer.n_cols):
                cimg[r, c] = COLORS_RGB.get(min(int(mat[r, c]), 4), (200, 20, 20))
        ax_mat2.imshow(cimg, interpolation='nearest', aspect='auto')
        fs2 = max(5, min(10, 60 // max(self.analyzer.n_rows, self.analyzer.n_cols, 1)))
        if not self.binary:
            for r in range(self.analyzer.n_rows):
                for c in range(self.analyzer.n_cols):
                    ax_mat2.text(c, r, str(int(mat[r, c])), ha='center', va='center',
                                 color='white', fontsize=fs2, fontweight='bold')
        ax_mat2.legend(handles=self._legend_patches(), loc='upper right', fontsize=7)
        ax_mat2.set_title(f"Matrice — frame {fi}", color='black')
        ax_mat2.tick_params(colors='black')

        pct = 100*n_filled/n_traps if n_traps else 0
        fig2.suptitle(f"Frame {fi}  —  {n_filled}/{n_traps} pièges remplis ({pct:.1f}%)",
                      color='black', fontsize=12)
        fig2.tight_layout()
        p2 = f"{base}_{tag}_distrib_matrix.svg"
        fig2.savefig(p2, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig2)
        print(f"    • {p2}")

    # ── Export Excel ──────────────────────────────────────────────────────
    def _export_excel(self, base, saved_frames):
        try:
            import openpyxl
            from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
            from openpyxl.utils import get_column_letter
        except ImportError:
            print("[WARN] openpyxl manquant — pip install openpyxl")
            self._export_csv_fallback(base, saved_frames)
            return

        wb = openpyxl.Workbook()

        # ── Feuille 0 : Paramètres (traçabilité) ─────────────────────────────
        ws_p = wb.active
        ws_p.title = "Parametres"
        a = self.analyzer
        params = [
            ("Vidéo",                 os.path.basename(a.video_path or "")),
            ("Version",               VERSION),
            ("Mode de comptage",      "binaire" if self.binary else "comptage"),
            ("Soustraction du fond",  "OUI" if a.bg_subtract else "non"),
            ("Frame de référence",    a.bg_ref_frame if a.bg_subtract else ""),
            ("Moyenne référence (N)", a.bg_ref_navg  if a.bg_subtract else ""),
            ("Invert après soustr.",  ("OUI" if a.bg_invert else "non") if a.bg_subtract else ""),
            ("LUT min / max",         f"{a.detect_lut_min} / {a.detect_lut_max}"),
            ("Seuil lo / hi",         f"{a.detect_thr_lo} / {a.detect_thr_hi}"),
            ("Aire min / max (px²)",  f"{a.detect_min_area} / {a.detect_max_area}"),
            ("Circularité min",       a.detect_circularity if not self.binary else "n/a"),
            ("Détection d'amas",      ("OUI" if a.detect_watershed else "non") if not self.binary else "n/a"),
            ("Rayon particule (px)",  a.detect_particle_r if not self.binary else "n/a"),
            ("Rotation (°)",          round(float(a.rotation_angle), 3)),
            ("Crop",                  str(a.crop_rect)),
            ("Grille",                f"{a.n_rows} × {a.n_cols}"),
            ("Frames analysées",      f"{a.frame_start} → {a.frame_end} (pas {a.analysis_step})"),
        ]
        for ri, (k, v) in enumerate(params, 1):
            ws_p.cell(ri, 1, k).font = Font(bold=True)
            ws_p.cell(ri, 2, v)
        ws_p.column_dimensions['A'].width = 26
        ws_p.column_dimensions['B'].width = 34

        # ── Feuille 1 : Timeline ─────────────────────────────────────────────
        ws_tl = wb.create_sheet("Timeline")
        col3 = "Pièges occupés" if self.binary else "Total particules"
        headers_tl = ["Frame index", "Temps (s)", col3, "Pièges occupés",
                      "Taux remplissage (%)", "Hors-focus"]
        for ci, h in enumerate(headers_tl, 1):
            cell = ws_tl.cell(1, ci, h)
            cell.font      = Font(bold=True, color="FFFFFF")
            cell.fill      = PatternFill("solid", fgColor="2E75B6")
            cell.alignment = Alignment(horizontal="center")

        n_traps   = self.analyzer.n_rows * self.analyzer.n_cols
        total_arr = self.analyzer.counts_matrix.sum(axis=(1, 2))
        occ_arr   = (self.analyzer.counts_matrix > 0).sum(axis=(1, 2))
        times_arr = np.array(self.analyzer.analyzed_frames) / self.analyzer.fps
        for ri, (fi, t, tot, occ) in enumerate(
                zip(self.analyzer.analyzed_frames, times_arr, total_arr, occ_arr), 2):
            oof = bool(self.analyzer.oof_flags[ri-2]) if ri-2 < len(self.analyzer.oof_flags) else False
            ws_tl.cell(ri, 1, int(fi))
            ws_tl.cell(ri, 2, round(float(t), 4))
            ws_tl.cell(ri, 3, int(tot))
            ws_tl.cell(ri, 4, int(occ))
            ws_tl.cell(ri, 5, round(100.0 * occ / n_traps, 2) if n_traps else 0)
            ws_tl.cell(ri, 6, "OUI" if oof else "non")
            if oof:
                for ci in range(1, 7):
                    ws_tl.cell(ri, ci).fill = PatternFill("solid", fgColor="FFD7D7")
        for ci in range(1, 7):
            ws_tl.column_dimensions[get_column_letter(ci)].width = 18

        def rgb_to_hex(rgb_tuple):
            return "{:02X}{:02X}{:02X}".format(*rgb_tuple)

        COLOR_HEX = {k: rgb_to_hex(v) for k, v in COLORS_RGB.items()}
        thin   = Side(style='thin', color='AAAAAA')
        border = Border(left=thin, right=thin, top=thin, bottom=thin)

        for fi_actual, ai in saved_frames:
            mat    = self.analyzer.counts_matrix[ai]
            nr, nc = mat.shape
            ws = wb.create_sheet(f"Frame_{fi_actual}"[:31])

            t_actual = fi_actual / self.analyzer.fps
            total_f  = int(mat.sum())
            n_filled = int(np.sum(mat > 0))
            pct      = 100 * n_filled / n_traps if n_traps else 0

            info_rows = [
                ("Frame", fi_actual),
                ("Temps (s)", round(t_actual, 3)),
                ("Mode", "binaire" if self.binary else "comptage"),
                ("Total particules" if not self.binary else "Pièges occupés", total_f),
                ("Pièges remplis", n_filled),
                ("Total pièges",   n_traps),
                ("Taux remplissage (%)", round(pct, 1)),
            ]
            for ri, (k, v) in enumerate(info_rows, 1):
                ws.cell(ri, 1, k).font = Font(bold=True)
                ws.cell(ri, 2, v)

            start_row = len(info_rows) + 2
            start_col = 1

            ws.cell(start_row, start_col, "Ligne \\ Colonne").font = Font(bold=True)
            for c in range(nc):
                cell = ws.cell(start_row, start_col + 1 + c, f"Col {c}")
                cell.font      = Font(bold=True)
                cell.alignment = Alignment(horizontal="center")

            for r in range(nr):
                ws.cell(start_row + 1 + r, start_col, f"Ligne {r}").font = Font(bold=True)
                for c in range(nc):
                    cnt  = int(mat[r, c])
                    cell = ws.cell(start_row + 1 + r, start_col + 1 + c, cnt)
                    cell.fill      = PatternFill("solid",
                                                 fgColor=COLOR_HEX.get(min(cnt, 4), "C81414"))
                    cell.font      = Font(color="FFFFFF", bold=True)
                    cell.alignment = Alignment(horizontal="center")
                    cell.border    = border
                    ws.column_dimensions[get_column_letter(start_col+1+c)].width = 6

            ws.column_dimensions[get_column_letter(start_col)].width = 12

            tbl_col = start_col + nc + 3
            for ci2, h in enumerate(["Ligne", "Colonne",
                                     "Occupé (0/1)" if self.binary else "Nb particules"],
                                    tbl_col):
                cell = ws.cell(start_row, ci2, h)
                cell.fill      = PatternFill("solid", fgColor="2E75B6")
                cell.font      = Font(bold=True, color="FFFFFF")
                cell.alignment = Alignment(horizontal="center")
            ri2 = start_row + 1
            for r in range(nr):
                for c in range(nc):
                    cnt = int(mat[r, c])
                    ws.cell(ri2, tbl_col,     r)
                    ws.cell(ri2, tbl_col + 1, c)
                    ws.cell(ri2, tbl_col + 2, cnt)
                    if cnt > 0:
                        ws.cell(ri2, tbl_col + 2).fill = PatternFill(
                            "solid", fgColor=COLOR_HEX.get(min(cnt, 4), "C81414"))
                        ws.cell(ri2, tbl_col + 2).font = Font(color="FFFFFF", bold=True)
                    ri2 += 1
            for ci2 in range(tbl_col, tbl_col + 3):
                ws.column_dimensions[get_column_letter(ci2)].width = 14

        xlsx_path = base + "_analyse.xlsx"
        wb.save(xlsx_path)
        print(f"  → Excel : {xlsx_path}")

    def _export_csv_fallback(self, base, saved_frames):
        import csv
        tl_path   = base + "_timeline.csv"
        total_arr = self.analyzer.counts_matrix.sum(axis=(1, 2))
        occ_arr   = (self.analyzer.counts_matrix > 0).sum(axis=(1, 2))
        times_arr = np.array(self.analyzer.analyzed_frames) / self.analyzer.fps
        with open(tl_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["Frame", "Temps(s)", "Total", "Pieges_occupes", "OOF"])
            for fi, t, tot, occ, oof in zip(self.analyzer.analyzed_frames, times_arr,
                                            total_arr, occ_arr, self.analyzer.oof_flags):
                w.writerow([fi, round(float(t), 4), int(tot), int(occ), int(oof)])
        print(f"  → CSV timeline : {tl_path}")

        for fi_actual, ai in saved_frames:
            mat = self.analyzer.counts_matrix[ai]
            p = f"{base}_frame{fi_actual:05d}_matrix.csv"
            with open(p, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow([f"Frame {fi_actual}"] + [f"Col{c}" for c in range(mat.shape[1])])
                for r, row in enumerate(mat):
                    w.writerow([f"Ligne{r}"] + [int(v) for v in row])
            print(f"  → CSV frame : {p}")

    def _fig_timeline(self, base):
        total = self.analyzer.counts_matrix.sum(axis=(1, 2))
        times = np.array(self.analyzer.analyzed_frames) / self.analyzer.fps
        dt    = (times[1]-times[0])/2 if len(times) > 1 else 0.5

        fig, ax = white_fig(figsize=(13, 4))
        ax.plot(times, total, color='steelblue', lw=2)
        for i, oof in enumerate(self.analyzer.oof_flags):
            if oof and i < len(times):
                ax.axvspan(times[i]-dt, times[i]+dt, color='salmon', alpha=0.3)
        peak_i = int(np.argmax(total))
        ax.annotate(f"  max = {int(total[peak_i])}",
                    xy=(times[peak_i], total[peak_i]),
                    xytext=(times[peak_i], total[peak_i]*0.80),
                    arrowprops=dict(arrowstyle='->', color='black'),
                    color='black', fontsize=9)
        ax.set_xlabel("Temps (s)", color='black')
        ax.set_ylabel("Pièges occupés" if self.binary else "Particules totales détectées",
                      color='black')
        ax.set_title("Évolution temporelle — remplissage des pièges",
                     color='black', fontsize=12)
        fig.tight_layout()
        path = base + "_capture_timeline.svg"
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  → {path}")


# =============================================================================
# APPLICATION PRINCIPALE
# =============================================================================

class App:

    def __init__(self):
        self.analyzer = MicrofluidicAnalyzer()

        self.root = tk.Tk()
        self.root.title(f"{APP_TITLE}  v{VERSION}")
        self.root.geometry("540x380")
        self.root.configure(bg="white")
        self.root.resizable(False, False)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TProgressbar", troughcolor='#dddddd', background='steelblue',
                        bordercolor='white', lightcolor='steelblue', darkcolor='steelblue')

        self._build_home()
        self.root.bind("<<SetupDone>>", self._on_setup_done)
        self.root.mainloop()

    def _build_home(self):
        for w in self.root.winfo_children():
            w.destroy()

        tk.Label(self.root, text="🔬   Microfluidic Trap Analyzer",
                 font=("Helvetica", 18, "bold"),
                 bg="white", fg="#1a5fa8").pack(pady=22)

        tk.Label(self.root,
                 text="Soustraction du fond  →  LUT  →  Threshold  →  Blobs\n"
                      "Modes binaire (vide/plein) et comptage  |  Vidéos AVI",
                 font=("Helvetica", 9),
                 bg="white", fg="#555555", justify="center").pack()

        btns = tk.Frame(self.root, bg="white")
        btns.pack(pady=18)

        tk.Button(btns, text="📂   Ouvrir une vidéo  (.avi / .mp4 / …)",
                  command=self._select_video, font=("Helvetica", 11), width=32,
                  bg="#4a90d9", fg="white", relief="flat",
                  padx=10, pady=8).pack(pady=6)

        tk.Button(btns, text="⚙   Importer une config existante (.json)",
                  command=self._import_config, font=("Helvetica", 10), width=32,
                  bg="#eeeeee", fg="#333333", relief="flat",
                  padx=10, pady=6).pack(pady=4)

        pf = tk.Frame(self.root, bg="white")
        pf.pack(pady=6)
        tk.Label(pf, text="Expansion ROI (px) :", bg="white", fg="#555555",
                 font=("Helvetica", 9)).pack(side="left", padx=6)
        self.v_expand = tk.IntVar(value=ROI_EXPAND)
        tk.Spinbox(pf, from_=0, to=30, textvariable=self.v_expand,
                   width=4, font=("Helvetica", 9)).pack(side="left")

    def _select_video(self):
        path = filedialog.askopenfilename(
            title="Sélectionner la vidéo",
            filetypes=[("AVI", "*.avi"), ("MP4", "*.mp4 *.mov *.mkv"),
                       ("TIFF", "*.tif *.tiff"), ("Tous", "*.*")])
        if not path:
            return
        try:
            self.analyzer.open_video(path)
            self.analyzer.roi_expand = self.v_expand.get()
        except Exception as e:
            messagebox.showerror("Erreur vidéo", str(e));  return
        SetupWizard(self.root, self.analyzer)

    def _import_config(self):
        path = filedialog.askopenfilename(title="Importer config JSON",
                                          filetypes=[("JSON", "*.json"), ("Tous", "*.*")])
        if not path:
            return
        try:
            self.analyzer.load_config(path)
        except Exception as e:
            messagebox.showerror("Erreur config", str(e));  return

        vpath = filedialog.askopenfilename(
            title="Sélectionner la vidéo",
            filetypes=[("AVI", "*.avi"), ("MP4", "*.mp4 *.mov *.mkv"),
                       ("TIFF", "*.tif *.tiff"), ("Tous", "*.*")])
        if not vpath:
            return
        try:
            self.analyzer.open_video(vpath)
            self.analyzer.roi_expand = self.v_expand.get()
            self.analyzer.compute_rois()
        except Exception as e:
            messagebox.showerror("Erreur", str(e));  return
        self._on_setup_done()

    def _on_setup_done(self, event=None):
        if not self.analyzer.rois:
            messagebox.showwarning("", "Aucune ROI définie.");  return

        params = AnalysisParamsWindow(self.root, self.analyzer)
        if not params.ok:
            return

        n_frames = len(list(range(self.analyzer.frame_start,
                                  self.analyzer.frame_end + 1,
                                  max(1, self.analyzer.analysis_step))))

        prog = ProgressBar(self.root, n_frames)
        try:
            self.analyzer.analyze_all_frames(progress_cb=prog.update)
        except Exception as e:
            prog.close()
            messagebox.showerror("Erreur analyse", str(e));  return
        prog.close()

        # Sauvegarde de la config enrichie (soustraction + mode inclus)
        try:
            base = os.path.splitext(self.analyzer.video_path)[0]
            self.analyzer.save_config(base + "_config.json")
        except Exception:
            pass

        VisualizationWindow(self.root, self.analyzer)


# =============================================================================
if __name__ == "__main__":
    print("=" * 64)
    print(f"  {APP_TITLE}  v{VERSION}")
    print("  Vérification des dépendances…")
    print("=" * 64)

    deps = {
        'cv2':        'opencv-python',
        'numpy':      'numpy',
        'matplotlib': 'matplotlib',
        'scipy':      'scipy',
        'PIL':        'Pillow',
    }
    missing = []
    for mod, pkg in deps.items():
        try:
            __import__(mod)
            print(f"  [OK]       {mod}")
        except ImportError:
            print(f"  [MANQUANT] {mod}  →  pip install {pkg}")
            missing.append(pkg)

    try:
        __import__('openpyxl')
        print("  [OK]       openpyxl")
    except ImportError:
        print("  [OPTION]   openpyxl manquant → export CSV utilisé à la place")

    if missing:
        print(f"\n  Commande : pip install {' '.join(missing)}")
        sys.exit(1)

    print("\n  Toutes les dépendances sont présentes. Démarrage…\n")
    App()