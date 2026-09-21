"""
=============================================================================
PARTICLE TRACKER - Ring Electrode Experiment  (mode semi-manuel)
=============================================================================
Usage:
    python particle_tracker.py               # ouvre un selecteur de fichier
    python particle_tracker.py video.avi     # charge directement la video

Dependances:
    pip install opencv-python numpy matplotlib scipy pandas openpyxl

── Deroulement ──────────────────────────────────────────────────────────────

  ETAPE 0  Saisie du frame rate (console)

  ETAPE 0b Choix du mode :
    · MODE CLASSIQUE      → saisie du nombre de particules, tracking en lot
    · MODE CARACTERISATION→ une particule a la fois, save ou non, continuer/stop

  ETAPE 1  Selection du centre de l'anneau
    · Clic gauche        → pose le centre
    · P                  → augmente le rayon du cercle guide (+5 px)
    · M                  → diminue le rayon du cercle guide (-5 px)
    · Entree / Espace    → valide et passe a l'etape suivante
    · Q                  → quitter

  ETAPE 2  (mode classique) Saisie du nombre de particules a suivre (console)

  ETAPE 3  Selection des particules sur la frame 0
    · Clic gauche        → marque une particule (cercle vert numerote)
    · Retour arriere     → annule le dernier clic
    · Entree / Espace    → lance le tracking (quand le bon nombre est atteint)
    · Q                  → quitter

  ETAPE 4  Tracking automatique frame par frame (template matching NCC)
    · Q                  → interrompt le suivi

── Sorties ──────────────────────────────────────────────────────────────────
    · tracking_results.xlsx   (mode classique)  onglet unique
    · tracking_results.xlsx   (mode caract.)    un onglet par cellule sauvegardee
    · trajectories_rt.png     courbes r(t) et theta(t)
    · trajectories_polar.png  carte polaire des trajectoires
=============================================================================
"""

import sys
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

# =============================================================================
# PARAMETRES — modifiez ici selon votre experience
# =============================================================================

SCALE_PX_PER_UM   = 3.0    # echelle : pixels par micrometre
GUIDE_RADIUS_UM   = 140    # rayon du cercle guide affiche en µm
RADIUS_STEP       = 5      # pas de reglage P/M du cercle guide (px)

PATCH_SIZE        = 55     # taille du patch template NCC (px, cote)
WINDOW_HALF       = 80     # demi-fenetre de recherche NCC entre deux frames (px)
NCC_THRESHOLD     = 0.35   # score NCC minimum pour valider le suivi (0-1)

# =============================================================================

GUIDE_RADIUS_INIT = int(GUIDE_RADIUS_UM * SCALE_PX_PER_UM)


# ── Utilitaires affichage ────────────────────────────────────────────────────

def get_display_size():
    """Retourne (max_w, max_h) disponibles pour les fenetres OpenCV."""
    try:
        root = tk.Tk()
        root.withdraw()
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        root.destroy()
        # Laisse une marge pour la barre des taches / titre de fenetre
        return int(sw * 0.90), int(sh * 0.85)
    except Exception:
        return 1280, 720


def fit_frame(frame, max_w, max_h):
    """
    Retourne (resized_frame, scale) ou scale = ratio px_affiche / px_original.
    Ne deforme pas l'image (letterbox implicite via ratio uniforme).
    """
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)   # ne jamais agrandir
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized = frame.copy()
        scale   = 1.0
    return resized, scale


def scale_point(pt, scale):
    """Convertit un point ecran (apres fit_frame) en coordonnees image originale."""
    return (int(pt[0] / scale), int(pt[1] / scale))


def unscale_point(pt, scale):
    """Coordonnees image originale → coordonnees ecran."""
    return (int(pt[0] * scale), int(pt[1] * scale))


# ── Utilitaires metier ───────────────────────────────────────────────────────

def px_to_um(px):
    return px / SCALE_PX_PER_UM


def cart_to_polar(x, y, cx, cy):
    """Retourne (r_px, theta_deg) dans le repere trigo (Y image inverse)."""
    dx, dy = x - cx, y - cy
    r      = np.sqrt(dx**2 + dy**2)
    theta  = np.degrees(np.arctan2(-dy, dx))
    return r, theta


# ── Estimation du rayon de particule ─────────────────────────────────────────

def estimate_radius_um(gray_patch):
    if gray_patch is None or gray_patch.size == 0:
        return float('nan')
    blurred = cv2.GaussianBlur(gray_patch, (5, 5), 0)
    _, bw   = cv2.threshold(blurred, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bw      = cv2.morphologyEx(bw, cv2.MORPH_OPEN,  kernel, iterations=1)
    bw      = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return float('nan')
    cy_patch  = gray_patch.shape[0] / 2.0
    cx_patch  = gray_patch.shape[1] / 2.0
    MIN_AREA  = 10
    best_area = None
    best_dist = float('inf')
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        mx   = M['m10'] / M['m00']
        my   = M['m01'] / M['m00']
        dist = (mx - cx_patch)**2 + (my - cy_patch)**2
        if dist < best_dist:
            best_dist = dist
            best_area = area
    if best_area is None:
        return float('nan')
    return round(px_to_um((best_area / 3.14159265) ** 0.5), 3)


# ── Etape 1 : selection interactive du centre ─────────────────────────────────

class CenterSelector:
    WIN = "Etape 1  |  Clic = centre   P/M = rayon   Entree = valider   Q = quitter"

    def __init__(self, frame):
        self.frame_orig = frame.copy()
        self.center     = None          # en coord. originales
        self.radius     = GUIDE_RADIUS_INIT
        self.max_w, self.max_h = get_display_size()

    def _redraw(self):
        img, scale = fit_frame(self.frame_orig, self.max_w, self.max_h)
        if self.center:
            c_screen = unscale_point(self.center, scale)
            r_screen = max(1, int(self.radius * scale))
            cv2.circle(img, c_screen, r_screen, (0, 0, 255), 2)
            cv2.drawMarker(img, c_screen, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(img,
                        "r = {:.1f} um  |  centre {}".format(
                            px_to_um(self.radius), self.center),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        cv2.putText(img,
                    "P = rayon+   M = rayon-   Entree = valider   Q = quitter",
                    (10, img.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow(self.WIN, img)
        self._scale = scale   # memorise pour la callback souris

    def _mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # convertit coordonnees ecran → image originale
            self.center = scale_point((x, y), self._scale)
            self._redraw()

    def run(self):
        self._scale = 1.0
        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WIN, self.max_w, self.max_h)
        cv2.setMouseCallback(self.WIN, self._mouse)
        self._redraw()
        while True:
            key = cv2.waitKey(50) & 0xFF
            if   key in (ord('p'), ord('P')):
                self.radius += RADIUS_STEP; self._redraw()
            elif key in (ord('m'), ord('M')):
                self.radius = max(10, self.radius - RADIUS_STEP); self._redraw()
            elif key in (13, 32):
                if self.center: break
                print("[!] Cliquez d'abord sur le centre de l'anneau.")
            elif key in (ord('q'), ord('Q')):
                self.center = None; break
        cv2.destroyAllWindows()
        return self.center, self.radius


# ── Etape 3 : selection des particules sur la frame 0 ────────────────────────

class ParticleSelector:

    def __init__(self, frame, n_target):
        self.frame_orig = frame.copy()
        self.n_target   = n_target
        self.points     = []            # coord. originales
        self.win        = ("Etape 3  |  Cliquez sur {} particule(s)  "
                           "|  Retour = annuler  |  Entree = lancer  "
                           "|  Q = quitter").format(n_target)
        self.max_w, self.max_h = get_display_size()
        self._scale     = 1.0

    def _redraw(self):
        img, scale = fit_frame(self.frame_orig, self.max_w, self.max_h)
        self._scale = scale
        for i, (px, py) in enumerate(self.points):
            ps = unscale_point((px, py), scale)
            cv2.circle(img, ps, 12, (0, 255, 0), 2)
            cv2.putText(img, "P{}".format(i + 1), (ps[0] + 14, ps[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        remaining = self.n_target - len(self.points)
        msg = ("{}/{} — encore {} clic(s)".format(
                   len(self.points), self.n_target, remaining)
               if remaining > 0 else
               "{}/{} — Entree pour lancer".format(self.n_target, self.n_target))
        cv2.putText(img, msg, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(self.win, img)

    def _mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < self.n_target:
                self.points.append(scale_point((x, y), self._scale))
                self._redraw()
                if len(self.points) == self.n_target:
                    print("[✓] {} particule(s) selectionnee(s). "
                          "Appuyez sur Entree pour lancer.".format(self.n_target))

    def run(self):
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, self.max_w, self.max_h)
        cv2.setMouseCallback(self.win, self._mouse)
        self._redraw()
        while True:
            key = cv2.waitKey(50) & 0xFF
            if key in (13, 32):
                if len(self.points) == self.n_target: break
                print("[!] Selectionnez exactement {} particule(s) "
                      "(actuellement {}).".format(self.n_target, len(self.points)))
            elif key == 8:
                if self.points:
                    self.points.pop(); self._redraw()
            elif key in (ord('q'), ord('Q')):
                self.points = []; break
        cv2.destroyAllWindows()
        return self.points


# ── Tracker NCC ──────────────────────────────────────────────────────────────

class PatchTracker:
    def __init__(self, particle_id, gray0, init_cx, init_cy):
        self.pid  = particle_id
        self.cx   = init_cx
        self.cy   = init_cy
        self.lost = False
        h, w      = gray0.shape
        half      = PATCH_SIZE // 2
        x1, x2   = max(0, init_cx - half), min(w, init_cx + half)
        y1, y2   = max(0, init_cy - half), min(h, init_cy + half)
        self.template  = gray0[y1:y2, x1:x2].copy()
        self.radius_um = estimate_radius_um(self.template)

    def update(self, gray):
        if self.lost or self.template is None:
            return self.cx, self.cy, False
        h, w = gray.shape
        x1 = max(0, self.cx - WINDOW_HALF);  x2 = min(w, self.cx + WINDOW_HALF)
        y1 = max(0, self.cy - WINDOW_HALF);  y2 = min(h, self.cy + WINDOW_HALF)
        roi  = gray[y1:y2, x1:x2]
        th, tw = self.template.shape
        if roi.shape[0] < th or roi.shape[1] < tw:
            return self.cx, self.cy, False
        result             = cv2.matchTemplate(roi, self.template,
                                               cv2.TM_CCOEFF_NORMED)
        _, max_val, _, loc = cv2.minMaxLoc(result)
        if max_val < NCC_THRESHOLD:
            self.lost = True
            return self.cx, self.cy, False
        new_cx = x1 + loc[0] + tw // 2
        new_cy = y1 + loc[1] + th // 2
        self.cx, self.cy = new_cx, new_cy
        xp1 = max(0, new_cx - PATCH_SIZE // 2); xp2 = min(w, new_cx + PATCH_SIZE // 2)
        yp1 = max(0, new_cy - PATCH_SIZE // 2); yp2 = min(h, new_cy + PATCH_SIZE // 2)
        patch = gray[yp1:yp2, xp1:xp2]
        if patch.shape == self.template.shape:
            self.template = cv2.addWeighted(self.template, 0.7, patch, 0.3, 0)
        r_est = estimate_radius_um(gray[yp1:yp2, xp1:xp2])
        if r_est == r_est:
            self.radius_um = r_est
        return new_cx, new_cy, True


# ── Boucle de tracking ───────────────────────────────────────────────────────

def run_tracking(cap, trackers, center, frame_count, fps, label="Tracking"):
    """
    Retourne un DataFrame avec les colonnes standard.
    label : texte affiche dans la fenetre (utile en mode caract.)
    """
    cx0, cy0 = center
    records   = []
    win_name  = "{}  |  Q = arreter".format(label)
    max_w, max_h = get_display_size()

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display = frame.copy()
        t_s     = frame_idx / fps

        for tr in trackers:
            new_cx, new_cy, ok = tr.update(gray)
            if not ok and frame_idx > 0:
                continue
            r_px, theta = cart_to_polar(new_cx, new_cy, cx0, cy0)
            r_um        = px_to_um(r_px)
            r_part      = tr.radius_um
            records.append({
                "frame":       frame_idx,
                "time_s":      round(t_s, 4),
                "particle_id": tr.pid + 1,
                "x_px":        new_cx,
                "y_px":        new_cy,
                "r_um":        round(r_um, 3),
                "theta_deg":   round(theta, 3),
                "R_part_um":   r_part,
                "tracked":     ok
            })
            color  = (0, 255, 0) if ok else (0, 0, 200)
            cv2.circle(display, (new_cx, new_cy), 10, color, 2)
            if r_part == r_part:
                cv2.circle(display, (new_cx, new_cy),
                           max(3, int(r_part * SCALE_PX_PER_UM)), color, 1)
            cv2.putText(display,
                        "P{}  r={:.0f}um  Rp={:.1f}um".format(
                            tr.pid + 1, r_um,
                            r_part if r_part == r_part else 0),
                        (new_cx + 13, new_cy - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

        cv2.drawMarker(display, (cx0, cy0), (0, 0, 255),
                       cv2.MARKER_CROSS, 20, 2)
        cv2.putText(display,
                    "Frame {}/{}   t = {:.3f} s".format(frame_idx, frame_count, t_s),
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

        disp_scaled, _ = fit_frame(display, max_w, max_h)
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, disp_scaled.shape[1], disp_scaled.shape[0])
        cv2.imshow(win_name, disp_scaled)
        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            print("[!] Tracking interrompu.")
            break

    cv2.destroyAllWindows()

    if not records:
        return None
    return pd.DataFrame(records)


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_results(df, output_dir, save=True, title_suffix=""):
    ids    = sorted(df["particle_id"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(ids), 1)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for pid, col in zip(ids, colors):
        sub = df[df["particle_id"] == pid].sort_values("time_s")
        ax1.plot(sub["time_s"], sub["r_um"],
                 label="P{}".format(pid), color=col, lw=1.5)
        ax2.plot(sub["time_s"], sub["theta_deg"],
                 label="P{}".format(pid), color=col, lw=1.5)
    ax1.set_ylabel("r (µm)"); ax1.set_title("Distance radiale r(t)" + title_suffix)
    ax1.legend(fontsize=8, ncol=4); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("Temps (s)"); ax2.set_ylabel("theta (deg)")
    ax2.set_title("Angle theta(t)" + title_suffix)
    ax2.legend(fontsize=8, ncol=4); ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    p1 = None
    if save:
        suffix = title_suffix.replace(" ", "_").replace("/", "-")
        p1 = os.path.join(output_dir, "trajectories_rt{}.png".format(suffix))
        plt.savefig(p1, dpi=150)
        print("[✓] Graphe sauvegarde : {}".format(p1))
    plt.show()

    fig2, ax3 = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    for pid, col in zip(ids, colors):
        sub       = df[df["particle_id"] == pid].sort_values("time_s")
        theta_rad = np.radians(sub["theta_deg"].values)
        r_vals    = sub["r_um"].values
        ax3.plot(theta_rad, r_vals, label="P{}".format(pid), color=col,
                 lw=1.0, alpha=0.8)
        ax3.scatter(theta_rad[0], r_vals[0], color=col, s=60, zorder=5)
    ax3.set_title("Trajectoires polaires (µm)" + title_suffix, pad=20)
    ax3.legend(fontsize=7, loc="upper right", ncol=2)

    p2 = None
    if save:
        p2 = os.path.join(output_dir, "trajectories_polar{}.png".format(suffix))
        plt.savefig(p2, dpi=150)
        print("[✓] Graphe sauvegarde : {}".format(p2))
    plt.show()

    return p1, p2


# ── Sauvegarde Excel ──────────────────────────────────────────────────────────

def save_to_excel(df, output_dir, sheet_name="Sheet1"):
    """Ajoute un onglet dans tracking_results.xlsx (cree le fichier si absent)."""
    xlsx_path = os.path.join(output_dir, "tracking_results.xlsx")
    mode      = "a" if os.path.exists(xlsx_path) else "w"
    kw        = {"if_sheet_exists": "replace"} if mode == "a" else {}
    with pd.ExcelWriter(xlsx_path, engine="openpyxl", mode=mode, **kw) as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    print("[✓] Excel sauvegarde : {} (onglet '{}')".format(xlsx_path, sheet_name))


# ── Boite de dialogue Tkinter ─────────────────────────────────────────────────

def ask_yes_no(title, message):
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    ans  = messagebox.askyesno(title, message, parent=root)
    root.destroy()
    return ans


def ask_continue_stop(title, message):
    """Retourne True = continuer, False = stop."""
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    ans  = messagebox.askyesno(title, message,
                               detail="OUI = Continuer    NON = Stop",
                               parent=root)
    root.destroy()
    return ans


def ask_fps_dialog(detected_fps):
    """Fenetre Tkinter pour saisir le FPS, avec le FPS detecte en valeur par defaut."""
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    fps_str = simpledialog.askstring(
        "Frame rate",
        "Entrez le frame rate (fps) :\n(FPS detecte dans la video : {:.2f})".format(
            detected_fps),
        initialvalue="{:.2f}".format(detected_fps),
        parent=root)
    root.destroy()
    if fps_str is None:
        return None
    try:
        val = float(fps_str.replace(",", "."))
        if val <= 0:
            raise ValueError
        return val
    except ValueError:
        print("[!] FPS invalide. Utilisation du FPS detecte : {:.2f}".format(detected_fps))
        return detected_fps


def ask_mode_dialog():
    """Retourne 'classique' ou 'caracterisation'."""
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)

    result = {"mode": None}

    win = tk.Toplevel(root)
    win.title("Choix du mode")
    win.attributes("-topmost", True)
    win.resizable(False, False)

    tk.Label(win, text="Choisissez le mode de tracking :",
             font=("Helvetica", 13, "bold"), pady=10).pack()

    def choose(m):
        result["mode"] = m
        win.destroy()

    tk.Button(win, text="Mode Classique\n(plusieurs cellules en lot)",
              width=32, height=3, bg="#2196F3", fg="white",
              font=("Helvetica", 11),
              command=lambda: choose("classique")).pack(padx=20, pady=8)

    tk.Button(win, text="Mode Caracterisation\n(une cellule a la fois, save optionnel)",
              width=32, height=3, bg="#4CAF50", fg="white",
              font=("Helvetica", 11),
              command=lambda: choose("caracterisation")).pack(padx=20, pady=8)

    win.protocol("WM_DELETE_WINDOW", lambda: choose(None))
    root.wait_window(win)
    root.destroy()
    return result["mode"]


# ── MODE CLASSIQUE ────────────────────────────────────────────────────────────

def run_mode_classique(cap, first_frame, center, frame_count, fps, output_dir):
    print("\n── Etape 2 : nombre de particules a suivre ──")
    while True:
        try:
            n = int(input("Combien de particules voulez-vous tracker ? "))
            if n >= 1: break
        except ValueError:
            pass
        print("[!] Entrez un entier positif.")

    print("\n── Etape 3 : cliquez sur {} particule(s) ──".format(n))
    points = ParticleSelector(first_frame, n).run()

    if len(points) != n:
        print("[!] Selection incomplete. Abandon.")
        return

    gray0    = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    trackers = [PatchTracker(i, gray0, px, py) for i, (px, py) in enumerate(points)]

    print("\n── Etape 4 : tracking ({} frames) ──".format(frame_count))
    df = run_tracking(cap, trackers, center, frame_count, fps)

    if df is not None and not df.empty:
        save_to_excel(df, output_dir, sheet_name="Classique")
        print("[✓] {} mesures  |  {} particule(s).".format(
            len(df), df["particle_id"].nunique()))
        plot_results(df, output_dir, save=True)
    else:
        print("[!] Aucune donnee a tracer.")


# ── MODE CARACTERISATION ──────────────────────────────────────────────────────

def run_mode_caracterisation(cap, first_frame, center, frame_count, fps, output_dir):
    all_dfs      = []    # toutes les analyses sauvegardees (pour graphe final)
    cell_counter = 0     # numero de cellule au total (pour le nom d'onglet)

    while True:
        cell_counter += 1
        print("\n── Caracterisation cellule {} ──".format(cell_counter))
        print("   Cliquez sur 1 particule (Entree = lancer | Q = quitter)")

        points = ParticleSelector(first_frame, 1).run()
        if not points:
            print("[!] Aucune selection. Fin du mode caracterisation.")
            break

        px_pt, py_pt = points[0]
        gray0        = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        tracker      = PatchTracker(0, gray0, px_pt, py_pt)

        print("   Lancement du tracking…")
        df = run_tracking(cap, [tracker], center, frame_count, fps,
                          label="Caracterisation C{}".format(cell_counter))

        if df is None or df.empty:
            print("[!] Aucune donnee pour cette cellule.")
        else:
            # Renumérotation particle_id pour le graphe (affiche C{n})
            df["particle_id"] = cell_counter

            # Affichage des graphes (pas de sauvegarde encore)
            print("   Affichage des graphes…")
            plot_results(df, output_dir, save=False,
                         title_suffix=" — Cellule {}".format(cell_counter))

            # Demande de sauvegarde
            save_it = ask_yes_no(
                "Sauvegarder ?",
                "Voulez-vous sauvegarder l'analyse de la cellule {} ?".format(
                    cell_counter))

            if save_it:
                sheet = "Cellule_{}".format(cell_counter)
                save_to_excel(df, output_dir, sheet_name=sheet)
                # Sauvegarde aussi les graphes
                plot_results(df, output_dir, save=True,
                             title_suffix="_C{}".format(cell_counter))
                all_dfs.append(df.copy())
                print("[✓] Cellule {} sauvegardee.".format(cell_counter))
            else:
                print("[–] Cellule {} non sauvegardee.".format(cell_counter))

        # Continuer ou stop ?
        go_on = ask_continue_stop(
            "Continuer ?",
            "Voulez-vous analyser une autre cellule ?")

        if not go_on:
            print("[✓] Fin du mode caracterisation.")
            break

        # Rewind video pour la prochaine cellule
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Graphe combiné final (toutes les cellules sauvegardees)
    if all_dfs:
        print("\n── Graphe combiné de toutes les cellules sauvegardees ──")
        df_all = pd.concat(all_dfs, ignore_index=True)
        plot_results(df_all, output_dir, save=True,
                     title_suffix="_combined")
    else:
        print("[!] Aucune cellule sauvegardee — pas de graphe combine.")


# ── Point d'entree ────────────────────────────────────────────────────────────

def main():
    # Etape 0a : selection du fichier video
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        try:
            root = tk.Tk(); root.withdraw()
            video_path = filedialog.askopenfilename(
                title="Selectionner une video",
                filetypes=[("Videos", "*.avi *.mp4 *.mov *.tif *.tiff"),
                           ("Tous",   "*.*")])
            root.destroy()
        except Exception:
            video_path = input("Chemin de la video : ").strip()

    if not video_path or not os.path.exists(video_path):
        print("[!] Fichier introuvable. Abandon.")
        return

    output_dir = os.path.dirname(os.path.abspath(video_path))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[!] Impossible d'ouvrir : {}".format(video_path))
        return

    frame_count   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_detected  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w             = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h             = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("[✓] Video : {} frames  |  {:.2f} fps detecte  |  {}x{} px".format(
        frame_count, fps_detected, w, h))

    # Etape 0b : saisie / confirmation du FPS
    fps = ask_fps_dialog(fps_detected)
    if fps is None:
        print("[!] FPS non renseigne. Abandon.")
        cap.release(); return
    print("[✓] FPS utilise : {:.4f}  |  Duree totale estimee : {:.2f} s".format(
        fps, frame_count / fps))

    # Etape 0c : choix du mode
    mode = ask_mode_dialog()
    if mode is None:
        print("[!] Aucun mode selectionne. Abandon.")
        cap.release(); return
    print("[✓] Mode : {}".format(mode))

    # Lecture de la premiere frame
    ret, first_frame = cap.read()
    if not ret:
        print("[!] Impossible de lire la premiere frame.")
        cap.release(); return

    # Etape 1 : centre
    print("\n── Etape 1 : selectionnez le centre de l'anneau ──")
    center, guide_radius = CenterSelector(first_frame).run()
    if center is None:
        print("[!] Centre non selectionne. Abandon.")
        cap.release(); return
    print("[✓] Centre : {}  |  Rayon guide : {} px = {:.1f} µm".format(
        center, guide_radius, px_to_um(guide_radius)))

    # Dispatch selon le mode
    if mode == "classique":
        run_mode_classique(cap, first_frame, center, frame_count, fps, output_dir)
    else:
        run_mode_caracterisation(cap, first_frame, center, frame_count, fps, output_dir)

    cap.release()
    print("\n[✓] Programme termine.")


if __name__ == "__main__":
    main()