"""
=============================================================================
FIGURE w_max vs ΔP — depuis analyse_biaxiale.xlsx
=============================================================================
DEUX figures sont produites :

  FIGURE 1 — Modèle de Schomburg 1D, deux fits par axe :
    FIT 1 : e₀ et W_eff libres
    FIT 2 : e₀ (du Fit 1) fixé, W_geom fixée

  FIGURE 2 — Modèle de plaque elliptique 2D (Kirchhoff) :
    Déformée : w(x,y) = w0 (1 - x²/a² - y²/b²)²
      w0 = P/(8D) · 1/(3/a⁴ + 3/b⁴ + 2/(a²b²)),  D = E t³ / [12(1-ν²)]
    La flèche centrale w0 est UNIQUE (sommet unique) : même droite sur
    les deux axes. FIT : E et e₀ LIBRES, géométrie (a,b,t,ν) FIXÉE.
    Puis tracé de la bande d'incertitude combinée E ± σE et t ± σt.

UTILISATION :
  Modifier 'EXCEL_FILE' ci-dessous puis F5 (ou python plot_wmax_vs_dP.py)
=============================================================================
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import curve_fit
from openpyxl import load_workbook

# =============================================================================
# CONFIGURATION
# =============================================================================
EXCEL_FILE = None      # None → fenêtre de sélection
E0_UM      = 12.0      # µm — gap initial verre-PDMS (courbe de référence)
X_MAX_MBAR = 160       # mbar — limite axe X
Y_MAX_UM   = 160       # µm  — limite axe Y
SAVE_svg   = True

# Bornes littérature du module d'Young PDMS RTV 615 1:10
# E_max : Schneider et al. 2008 (cuisson prolongee haute temperature)
# E_min : Lotters et al. 1997 (E ~ 3G, G ~ 250 kPa)
E_MAX_PA = 1.9e6       # Pa — borne superieure (Schneider 2008)
E_MIN_PA = 0.75e6      # Pa — borne inferieure (Lotters 1997)
# =============================================================================


def ask_file():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
    path = filedialog.askopenfilename(
        title="Sélectionner analyse_biaxiale.xlsx",
        filetypes=[('Excel', '*.xlsx'), ('Tous', '*.*')])
    root.destroy()
    return path


def read_excel(path):
    wb = load_workbook(path, data_only=True)
    params = {}
    for row in wb['Paramètres et fit'].iter_rows(values_only=True):
        if row[0] and row[1] is not None:
            params[str(row[0]).strip()] = row[1]

    ws2 = wb['Résultats exp.']
    data_G, data_P = [], []
    current = None
    for row in ws2.iter_rows(values_only=True):
        if row[0] is None: continue
        c0 = str(row[0]).strip()
        if 'GRAND AXE' in c0: current = 'G'; continue
        if 'PETIT AXE' in c0: current = 'P'; continue
        if c0 == 'Image': continue
        if current and row[1] is not None:
            try:
                e = {'dP_mbar': float(row[1]), 'wm_fit': float(row[4]),
                     'r2': float(row[6]), 'label': c0}
                (data_G if current == 'G' else data_P).append(e)
            except: pass
    return params, data_G, data_P


# =============================================================================
# MODÈLE 1D — SCHOMBURG
# =============================================================================

def schomburg_K(W_um, t_um, E_pa, nu):
    """Pente K [µm/Pa] = W⁴(1-ν²) / (66·t³·E)."""
    return (W_um*1e-6)**4 * (1-nu**2) / (66*(t_um*1e-6)**3 * E_pa) * 1e6


def fit1_e0_Weff_free(dP_mbar, wm, t, E, nu):
    """FIT 1 : e₀ et W_eff libres."""
    valid = np.isfinite(dP_mbar) & np.isfinite(wm)
    if valid.sum() < 3:
        return None, None, None, None
    dP_pa = dP_mbar[valid] * 100
    wm_v  = wm[valid]

    def model(dP, e0, W_um):
        return e0 + schomburg_K(W_um, t, E, nu) * dP

    e0_init = float(wm_v[dP_pa == dP_pa.min()][0]) if (dP_pa == dP_pa.min()).any() else 10.0
    try:
        popt, _ = curve_fit(model, dP_pa, wm_v, p0=[e0_init, 6000.0],
                            bounds=([0.0, 100.0], [50.0, 30000.0]), maxfev=20000)
        e0_f, Weff_f = float(popt[0]), float(popt[1])
    except Exception as ex:
        print(f"  [Fit1] {ex}"); return None, None, None, None

    K_f    = schomburg_K(Weff_f, t, E, nu)
    w_pred = model(dP_pa, e0_f, Weff_f)
    ss_res = np.sum((wm_v - w_pred)**2)
    ss_tot = np.sum((wm_v - wm_v.mean())**2)
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return e0_f, Weff_f, r2, K_f


def fit2_e0_fixed_Wgeom_fixed(dP_mbar, wm, t, E, nu, e0_fixed, W_geom):
    """FIT 2 : e₀ fixé, W_geom fixée."""
    valid = np.isfinite(dP_mbar) & np.isfinite(wm)
    if valid.sum() < 2:
        return None, None
    dP_pa  = dP_mbar[valid] * 100
    wm_v   = wm[valid]
    K_geom = schomburg_K(W_geom, t, E, nu)
    w_pred = e0_fixed + K_geom * dP_pa
    ss_res = np.sum((wm_v - w_pred)**2)
    ss_tot = np.sum((wm_v - wm_v.mean())**2)
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return r2, K_geom


# =============================================================================
# MODÈLE 2D — PLAQUE ELLIPTIQUE ENCASTRÉE (KIRCHHOFF, Timoshenko 1959)
# =============================================================================

def elliptic_w0_per_Pa(a_um, b_um, t_um, E_pa, nu):
    """
    Flèche maximale (au centre) par unité de pression, w0/P  [µm/Pa].
        w(x,y) = w0 (1 - x²/a² - y²/b²)²
        w0 = P/(8D) · 1 / (3/a⁴ + 3/b⁴ + 2/(a²b²))
        D  = E t³ / [12(1-ν²)]
    """
    a = a_um*1e-6; b = b_um*1e-6; t = t_um*1e-6
    D = E_pa * t**3 / (12*(1-nu**2))
    denom = 3/a**4 + 3/b**4 + 2/(a**2 * b**2)
    return (1.0 / (8*D) / denom) * 1e6


def elliptic_K(a_um, b_um, t_um, E_pa, nu):
    """Pente du modèle elliptique : K = w0/ΔP  [µm/Pa]."""
    return elliptic_w0_per_Pa(a_um, b_um, t_um, E_pa, nu)


def fit_elliptic_e0_Eeff_free(dP_mbar, wm, a, b, t, E0_guess, nu):
    """
    FIT : e₀ (offset) et E_eff (module effectif) LIBRES.
    Géométrie a, b, t, ν FIXÉE.
    Modèle :  wm_exp = e₀ + K_ellip(E_eff) · ΔP
    Retourne (e0_fit, E_eff_fit, r2, K_fit).
    """
    valid = np.isfinite(dP_mbar) & np.isfinite(wm)
    if valid.sum() < 3:
        return None, None, None, None
    dP_pa = dP_mbar[valid] * 100
    wm_v  = wm[valid]

    def model(dP, e0, E_eff):
        return e0 + elliptic_K(a, b, t, E_eff, nu) * dP

    try:
        popt, _ = curve_fit(
            model, dP_pa, wm_v,
            p0=[10.0, E0_guess],
            bounds=([0.0, 1e5], [50.0, 1e8]),
            maxfev=20000)
        e0_f, Eeff_f = float(popt[0]), float(popt[1])
    except Exception as ex:
        print(f"  [Fit elliptique] {ex}"); return None, None, None, None

    K_f    = elliptic_K(a, b, t, Eeff_f, nu)
    w_pred = model(dP_pa, e0_f, Eeff_f)
    ss_res = np.sum((wm_v - w_pred)**2)
    ss_tot = np.sum((wm_v - wm_v.mean())**2)
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return e0_f, Eeff_f, r2, K_f


# =============================================================================
# FIGURE 1 — SCHOMBURG
# =============================================================================

def make_figure_schomburg(params, data_G, data_P, out_dir):
    t  = float(params.get('t (µm)', 1400))
    E  = float(params.get('E (Pa)', 1.3e6))
    sE = float(params.get('σE (Pa)', 3e5))
    st = float(params.get('σt (µm)', 50.0))   # µm — incertitude épaisseur
    nu = float(params.get('ν', 0.5))
    a  = float(params.get('a géom. (µm)', 3750))
    b  = float(params.get('b géom. (µm)', 2800))

    dP_plot = np.linspace(0, X_MAX_MBAR, 400)
    dP_plot_pa = dP_plot * 100

    def to_arr(data, key):
        return np.array([d[key] for d in data], dtype=float)

    dP_G = to_arr(data_G, 'dP_mbar'); wm_G = to_arr(data_G, 'wm_fit')
    dP_P = to_arr(data_P, 'dP_mbar'); wm_P = to_arr(data_P, 'wm_fit')

    print("\n  ── FIGURE 1 — FITS SCHOMBURG (1D) ──")

    results = {}
    for nom, dP, wm, W_geom in [
            ('Grand axe', dP_G, wm_G, 2*a),
            ('Petit axe',  dP_P, wm_P, 2*b)]:
        print(f"\n  ── {nom} ──")
        e0_f1, Weff_f1, r2_f1, K_f1 = fit1_e0_Weff_free(dP, wm, t, E, nu)
        if e0_f1 is not None:
            print(f"  Fit 1 (e0 + W_eff libres) : e0={e0_f1:.3f}um  "
                  f"W_eff={Weff_f1:.0f}um (ratio {Weff_f1/W_geom:.3f})  R2={r2_f1:.4f}")
        e0_for_f2 = e0_f1 if e0_f1 is not None else E0_UM
        r2_f2, K_f2 = fit2_e0_fixed_Wgeom_fixed(dP, wm, t, E, nu, e0_for_f2, W_geom)
        if r2_f2 is not None:
            print(f"  Fit 2 (e0={e0_for_f2:.2f}um fixe, W_geom={W_geom:.0f}um) : "
                  f"K={K_f2*1e5:.4f}nm/mbar  R2={r2_f2:.4f}")
        results[nom] = {'e0_f1': e0_f1, 'Weff_f1': Weff_f1, 'r2_f1': r2_f1,
                        'K_f1': K_f1, 'e0_f2': e0_for_f2, 'r2_f2': r2_f2,
                        'K_f2': K_f2, 'W_geom': W_geom}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=False)
    fig.suptitle('Deflexion w_max vs dP — Modele de Schomburg 1D\n'
                 f't = {t:.0f}±{st:.0f} µm  |  E = {E/1e6:.2f}±{sE/1e6:.1f} MPa  |  nu = {nu}  |  '
                 f'e0 nominal = {E0_UM:.0f} µm', fontsize=12, fontweight='bold')

    for ax, nom, data, col_dark, cmap_ in [
            (axes[0], 'Grand axe', data_G, '#a93226', cm.Reds),
            (axes[1], 'Petit axe',  data_P, '#1a5276', cm.Blues)]:
        r = results[nom]; W_geom = r['W_geom']
        # Bornes E litterature appliquees a W_geom fixee (Fit 2)
        #   E_max=1.9MPa -> fleche MIN (bas, bleu) ; E_min=0.75MPa -> fleche MAX (haut, rouge)
        K_geom_Emax = schomburg_K(W_geom, t, E_MAX_PA, nu)
        K_geom_Emin = schomburg_K(W_geom, t, E_MIN_PA, nu)
        if r['K_f1'] is not None:
            z_f1 = r['e0_f1'] + r['K_f1'] * dP_plot_pa
            ax.plot(dP_plot, z_f1, '-', color=col_dark, lw=2.5,
                    label=(f'Fit 1 — e0={r["e0_f1"]:.2f}µm, W_eff={r["Weff_f1"]:.0f}µm  '
                           f'R2={r["r2_f1"]:.4f}'))
            ax.axhline(r['e0_f1'], color=col_dark, lw=0.8, ls='--', alpha=0.45)
        if r['K_f2'] is not None:
            z_f2   = r['e0_f2'] + r['K_f2']      * dP_plot_pa   # E nominal
            z_Emax = r['e0_f2'] + K_geom_Emax    * dP_plot_pa   # E=1.9 (bas)
            z_Emin = r['e0_f2'] + K_geom_Emin    * dP_plot_pa   # E=0.75 (haut)
            # Zone grisee entre bornes litterature (W_geom fixee)
            ax.fill_between(dP_plot, z_Emax, z_Emin, color='#888888', alpha=0.18)
            # Borne sup E=1.9MPa -> courbe basse, pointille BLEU
            ax.plot(dP_plot, z_Emax, '--', color='#1f4e9c', lw=1.6,
                    label=f'E_max = {E_MAX_PA/1e6:.2f} MPa (Schneider 2008)')
            # Borne inf E=0.75MPa -> courbe haute, pointille ROUGE
            ax.plot(dP_plot, z_Emin, '--', color='#c0392b', lw=1.6,
                    label=f'E_min = {E_MIN_PA/1e6:.2f} MPa (Lotters 1997)')
            # Courbe Fit 2 (E nominal, W_geom fixee) : noir plein
            ax.plot(dP_plot, z_f2, '-', color='#1a1a1a', lw=2.0,
                    label=(f'Schomburg W_geom={W_geom:.0f}µm fixee (E={E/1e6:.1f}MPa)  '
                           f'R2={r["r2_f2"]:.4f}'))
        ax.axhline(E0_UM, color='#95a5a6', lw=0.9, ls=':', alpha=0.8)
        n_pts = len(data)
        colors = [cmap_(0.38 + 0.55*i/max(n_pts-1, 1)) for i in range(n_pts)]
        for i, (d, col) in enumerate(zip(data, colors)):
            ax.scatter(d['dP_mbar'], d['wm_fit'], color=col, s=72, zorder=8,
                       edgecolors=col_dark, lw=0.7,
                       marker='o' if 'Grand' in nom else 's')
        ax_r = ax.twinx()
        ylo, yhi = ax.get_ylim(); ax_r.set_ylim(ylo + E0_UM, yhi + E0_UM)
        ax_r.set_ylabel('z absolu verre->PDMS (µm)', fontsize=8, color='#7f8c8d')
        ax_r.tick_params(axis='y', labelcolor='#7f8c8d', labelsize=7)
        ax.set_title(f'{nom}  (W_geom = {W_geom:.0f} µm)', fontsize=9)
        ax.set_xlabel('Depression dP (mbar)', fontsize=10)
        ax.set_ylabel('w_max mesure — deflexion relative (µm)', fontsize=9)
        ax.set_xlim(0, X_MAX_MBAR); ax.set_ylim(0, Y_MAX_UM)
        ax.legend(fontsize=7.5, loc='upper left', framealpha=0.92)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    if SAVE_svg and out_dir:
        fpath = os.path.join(out_dir, 'wmax_vs_dP_schomburg.svg')
        fig.savefig(fpath, dpi=150, bbox_inches='tight')
        print(f"\n  -> Figure 1 sauvegardee : {fpath}")
    plt.show(); plt.close()


# =============================================================================
# FIGURE 2 — ELLIPTIQUE (E et e0 libres)
# =============================================================================

def make_figure_elliptic(params, data_G, data_P, out_dir):
    t  = float(params.get('t (\u00b5m)', 1400))
    E  = float(params.get('E (Pa)', 1.3e6))
    sE = float(params.get('\u03c3E (Pa)', 3e5))
    nu = float(params.get('\u03bd', 0.5))
    a  = float(params.get('a g\u00e9om. (\u00b5m)', 3750))   # demi-grand axe
    b  = float(params.get('b g\u00e9om. (\u00b5m)', 2800))   # demi-petit axe
    st = float(params.get('\u03c3t (\u00b5m)', 50.0))        # incertitude \u00e9paisseur

    dP_plot = np.linspace(0, X_MAX_MBAR, 400)
    dP_plot_pa = dP_plot * 100

    def to_arr(data, key):
        return np.array([d[key] for d in data], dtype=float)
    dP_G = to_arr(data_G, 'dP_mbar'); wm_G = to_arr(data_G, 'wm_fit')
    dP_P = to_arr(data_P, 'dP_mbar'); wm_P = to_arr(data_P, 'wm_fit')

    print("\n  \u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557")
    print("  \u2551     FIGURE 2 \u2014 MOD\u00c8LE PLAQUE ELLIPTIQUE (2D)     \u2551")
    print("  \u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d")
    print(f"\n  w(x,y) = w0 (1 - x\u00b2/a\u00b2 - y\u00b2/b\u00b2)\u00b2   [Timoshenko 1959]")
    print(f"  a = {a:.0f} \u00b5m | b = {b:.0f} \u00b5m | t = {t:.0f} \u00b1 {st:.0f} \u00b5m | \u03bd = {nu}")

    # Tous les points (deux axes ensemble : m\u00eame w0 central physique)
    dP_all = np.concatenate([dP_G, dP_P])
    wm_all = np.concatenate([wm_G, wm_P])

    # \u2500\u2500 FIT : E et e\u2080 LIBRES (g\u00e9om\u00e9trie a,b,t,\u03bd fix\u00e9e) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    e0_fit, Eeff_fit, r2_fit, K_fit = fit_elliptic_e0_Eeff_free(
        dP_all, wm_all, a, b, t, E, nu)
    if e0_fit is None:
        e0_fit, Eeff_fit = E0_UM, E
        K_fit = elliptic_K(a, b, t, Eeff_fit, nu); r2_fit = 0.0
    print(f"\n  \u2500\u2500 FIT (E et e\u2080 libres, g\u00e9om\u00e9trie fix\u00e9e) \u2500\u2500")
    print(f"    e\u2080    = {e0_fit:.3f} \u00b5m")
    print(f"    E_eff = {Eeff_fit/1e6:.4f} MPa  (nominal {E/1e6:.2f} MPa)")
    print(f"    K     = {K_fit*100:.4f} \u00b5m/mbar")
    print(f"    R\u00b2    = {r2_fit:.4f}")

    # Bornes d'incertitude issues de la LITTERATURE (pas de +-sigma)
    #   E_max = 1.9 MPa (Schneider 2008)  -> fleche MIN (courbe basse, bleue)
    #   E_min = 0.75 MPa (Lotters 1997)   -> fleche MAX (courbe haute, rouge)
    # fleche \u221d 1/E : E fort => fleche faible ; E faible => fleche forte
    K_Emax = elliptic_K(a, b, t, E_MAX_PA, nu)   # borne sup E => courbe basse
    K_Emin = elliptic_K(a, b, t, E_MIN_PA, nu)   # borne inf E => courbe haute

    z_fit  = e0_fit + K_fit  * dP_plot_pa        # effectif (vert plein)
    z_Emax = e0_fit + K_Emax * dP_plot_pa        # E=1.9MPa (bleu, courbe basse)
    z_Emin = e0_fit + K_Emin * dP_plot_pa        # E=0.75MPa (rouge, courbe haute)

    # \u2500\u2500 Figure : 2 panneaux (projection grand axe / petit axe) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    fig.suptitle(
        'D\u00e9flexion w_max vs \u0394P \u2014 Mod\u00e8le de plaque elliptique 2D (Kirchhoff)\n'
        f'w(x,y)=w\u2080(1\u2212x\u00b2/a\u00b2\u2212y\u00b2/b\u00b2)\u00b2   |   a={a:.0f}\u00b5m, b={b:.0f}\u00b5m, t={t:.0f}\u00b5m   |   '
        f'E_eff={Eeff_fit/1e6:.2f}MPa entre E_min={E_MIN_PA/1e6:.2f} et E_max={E_MAX_PA/1e6:.2f}MPa',
        fontsize=11, fontweight='bold')

    for ax, nom, data, dP_arr, wm_arr, col_dark, cmap_, mk in [
            (axes[0], 'Grand axe (projection y=0)', data_G, dP_G, wm_G,
             '#a93226', cm.Reds, 'o'),
            (axes[1], 'Petit axe (projection x=0)',  data_P, dP_P, wm_P,
             '#1a5276', cm.Blues, 's')]:

        # Zone d'incertitude grisee entre les deux bornes litterature
        ax.fill_between(dP_plot, z_Emax, z_Emin, color='#888888', alpha=0.18)

        # Borne SUPERIEURE E=1.9MPa (Schneider) -> courbe basse, pointille BLEU
        ax.plot(dP_plot, z_Emax, '--', color='#1f4e9c', lw=1.6,
                label=f'E_max = {E_MAX_PA/1e6:.2f} MPa (Schneider 2008)')
        # Borne INFERIEURE E=0.75MPa (Lotters) -> courbe haute, pointille ROUGE
        ax.plot(dP_plot, z_Emin, '--', color='#c0392b', lw=1.6,
                label=f'E_min = {E_MIN_PA/1e6:.2f} MPa (Lotters 1997)')

        # FIT effectif : vert PLEIN
        ax.plot(dP_plot, z_fit, '-', color='#117733', lw=2.6, zorder=6,
                label=(f'E_eff = {Eeff_fit/1e6:.3f} MPa (fit, e\u2080={e0_fit:.2f}\u00b5m)  '
                       f'R\u00b2={r2_fit:.4f}'))

        # Ligne e\u2080
        ax.axhline(e0_fit, color='#117733', lw=0.8, ls='--', alpha=0.5)
        ax.text(X_MAX_MBAR*0.01, e0_fit + 1.2, f'e\u2080 = {e0_fit:.2f}\u00b5m',
                fontsize=7, color='#117733', va='bottom')

        # Points exp\u00e9rimentaux de CET axe
        n_pts = len(data)
        colors = [cmap_(0.38 + 0.55*i/max(n_pts-1, 1)) for i in range(n_pts)]
        for i, (d, col) in enumerate(zip(data, colors)):
            ax.scatter(d['dP_mbar'], d['wm_fit'], color=col, s=72, zorder=8,
                       edgecolors=col_dark, lw=0.7, marker=mk)
            if i == 0 or i == n_pts-1 or i % 4 == 0:
                ax.annotate(f"{d['dP_mbar']:.0f}mb", (d['dP_mbar'], d['wm_fit']),
                            textcoords='offset points', xytext=(4, 3),
                            fontsize=6, color=col_dark)

        # Axe secondaire : z absolu
        ax_r = ax.twinx()
        ax_r.set_ylim(0 + E0_UM, Y_MAX_UM + E0_UM)
        ax_r.set_ylabel('z absolu verre\u2192PDMS (\u00b5m)', fontsize=8, color='#7f8c8d')
        ax_r.tick_params(axis='y', labelcolor='#7f8c8d', labelsize=7)

        ax.set_title(nom, fontsize=10)
        ax.set_xlabel('D\u00e9pression \u0394P (mbar)', fontsize=10)
        ax.set_ylabel('w_max mesur\u00e9 \u2014 d\u00e9flexion relative (\u00b5m)', fontsize=9)
        ax.set_xlim(0, X_MAX_MBAR); ax.set_ylim(0, Y_MAX_UM)
        ax.legend(fontsize=7.5, loc='upper left', framealpha=0.92)
        ax.grid(True, alpha=0.25)

    # Encadr\u00e9 explicatif
    axes[0].text(0.02, 0.52,
                 "E_eff (vert) ajuste librement,\n"
                 "encadre par les bornes litterature :\n"
                 "E_min=0.75MPa (Lotters, rouge) et\n"
                 "E_max=1.9MPa (Schneider, bleu).",
                 transform=axes[0].transAxes, fontsize=7.5,
                 color='#333333', va='top',
                 bbox=dict(boxstyle='round', fc='#f2f2f2', ec='#888888', alpha=0.85))

    plt.tight_layout()
    if SAVE_svg and out_dir:
        fpath = os.path.join(out_dir, 'wmax_vs_dP_elliptic2D.svg')
        fig.savefig(fpath, dpi=150, bbox_inches='tight')
        print(f"\n  \u2192 Figure 2 sauvegard\u00e9e : {fpath}")
    plt.show(); plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    xl = EXCEL_FILE or ask_file()
    if not xl:
        print("Aucun fichier sélectionné."); return
    out_dir = os.path.dirname(xl)
    print(f"\n=== FIGURES w_max vs ΔP ===\n  Fichier : {xl}")
    params, data_G, data_P = read_excel(xl)

    # Figure 1 — Schomburg 1D
    make_figure_schomburg(params, data_G, data_P, out_dir)
    # Figure 2 — Plaque elliptique 2D (E et e₀ libres + incertitude E,t)
    make_figure_elliptic(params, data_G, data_P, out_dir)

    print("\n  Terminé.")


main()