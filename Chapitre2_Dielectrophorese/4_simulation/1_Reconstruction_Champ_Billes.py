# -*- coding: utf-8 -*-
# =============================================================================
# CHAMP DIELECTROPHORETIQUE : RECONSTRUCTION, SIMULATION ET COMPARAISON
# Billes de polystyrene de reference, electrode annulaire
# =============================================================================
#
# Le programme est organise en trois sections independantes.
#
#   SECTION A, chemin ASCENDANT : des billes vers le champ.
#       vitesse mesuree -> force -> gradient -> carre du champ -> champ
#       -> potentiel effectif
#       Aucune resolution de Laplace. Le champ obtenu est celui que subissent
#       effectivement les billes.
#
#   SECTION B, chemin DESCENDANT : du potentiel vers les vitesses.
#       potentiel impose -> Laplace -> champ -> gradient -> force
#       -> vitesse predite
#       Aucun parametre ajuste. C'est la prediction du modele pour la
#       geometrie et la tension declarees.
#
#   SECTION C : confrontation des deux.
#       Les deux chemins partent des memes particules et de la meme geometrie.
#       Leur ecart mesure donc ce que le modele ne decrit pas, et sa
#       dependance radiale en indique la nature.
#
# -----------------------------------------------------------------------------
# FIGURES PRODUITES
# -----------------------------------------------------------------------------
# Chaque panneau est enregistre entier, et chacune de ses vignettes l'est aussi
# separement, sous le meme numero suivi d'une lettre.
#
#   01  cinematique              trajectoires, positions, vitesses, profil moyen
#   02  chaine ascendante        vitesse, force, gradient, carre du champ
#   03  champ et potentiel       champ reconstruit, extrapolation, potentiel
#   04  simulation directe       champ de Laplace en deux dimensions et profils
#   05  comparaison              les deux chemins confrontes
#
# -----------------------------------------------------------------------------
# DEPENDANCES
# -----------------------------------------------------------------------------
#   pip install numpy scipy matplotlib pandas openpyxl
# =============================================================================

import os
import json
import math

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    TK_DISPO = True
except ImportError:
    TK_DISPO = False
    print("[ALERTE] tkinter indisponible : les boites de dialogue ne peuvent "
          "pas s'ouvrir. Les fonctions de calcul restent importables.")


# =============================================================================
# CONFIGURATION
# =============================================================================

R_INT_UM        = 50.0
R_EXT_UM        = 140.0
HAUTEUR_CAV_UM  = 20.0
Z_MESURE_UM     = 5.0

TENSION_AFFICHEE = 10.0
TENSION_EN_CRETE_A_CRETE = True
FREQUENCE_KHZ = 100.0
FORME_SIGNAL  = "creneau"

EPS_MILIEU_REL  = 78.0
SIGMA_MILIEU    = 0.120
VISCOSITE_MPA_S = 1.0
RHO_MILIEU      = 1000.0

RAYON_BILLE_UM = 3.0
EPS_BILLE_REL  = 2.55
SIGMA_BILLE    = 6.7e-4

R_MIN_UM = 165.0

# --- Plage retenue pour l'extrapolation --------------------------------------
# L'extrapolation vers le bord de l'electrode ne doit s'appuyer que sur la
# portion ou le mouvement est effectivement dielectrophoretique. Au dela d'un
# certain rayon, le gradient de champ est devenu negligeable et les billes
# derivent sous l'effet d'autre chose : y ajuster la decroissance fausserait la
# longueur, et donc toute la remontee vers le potentiel.
R_MAX_EXTRAPOLATION_UM = 220.0

SEUIL_ACTIVATION_UM_S = 5.0
POINTS_CONSECUTIFS = 3
POINTS_ECARTES = 2
LISSAGE_DETECTION_S = 0.30

# Ecart au dela duquel deux instants d'activation sont attribues a des
# acquisitions differentes. Au sein d'une meme video, le champ est applique une
# seule fois et les detections ne peuvent differer que du temps de reponse de
# la detection, soit une fraction de seconde. Un ecart de plusieurs secondes
# signale donc un changement de video.
ECART_MAX_MEME_ACQUISITION = 1.5

# Mode de recalage temporel.
#
#   "individuel" chaque bille est recalee sur SON PROPRE instant d'activation,
#                de sorte que toutes les courbes partent exactement de zero.
#                C'est le mode a retenir pour comparer des trajectoires entre
#                elles ou a une simulation : la latence de detection, qui
#                depend de la vitesse locale et donc de la position, ne vient
#                plus decaler les courbes les unes par rapport aux autres.
#
#   "acquisition" toutes les billes d'une meme video partagent l'instant median
#                du groupe. C'est plus fidele a la physique, le champ n'etant
#                applique qu'une fois, mais les billes lentes semblent alors
#                demarrer en retard.
MODE_RECALAGE = "individuel"

# Les points anterieurs a l'activation sont CONSERVES et portes en temps
# negatif. Ils montrent l'etat de repos avant application du champ, ce qui
# permet de verifier d'un coup d'oeil que la rupture de pente est bien detectee
# et que le zero tombe au bon endroit.
CONSERVER_AVANT_ACTIVATION = True
DUREE_AVANT_ACTIVATION_S = 2.0

ECHELLE_PX_PAR_UM  = 3.06
PAS_MINIMAUX       = 3.0
FENETRES_CANDIDATES = (5, 9, 15, 25, 41, 65, 101, 151)
LISSAGE_ORDRE = 2

NB_BANDES = 40
POINTS_MIN_PAR_BANDE = 3

# Maillage de la simulation directe
NR_SIMU = 500
NZ_SIMU = 200
R_MAX_SIMU_UM = 500.0

# --- Plage radiale des figures -----------------------------------------------
# Toutes les echelles sont lineaires. Le champ variant de plusieurs decades
# entre le bord de l'electrode et le champ lointain, une echelle lineaire ne
# reste lisible que sur une plage bornee : au dela, toutes les courbes se
# confondent avec l'axe. Cette borne ne concerne que l'affichage, jamais le
# calcul.
R_AFFICHAGE_UM = 260.0

COLONNES_TEMPS = ['temps_s', 'time_s', 'temps', 'time', 't_s', 't']
COLONNES_RAYON = ['r_um', 'rayon_um', 'radius_um', 'r', 'rayon', 'distance_um']
COLONNES_ANGLE = ['theta_deg', 'angle_deg', 'theta', 'angle']

EPSILON_0 = 8.854187817e-12
GRAVITE   = 9.81

AFFICHER_PANNEAUX = True
ENREGISTRER_VIGNETTES = True

matplotlib.rcParams['figure.facecolor']  = 'white'
matplotlib.rcParams['savefig.facecolor'] = 'white'
matplotlib.rcParams['axes.facecolor']    = 'white'

COUL_V    = 'magenta'
COUL_F    = '#2ca02c'
COUL_GRAD = '#d62728'
COUL_E    = '#1f77b4'
COUL_SIMU = '#9467bd'
COUL_FIT  = 'darkorange'
COUL_OR   = 'gold'
COUL_GRIS = '#7f7f7f'


# =============================================================================
# PHYSIQUE COMMUNE
# =============================================================================

def facteur_forme_onde(forme=FORME_SIGNAL):
    # Rapport entre la moyenne temporelle du carre du champ et le carre de la
    # valeur crete. Un demi pour une sinusoide, un pour un creneau.
    if forme.lower().startswith("sin"):
        return 0.5
    if forme.lower().startswith("cre"):
        return 1.0
    raise ValueError("FORME_SIGNAL doit valoir 'creneau' ou 'sinus'.")


def facteur_clausius_mossotti(p):
    w = 2.0 * math.pi * p['frequence']
    ep = p['eps_bille'] - 1j * p['sigma_bille'] / w
    em = p['eps_milieu'] - 1j * p['sigma_milieu'] / w
    return float(np.real((ep - em) / (ep + 2.0 * em)))


def coefficient_trainee(rayon, hauteur, viscosite):
    # Trainee corrigee du confinement, modele de Happel et Brenner.
    g0 = 6.0 * math.pi * viscosite * rayon
    lam = rayon / (hauteur / 2.0)
    if lam < 0.95:
        den = (1.0 - 1.004 * lam + 0.418 * lam ** 3
               + 0.21 * lam ** 4 - 0.169 * lam ** 5)
        f = 1.0 / den
    else:
        f = 1.0 + (9.0 / 8.0) * lam + lam ** 3
    return g0 * f, f, lam   ## A MODIFIER EN FONCTION


def prefacteur_dep(p, re_k):
    # 2 pi a^3 eps_m Re[K], le rayon etant celui de la PARTICULE.
    return 2.0 * math.pi * p['r_bille'] ** 3 * p['eps_milieu'] * re_k


# =============================================================================
# INTERFACE
# =============================================================================

_racine = None


def racine_tk():
    global _racine
    if _racine is None:
        _racine = tk.Tk()
        _racine.withdraw()
    return _racine


def saisir_parametres():
    racine = racine_tk()
    res = {'p': None}
    fen = tk.Toplevel(racine)
    fen.title("Champ dielectrophoretique")
    fen.resizable(False, False)
    fen.lift(); fen.focus_force(); fen.grab_set()
    st = ttk.Style()
    st.configure('T.TLabel', font=('Arial', 12, 'bold'))
    st.configure('S.TLabel', font=('Arial', 10, 'bold'))
    ttk.Label(fen, text="RECONSTRUCTION, SIMULATION ET COMPARAISON",
              style='T.TLabel').grid(row=0, column=0, columnspan=2,
                                     pady=(14, 10))
    g = ttk.Frame(fen, padding="14"); g.grid(row=1, column=0, sticky='n')
    d = ttk.Frame(fen, padding="14"); d.grid(row=1, column=1, sticky='n')
    ch = {}

    def bloc(parent, titre, entrees, depart):
        ttk.Label(parent, text=titre, style='S.TLabel').grid(
            row=depart, column=0, columnspan=2, sticky='w', pady=(10, 4))
        ttk.Separator(parent, orient='horizontal').grid(
            row=depart + 1, column=0, columnspan=2, sticky='ew', pady=(0, 6))
        li = depart + 2
        for cle, lib, dft in entrees:
            ttk.Label(parent, text=lib).grid(row=li, column=0, sticky='w',
                                             pady=3)
            e = ttk.Entry(parent, width=14); e.insert(0, str(dft))
            e.grid(row=li, column=1, sticky='w', padx=(10, 0), pady=3)
            ch[cle] = e; li += 1
        return li

    li = bloc(g, "Geometrie", [
        ('r_int', "Rayon interieur (um)", R_INT_UM),
        ('r_ext', "Rayon exterieur (um)", R_EXT_UM),
        ('gap', "Hauteur de cavite (um)", HAUTEUR_CAV_UM),
        ('z', "Hauteur des billes (um)", Z_MESURE_UM)], 0)
    li = bloc(g, "Signal", [
        ('tension', "Tension affichee (V)", TENSION_AFFICHEE),
        ('frequence', "Frequence (kHz)", FREQUENCE_KHZ)], li)
    ttk.Label(g, text="Convention").grid(row=li, column=0, sticky='w', pady=3)
    v_cc = tk.StringVar(master=fen, value="crete a crete"
                        if TENSION_EN_CRETE_A_CRETE else "amplitude")
    ttk.Combobox(g, textvariable=v_cc, values=["amplitude", "crete a crete"],
                 width=12, state="readonly").grid(row=li, column=1, sticky='w',
                                                  padx=(10, 0), pady=3)
    li += 1
    ttk.Label(g, text="Forme").grid(row=li, column=0, sticky='w', pady=3)
    v_f = tk.StringVar(master=fen, value=FORME_SIGNAL)
    ttk.Combobox(g, textvariable=v_f, values=["creneau", "sinus"], width=12,
                 state="readonly").grid(row=li, column=1, sticky='w',
                                        padx=(10, 0), pady=3)
    li += 1
    li = bloc(g, "Milieu", [
        ('eps_milieu', "Permittivite relative", EPS_MILIEU_REL),
        ('sigma_milieu', "Conductivite (S/m)", SIGMA_MILIEU),
        ('viscosite', "Viscosite (mPa.s)", VISCOSITE_MPA_S),
        ('rho_milieu', "Masse volumique (kg/m3)", RHO_MILIEU)], li)

    li = bloc(d, "Billes de polystyrene", [
        ('r_bille', "Rayon (um)", RAYON_BILLE_UM),
        ('eps_bille', "Permittivite relative", EPS_BILLE_REL),
        ('sigma_bille', "Conductivite (S/m)", SIGMA_BILLE)], 0)
    li = bloc(d, "Traitement", [
        ('r_min', "Rayon minimal retenu (um)", R_MIN_UM),
        ('seuil', "Seuil d'activation (um/s)", SEUIL_ACTIVATION_UM_S),
        ('ecartes', "Points ecartes apres activation", POINTS_ECARTES),
        ('bandes', "Nombre de bandes", NB_BANDES)], li)
    li = bloc(d, "Simulation directe", [
        ('nr', "Resolution radiale", NR_SIMU),
        ('rmax', "Rayon du domaine (um)", R_MAX_SIMU_UM)], li)

    bt = ttk.Frame(fen); bt.grid(row=2, column=0, columnspan=2, pady=(6, 16))

    def valider():
        try:
            t = float(ch['tension'].get())
            amp = t / 2.0 if v_cc.get() == "crete a crete" else t
            res['p'] = dict(
                r_int=float(ch['r_int'].get()) * 1e-6,
                r_ext=float(ch['r_ext'].get()) * 1e-6,
                gap=float(ch['gap'].get()) * 1e-6,
                z=float(ch['z'].get()) * 1e-6,
                tension_affichee=t, convention=v_cc.get(), amplitude=amp,
                frequence=float(ch['frequence'].get()) * 1e3, forme=v_f.get(),
                eps_milieu=float(ch['eps_milieu'].get()) * EPSILON_0,
                eps_milieu_rel=float(ch['eps_milieu'].get()),
                sigma_milieu=float(ch['sigma_milieu'].get()),
                viscosite=float(ch['viscosite'].get()) * 1e-3,
                rho_milieu=float(ch['rho_milieu'].get()),
                r_bille=float(ch['r_bille'].get()) * 1e-6,
                eps_bille=float(ch['eps_bille'].get()) * EPSILON_0,
                eps_bille_rel=float(ch['eps_bille'].get()),
                sigma_bille=float(ch['sigma_bille'].get()),
                r_min=float(ch['r_min'].get()) * 1e-6,
                seuil=float(ch['seuil'].get()) * 1e-6,
                ecartes=int(ch['ecartes'].get()),
                bandes=int(ch['bandes'].get()),
                nr=int(ch['nr'].get()),
                r_max=float(ch['rmax'].get()) * 1e-6,
                h_electrode=1e-6)
            fen.grab_release(); fen.destroy()
        except Exception as e:
            import traceback; traceback.print_exc()
            messagebox.showerror("Erreur", f"Valeur invalide :\n\n{e}")

    def annuler():
        res['p'] = None; fen.grab_release(); fen.destroy()

    ttk.Button(bt, text="Valider", command=valider, width=14).pack(
        side=tk.LEFT, padx=8)
    ttk.Button(bt, text="Annuler", command=annuler, width=14).pack(
        side=tk.LEFT, padx=8)
    fen.protocol("WM_DELETE_WINDOW", annuler)
    fen.wait_window()
    return res['p']


# =============================================================================
# LECTURE ET CINEMATIQUE
# =============================================================================

def _colonne(cols, cands):
    norm = {str(c).strip().lower().replace(' ', '_'): c for c in cols}
    for c in cands:
        if c in norm:
            return norm[c]
    for cle, orig in norm.items():
        for c in cands:
            if c in cle:
                return orig
    return None


def lire_trajectoires(chemin, cadence=20.0):
    cl = pd.ExcelFile(chemin)
    mo = []
    print(f"  {len(cl.sheet_names)} onglets detectes")
    for i, ong in enumerate(cl.sheet_names, 1):
        df = cl.parse(ong)
        if df.empty or len(df) < 8:
            continue
        df = df.copy(); df['particule'] = i; df['nom'] = ong
        mo.append(df)
    if not mo:
        return pd.DataFrame()
    b = pd.concat(mo, ignore_index=True)
    cr = _colonne(b.columns, COLONNES_RAYON)
    if cr is None:
        raise ValueError("aucune colonne de rayon reconnue")
    ct = _colonne(b.columns, COLONNES_TEMPS)
    ca = _colonne(b.columns, COLONNES_ANGLE)
    s = pd.DataFrame()
    s['particule'] = b['particule']; s['nom'] = b['nom']
    s['r_um'] = pd.to_numeric(b[cr], errors='coerce')
    s['temps_s'] = (pd.to_numeric(b[ct], errors='coerce') if ct is not None
                    else b.groupby('particule').cumcount() / cadence)
    s['theta_deg'] = (pd.to_numeric(b[ca], errors='coerce')
                      if ca is not None else np.nan)
    return s.dropna(subset=['r_um', 'temps_s']).reset_index(drop=True)


def vitesse_adaptative(t, r, pas_q=None, n_pas=PAS_MINIMAUX,
                       fenetres=FENETRES_CANDIDATES, ordre=LISSAGE_ORDRE):
    # Derivee sur une fenetre elargie localement jusqu'a ce que le deplacement
    # depasse plusieurs pas de quantification du suivi. Elle reste courte la ou
    # la bille va vite, ce qui preserve le maximum de vitesse.
    n = len(t)
    if n < 5:
        return (np.gradient(r, t) if n > 1 else np.zeros(n),
                np.zeros(n, bool), np.zeros(n, int))
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        return np.gradient(r, t), np.ones(n, bool), np.zeros(n, int)
    if pas_q is None:
        pas_q = 1.0e-6 / ECHELLE_PX_PAR_UM
    der = {}
    for f in fenetres:
        ff = min(f, n if n % 2 == 1 else n - 1)
        if ff % 2 == 0:
            ff -= 1
        if ff > ordre + 1:
            der[ff] = savgol_filter(r, ff, ordre, deriv=1, delta=dt)
    if not der:
        v = np.gradient(r, t); fi = np.ones(n, bool); fi[0] = fi[-1] = False
        return v, fi, np.full(n, 3)
    idx = np.arange(n); v = np.full(n, np.nan); fen = np.zeros(n, int)
    for f in sorted(der):
        dd = f // 2
        a = np.clip(idx - dd, 0, n - 1); b = np.clip(idx + dd, 0, n - 1)
        ok = np.isnan(v) & (np.abs(r[b] - r[a]) >= n_pas * pas_q)
        v[ok] = der[f][ok]; fen[ok] = f
    reste = np.isnan(v); fm = max(der)
    v[reste] = der[fm][reste]; fen[reste] = fm
    dl = fen // 2
    return v, (idx - dl >= 0) & (idx + dl <= n - 1), fen


def vitesse_lissee(t, r, duree, ordre=LISSAGE_ORDRE):
    n = len(t)
    if n < 5:
        return np.gradient(r, t) if n > 1 else np.zeros(n)
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        return np.gradient(r, t)
    f = int(round(duree / dt))
    f = max(5, min(f, 151, n if n % 2 == 1 else n - 1))
    if f % 2 == 0:
        f -= 1
    return (savgol_filter(r, f, ordre, deriv=1, delta=dt) if f > ordre + 1
            else np.gradient(r, t))


def regrouper_par_acquisition(detections, ecart_max=None):
    # Regroupe les billes par acquisition, a partir de leurs seuls instants
    # d'activation.
    #
    # Les billes suivies proviennent souvent de plusieurs videos, chacune ayant
    # son propre instant de mise sous tension. Recaler tout le monde sur une
    # mediane unique melangerait ces origines et decalerait des groupes entiers
    # de plusieurs secondes.
    #
    # Le regroupement ne repose PAS sur la numerotation des onglets, qui n'est
    # qu'une convention et peut changer d'un jeu a l'autre, mais sur les
    # instants eux-memes : les detections sont triees, et une coupure est
    # placee partout ou l'ecart entre deux instants consecutifs depasse un
    # seuil. Deux billes d'une meme video ne peuvent en effet differer que du
    # temps de reponse de la detection, soit une fraction de seconde.
    if ecart_max is None:
        ecart_max = ECART_MAX_MEME_ACQUISITION
    if not detections:
        return {}, {}

    pids = sorted(detections, key=lambda k: detections[k])
    groupes = {}
    numero = 0
    groupes[pids[0]] = numero
    for precedent, courant in zip(pids[:-1], pids[1:]):
        if detections[courant] - detections[precedent] > ecart_max:
            numero += 1
        groupes[courant] = numero

    # Instant de reference de chaque groupe : la mediane de ses detections
    instants = {}
    for g in set(groupes.values()):
        membres = [detections[k] for k in groupes if groupes[k] == g]
        instants[g] = float(np.median(membres))
    return groupes, instants


def synchroniser(df, p):
    # Recale l'axe des temps sur l'instant d'application du champ.
    #
    # Chaque acquisition ayant son propre instant de mise sous tension, le
    # recalage se fait PAR GROUPE et non globalement. Toutes les billes se
    # retrouvent ainsi a l'origine des temps au moment ou leur champ a ete
    # applique, ce qui les rend comparables entre elles et aux simulations.
    print("\n" + "=" * 70)
    print("SYNCHRONISATION TEMPORELLE")
    print("=" * 70)

    det, blocs, noms = {}, [], {}
    for pid in sorted(df['particule'].unique()):
        q = df[df['particule'] == pid].sort_values('temps_s').copy()
        t = q['temps_s'].values
        r = q['r_um'].values * 1e-6
        vd = vitesse_lissee(t, r, LISSAGE_DETECTION_S)
        dep = np.abs(vd) > p['seuil']
        ind = None
        for i in range(len(dep) - POINTS_CONSECUTIFS + 1):
            if np.all(dep[i:i + POINTS_CONSECUTIFS]):
                ind = i
                break
        noms[pid] = q['nom'].iloc[0]
        if ind is not None:
            det[pid] = float(t[ind])
        blocs.append((pid, q, t))

    if not det:
        print("  [ERREUR] aucune activation detectee.")
        return pd.DataFrame(), None, {}

    groupes, instants = regrouper_par_acquisition(
        det, p.get('ecart_acquisition', ECART_MAX_MEME_ACQUISITION))

    print(f"  {len(instants)} acquisition(s) distincte(s) detectee(s), "
          f"sur le critere d'un ecart superieur a "
          f"{p.get('ecart_acquisition', ECART_MAX_MEME_ACQUISITION):.1f} s\n")
    for g in sorted(instants):
        membres = [k for k in groupes if groupes[k] == g]
        print(f"  ACQUISITION {g + 1}, activation a t = {instants[g]:.3f} s")
        for pid in sorted(membres):
            print(f"    {noms[pid]:16s} detecte a {det[pid]:6.3f} s, "
                  f"ecart {det[pid] - instants[g]:+.3f} s")
        etendue = max(det[k] for k in membres) - min(det[k] for k in membres)
        if etendue > 1.0:
            print(f"    [ALERTE] dispersion de {etendue:.2f} s dans ce groupe :")
            print(f"             certaines detections portent peut-etre sur du")
            print(f"             bruit, ou deux acquisitions ont ete fusionnees.")
        print()

    # Les billes sans detection propre heritent du groupe le plus proche en
    # temps, plutot que d'etre perdues.
    for pid, q, t in blocs:
        if pid in groupes:
            continue
        t_milieu = float(np.median(t))
        g = min(instants, key=lambda k: abs(instants[k] - t_milieu))
        groupes[pid] = g
        print(f"  {noms[pid]:16s} sans detection propre, rattachee a "
              f"l'acquisition {g + 1}")

    mode = p.get('mode_recalage', MODE_RECALAGE)
    garder_avant = p.get('conserver_avant', CONSERVER_AVANT_ACTIVATION)
    duree_avant = p.get('duree_avant', DUREE_AVANT_ACTIVATION_S)
    print(f"  mode de recalage : {mode}")

    rec = []
    for pid, q, t in blocs:
        g = groupes.get(pid)
        if g is None:
            continue
        # En mode individuel, chaque bille a sa propre origine : sa detection
        # si elle existe, celle du groupe sinon.
        t_ref = det.get(pid, instants[g]) if mode == "individuel" else instants[g]

        w = q.copy()
        w['temps_origine'] = t
        w['temps_s'] = t - t_ref
        w['acquisition'] = g
        w['activation_s'] = t_ref

        if garder_avant:
            # Les points anterieurs sont gardes, bornes a quelques secondes
            # avant l'activation pour ne pas etirer inutilement les figures.
            w = w[w['temps_s'] >= -duree_avant].copy()
        else:
            w = w[w['temps_s'] >= 0].copy()

        # Les points ecartes le sont APRES l'activation seulement : ceux d'avant
        # ne servent qu'a montrer l'etat de repos et n'entrent dans aucun calcul.
        if p['ecartes'] > 0:
            apres = w['temps_s'] >= 0
            if int(np.sum(apres)) > p['ecartes'] + 5:
                indices = np.where(apres.values)[0][:p['ecartes']]
                w = w.drop(w.index[indices])

        if int(np.sum(w['temps_s'] >= 0)) >= 8:
            rec.append(w)

    if not rec:
        print("  [ERREUR] aucune bille exploitable apres recalage.")
        return pd.DataFrame(), None, {}

    sortie = pd.concat(rec, ignore_index=True)
    n_avant = int(np.sum(sortie['temps_s'] < 0))
    n_apres = int(np.sum(sortie['temps_s'] >= 0))
    print(f"  {len(rec)} billes recalees")
    print(f"    {n_apres} points sous champ, portes en temps positif")
    if n_avant:
        print(f"    {n_avant} points avant activation, portes en temps negatif")

    # Verification que le recalage a bien fonctionne
    departs = [float(sortie[sortie['particule'] == pid]['temps_s'][
        sortie[sortie['particule'] == pid]['temps_s'] >= 0].min())
        for pid in sortie['particule'].unique()]
    etendue = max(departs) - min(departs)
    print(f"\n  premier point sous champ : de {min(departs):+.3f} a "
          f"{max(departs):+.3f} s")
    if etendue < 0.05:
        print(f"  toutes les billes partent au meme instant a "
              f"{etendue*1e3:.0f} ms pres : le recalage est correct.")
    else:
        print(f"  [ALERTE] etendue de {etendue:.3f} s entre les departs.")
        print(f"           en mode acquisition, c'est attendu : les billes")
        print(f"           lentes sont detectees plus tard. passer en mode")
        print(f"           individuel pour les aligner.")
    print("=" * 70)
    return sortie, instants, groupes


def calculer_vitesses(df):
    li = []
    for pid in sorted(df['particule'].unique()):
        q = df[df['particule'] == pid].sort_values('temps_s')
        t = q['temps_s'].values; r = q['r_um'].values * 1e-6
        v, fi, fe = vitesse_adaptative(t, r)
        for i in range(len(q)):
            li.append(dict(particule=pid, nom=q['nom'].iloc[0], temps_s=t[i],
                           r_um=r[i] * 1e6, theta_deg=q['theta_deg'].values[i],
                           v_r=v[i], fiable=bool(fi[i])))
    return pd.DataFrame(li)


def profil_moyen(don, nb, r_min):
    # Profil de vitesse moyen par bandes radiales. Toutes les billes sont
    # conservees ; seuls sont ecartes les points de vitesse negative, qui ne
    # peuvent pas correspondre a une repulsion.
    print("\n" + "=" * 70)
    print("SECTION A, ETAPE 1 : PROFIL DE VITESSE MOYEN")
    print("=" * 70)
    d = don[don['fiable'] & (don['v_r'] > 0)]
    print(f"  {len(d)} points sur {len(don)}, {d['particule'].nunique()} billes")
    r = d['r_um'].values * 1e-6; v = d['v_r'].values
    bo = np.linspace(r.min(), r.max(), nb + 1)
    ce, mo, ec, ef = [], [], [], []
    for i in range(nb):
        m = (r >= bo[i]) & (r < bo[i + 1])
        if int(np.sum(m)) < POINTS_MIN_PAR_BANDE:
            continue
        ce.append(0.5 * (bo[i] + bo[i + 1])); mo.append(float(np.mean(v[m])))
        ec.append(float(np.std(v[m]))); ef.append(int(np.sum(m)))
    comp = dict(r=np.array(ce), v=np.array(mo), ecart=np.array(ec),
                n=np.array(ef))
    g = comp['r'] >= r_min
    ret = {k: val[g] for k, val in comp.items()}
    print(f"  {len(comp['r'])} bandes, dont {int(np.sum(g))} au dela de "
          f"{r_min*1e6:.0f} um")
    print("=" * 70)
    return comp, ret


# =============================================================================
# SECTION A : CHEMIN ASCENDANT
# =============================================================================

def integrer_gradient(r, grad):
    # Integration radiale du gradient, du plus grand rayon vers l'interieur,
    # avec pour condition l'annulation du carre du champ au dela du domaine.
    o = np.argsort(r); rt, gt = r[o], grad[o]
    e2 = np.zeros_like(rt)
    for i in range(len(rt) - 2, -1, -1):
        dr = rt[i + 1] - rt[i]
        e2[i] = e2[i + 1] - 0.5 * (gt[i] + gt[i + 1]) * dr
    if np.min(e2) < 0:
        e2 = e2 - np.min(e2)
    return rt, e2


def modele_decroissance(r, amplitude, longueur, r_bord):
    return amplitude * np.exp(-np.maximum(r - r_bord, 0.0) / longueur)


def extrapoler(r, e2, p, r_max=None):
    # Ajustement d'une decroissance exponentielle a partir du bord de
    # l'electrode, ou le champ est maximal. La position de ce maximum n'est pas
    # ajustee : elle est imposee au rayon exterieur, donnee de fabrication.
    # L'ajustement porte sur le logarithme, faute de quoi il serait domine par
    # les seuls points de plus grande valeur.
    if r_max is None:
        r_max = p.get('r_max_extrap', R_MAX_EXTRAPOLATION_UM * 1e-6)
    ok = np.isfinite(e2) & (e2 > 0) & (r <= r_max)
    if int(np.sum(ok)) < 4:
        print(f"  [ALERTE] moins de 4 points sous {r_max*1e6:.0f} um, "
              f"extrapolation sur toute la plage")
        ok = np.isfinite(e2) & (e2 > 0)
    if int(np.sum(ok)) < 4:
        return None
    ra, ea = r[ok], e2[ok]

    def ml(rr, la, L):
        return la - np.maximum(rr - p['r_ext'], 0.0) / L

    try:
        po, _ = curve_fit(ml, ra, np.log(ea),
                          p0=[float(np.log(np.max(ea))), 20e-6],
                          bounds=([-np.inf, 1e-6], [np.inf, 1e-3]),
                          maxfev=40000)
    except Exception as e:
        print(f"  [ALERTE] extrapolation non convergee : {e}")
        return None
    res = np.log(ea) - ml(ra, *po)
    sst = float(np.sum((np.log(ea) - np.mean(np.log(ea))) ** 2))
    return dict(amplitude=float(np.exp(po[0])), longueur=float(po[1]),
                r_bord=p['r_ext'], r2=1 - float(np.sum(res ** 2)) / sst,
                r_min=float(ra.min()), r_max=float(ra.max()),
                n_points=int(np.sum(ok)))


def section_a(profil, p, re_k, gamma):
    # Chaine ascendante complete, des vitesses au potentiel.
    print("\n" + "=" * 70)
    print("SECTION A : CHEMIN ASCENDANT, DES BILLES VERS LE CHAMP")
    print("=" * 70)

    force = gamma * profil['v']
    force_ec = gamma * profil['ecart']
    print(f"\n  ETAPE 2, force = gamma x vitesse")
    print(f"    gamma = {gamma:.4e} N.s/m")
    print(f"    force de {force.min()*1e12:.3f} a {force.max()*1e12:.2f} pN")

    pref = prefacteur_dep(p, re_k)
    grad = force / pref
    print(f"\n  ETAPE 3, gradient = force / (2 pi a^3 eps_m Re[K])")
    print(f"    prefacteur = {pref:.4e}")
    print(f"    gradient de {grad.min():.3e} a {grad.max():.3e} V2/m3")

    r_e2, e2 = integrer_gradient(profil['r'], grad)
    print(f"\n  ETAPE 4, integration radiale")
    print(f"    carre du champ de {e2.min():.3e} a {e2.max():.3e} V2/m2")

    ex = extrapoler(r_e2, e2, p)
    print(f"\n  ETAPE 5, extrapolation jusqu'au bord de l'electrode")
    if ex:
        print(f"    plage retenue            = {ex['r_min']*1e6:.0f} a "
              f"{ex['r_max']*1e6:.0f} um, {ex['n_points']} points")
        print(f"    longueur de decroissance = {ex['longueur']*1e6:.2f} um")
        print(f"    carre du champ au bord   = {ex['amplitude']:.4e} V2/m2")
        print(f"    qualite                  = {ex['r2']:.4f}")
        # La remontee depend entierement de la longueur ajustee
        d = ex['r_min'] - p['r_ext']
        facteur = math.exp(d / ex['longueur'])
        facteur_laplace = math.exp(d / (p['gap'] / math.pi))
        print(f"\n    la remontee franchit {d*1e6:.0f} um, du premier point "
              f"mesure au bord.")
        print(f"    facteur exp(distance / L) = {facteur:.1f} avec la longueur "
              f"mesuree,")
        print(f"    contre {facteur_laplace:.0f} avec la longueur de Laplace "
              f"h/pi = {p['gap']/math.pi*1e6:.1f} um.")
        print(f"    tout ecart sur L se repercute donc de facon exponentielle "
              f"sur le potentiel.")

    pot = None
    if ex:
        e_bord = math.sqrt(max(ex['amplitude'], 0.0))
        v_est = e_bord * p['gap']
        pot = dict(e_bord=e_bord, v_estime=v_est,
                   v_amplitude=p['amplitude'], rapport=v_est / p['amplitude'])
        print(f"\n  ETAPE 6, potentiel effectif")
        print(f"    champ au bord     = {e_bord:.4e} V/m")
        print(f"    potentiel estime  = {v_est:.3f} V")
        print(f"    amplitude imposee = {p['amplitude']:.2f} V")
        print(f"    rapport           = {100*pot['rapport']:.0f} %")

    # La forme obtenue est-elle celle d'un champ de Laplace ?
    if ex:
        h_deduit = ex['longueur'] * math.pi
        print(f"\n  FORME DU CHAMP RECONSTRUIT")
        print(f"    hors du metal, un champ de Laplace decroit comme")
        print(f"    exp(-pi r / h) pour son carre, soit une longueur h/pi.")
        print(f"    la longueur mesuree, {ex['longueur']*1e6:.1f} um, correspond")
        print(f"    donc a une cavite de {h_deduit*1e6:.0f} um.")
        rap = h_deduit / p['gap']
        if rap > 1.5 or rap < 0.67:
            print(f"    or la cavite mesuree vaut {p['gap']*1e6:.0f} um, "
                  f"soit {rap:.1f} fois moins.")
            print(f"    le champ reconstruit n'a donc PAS la forme d'un champ")
            print(f"    de Laplace pour cette geometrie : une partie du")
            print(f"    mouvement des billes n'est pas dielectrophoretique.")
        else:
            print(f"    ce qui concorde avec la cavite mesuree.")
    print("=" * 70)

    return dict(force=force, force_ecart=force_ec, gradient=grad,
                prefacteur=pref, r_e2=r_e2, e2=e2,
                e=np.sqrt(np.maximum(e2, 0.0)), extrapolation=ex,
                potentiel=pot)


# =============================================================================
# SECTION B : CHEMIN DESCENDANT
# =============================================================================

def resoudre_laplace(p, bavard=True):
    # Equation de Laplace en coordonnees (r, z), assemblage vectorise.
    #
    # Conditions aux limites : potentiel impose sur l'anneau d'or et sur la
    # contre-electrode, Neumann sur le substrat isolant et sur l'axe de
    # revolution, potentiel nul au bord exterieur du domaine.
    nr, nz = p['nr'], NZ_SIMU
    r1 = np.linspace(0.0, p['r_max'], nr)
    z1 = np.linspace(0.0, p['gap'], nz)
    dr, dz = r1[1] - r1[0], z1[1] - z1[0]
    R, Z = np.meshgrid(r1, z1, indexing='ij')
    if bavard:
        print(f"  maillage {nr} x {nz}, dr = {dr*1e6:.3f} um, "
              f"dz = {dz*1e6:.3f} um", flush=True)

    m_or = ((R >= p['r_int']) & (R <= p['r_ext'])
            & (Z <= p['h_electrode'] + dz / 2))
    m_ito = Z >= p['gap'] - dz / 2
    m_ext = R >= p['r_max'] - dr / 2
    m_ver = (Z < dz / 2) & (~m_or)
    m_axe = R < dr / 2
    imp = m_or | m_ito | m_ext

    n = nr * nz
    idx = np.arange(n).reshape(nr, nz)
    I, J = np.meshgrid(np.arange(nr), np.arange(nz), indexing='ij')
    L, C, V_ = [], [], []
    b = np.zeros(n)

    def add(li, co, va):
        L.append(np.asarray(li).ravel()); C.append(np.asarray(co).ravel())
        V_.append(np.asarray(va).ravel())

    add(idx[imp], idx[imp], np.ones(int(imp.sum())))
    b[idx[m_or]] = p['amplitude']
    mv = m_ver & (~imp)
    if mv.any():
        add(idx[mv], idx[mv], np.ones(int(mv.sum())))
        add(idx[mv], idx[I[mv], 1], -np.ones(int(mv.sum())))
    ma = m_axe & (~imp) & (~mv)
    if ma.any():
        add(idx[ma], idx[ma], np.ones(int(ma.sum())))
        add(idx[ma], idx[1, J[ma]], -np.ones(int(ma.sum())))
    it = ~(imp | mv | ma)
    it[0, :] = it[-1, :] = False; it[:, 0] = it[:, -1] = False
    nb = int(it.sum())
    if nb:
        ri = R[it]
        add(idx[it], idx[it], np.full(nb, -(2 / dr ** 2 + 2 / dz ** 2)))
        add(idx[it], idx[I[it] + 1, J[it]], 1 / dr ** 2 + 1 / (2 * ri * dr))
        add(idx[it], idx[I[it] - 1, J[it]], 1 / dr ** 2 - 1 / (2 * ri * dr))
        add(idx[it], idx[I[it], J[it] + 1], np.full(nb, 1 / dz ** 2))
        add(idx[it], idx[I[it], J[it] - 1], np.full(nb, 1 / dz ** 2))
    rs = ~(imp | mv | ma | it)
    if rs.any():
        add(idx[rs], idx[rs], np.ones(int(rs.sum())))

    A = coo_matrix((np.concatenate(V_),
                    (np.concatenate(L), np.concatenate(C))),
                   shape=(n, n)).tocsr()
    if bavard:
        print("  resolution du systeme lineaire...", flush=True)
    V = spsolve(A, b).reshape(nr, nz)
    Er, Ez = np.gradient(-V, dr, dz)
    E2 = (Er ** 2 + Ez ** 2) * facteur_forme_onde(p['forme'])
    dE2, _ = np.gradient(E2, dr, dz, edge_order=2)
    return dict(r=r1, z=z1, dr=dr, dz=dz, V=V, E2=E2, dE2_dr=dE2)


def section_b(p, re_k, gamma, r_cible):
    # Chemin descendant : du potentiel impose vers les vitesses predites.
    # Aucun parametre n'est ajuste.
    print("\n" + "=" * 70)
    print("SECTION B : CHEMIN DESCENDANT, DU POTENTIEL VERS LES VITESSES")
    print("=" * 70)
    print(f"  amplitude imposee : {p['amplitude']:.2f} V "
          f"({p['tension_affichee']:.0f} V {p['convention']})")
    print(f"  aucun parametre ajuste\n")

    ch = resoudre_laplace(p)
    j = int(np.argmin(np.abs(ch['z'] - p['z'])))
    r = ch['r']
    e2 = ch['E2'][:, j]
    grad = ch['dE2_dr'][:, j]
    pref = prefacteur_dep(p, re_k)
    v = pref * grad / gamma
    force = pref * grad

    print(f"\n  a z = {ch['z'][j]*1e6:.2f} um")
    print(f"    champ maximal          = {math.sqrt(np.max(e2)):.4e} V/m")
    print(f"    champ au bord (r={p['r_ext']*1e6:.0f} um) = "
          f"{math.sqrt(np.interp(p['r_ext'], r, e2)):.4e} V/m")
    print(f"    amplitude / hauteur    = {p['amplitude']/p['gap']:.4e} V/m")

    # Longueur de decroissance theorique
    m = (r > p['r_ext'] * 1.15) & (r < p['r_ext'] * 1.9) & (e2 > 0)
    L = -1 / np.polyfit(r[m], np.log(e2[m]), 1)[0]
    print(f"\n    longueur de decroissance calculee = {L*1e6:.2f} um")
    print(f"    valeur attendue h/pi              = {p['gap']/math.pi*1e6:.2f} um")

    v_cible = np.interp(r_cible, r, v)
    return dict(champ=ch, j=j, r=r, e2=e2, gradient=grad, force=force,
                v=v, longueur=L, r_cible=r_cible, v_cible=v_cible)


def integrer_trajectoire(r0, grad_fn, pref, gamma, duree, dt=0.01,
                         r_min=5e-6, r_max=480e-6, deplacement_max=1e-6):
    # Integre dr/dt = v(r) avec un pas de temps adaptatif.
    #
    # Pres du bord de l'electrode, la vitesse depasse plusieurs milliers de
    # micrometres par seconde : un pas fixe de dix millisecondes y ferait
    # avancer la particule de plusieurs dizaines de micrometres d'un coup,
    # c'est-a-dire plus que la longueur sur laquelle le champ varie. La
    # trajectoire calculee serait alors une suite de sauts sans rapport avec la
    # solution.
    #
    # Le pas est donc subdivise autant que necessaire pour que le deplacement
    # elementaire reste sous deplacement_max, tout en conservant une sortie
    # echantillonnee regulierement pour l'affichage.
    n = int(duree / dt) + 1
    t = np.linspace(0.0, duree, n)
    r = np.zeros(n)
    r[0] = r0
    for i in range(n - 1):
        courant = r[i]
        restant = dt
        garde = 0
        while restant > 1e-12 and garde < 10000:
            garde += 1
            v = pref * float(grad_fn(courant)) / gamma
            if abs(v) < 1e-15:
                break
            # Sous-pas limite par le deplacement autorise
            sous_pas = min(restant, deplacement_max / abs(v))
            # Schema du point milieu, plus precis qu'Euler a cout comparable
            v_milieu = pref * float(
                grad_fn(min(max(courant + 0.5 * v * sous_pas, r_min), r_max))
            ) / gamma
            courant = min(max(courant + v_milieu * sous_pas, r_min), r_max)
            restant -= sous_pas
        r[i + 1] = courant
    return t, r


def trajectoires_comparees(sync, reco, simu, p, re_k, gamma, instants,
                           duree=None):
    # Trajectoires reellement suivies, puis simulees dans les deux champs.
    #
    # POINT ESSENTIEL : la simulation part de la position occupee A L'INSTANT
    # D'ACTIVATION du champ, et non du premier point de la video. Ces deux
    # instants different de plusieurs secondes, pendant lesquelles la bille est
    # immobile. Confondre les deux decalerait toutes les courbes simulees et
    # les rendrait incomparables aux mesures.
    #
    # C'est pourquoi la fonction recoit le tableau SYNCHRONISE, dont l'axe des
    # temps a deja ete recale sur la rupture de pente, et non les donnees
    # brutes.
    ex = reco['extrapolation']
    if ex is None:
        return None
    pref = prefacteur_dep(p, re_k)

    def grad_eff(r):
        # Derivee analytique du modele exponentiel ajuste
        if r <= p['r_ext']:
            return 0.0
        return -ex['amplitude'] / ex['longueur'] * math.exp(
            -(r - ex['r_bord']) / ex['longueur'])

    ch = simu['champ']
    j = simu['j']

    def grad_lap(r):
        return float(np.interp(r, ch['r'], ch['dE2_dr'][:, j]))

    if duree is None:
        duree = float(sync['temps_s'].max())

    sorties = []
    for pid in sorted(sync['particule'].unique()):
        q = sync[sync['particule'] == pid].sort_values('temps_s')
        t_mes = q['temps_s'].values
        r_mes = q['r_um'].values * 1e-6
        # Position a l'instant d'activation, c'est-a-dire au premier point
        # conserve apres recalage.
        r0 = float(r_mes[0])
        duree_bille = float(t_mes.max())
        t_e, r_e = integrer_trajectoire(r0, grad_eff, pref, gamma, duree_bille)
        t_l, r_l = integrer_trajectoire(r0, grad_lap, pref, gamma, duree_bille)
        acq = (int(q['acquisition'].iloc[0])
               if 'acquisition' in q.columns else 0)
        sorties.append(dict(particule=pid, nom=q['nom'].iloc[0], r0=r0,
                            acquisition=acq, t_mes=t_mes, r_mes=r_mes,
                            t=t_e, r_effectif=r_e, r_laplace=r_l))

    print("\n" + "=" * 70)
    print("TRAJECTOIRES SIMULEES DANS LES DEUX CHAMPS")
    print("=" * 70)
    if isinstance(instants, dict):
        print(f"  origine des temps : activation propre a chaque acquisition")
        for g in sorted(instants):
            print(f"    acquisition {g + 1} : t = {instants[g]:.3f} s")
    else:
        print(f"  origine des temps : t = {instants:.3f} s")
    print(f"  position de depart : celle occupee a cet instant")
    print(f"\n{'bille':>14} {'depart':>9} {'mesure':>10} {'champ reconstruit':>19} "
          f"{'Laplace impose':>16}")
    print("-" * 72)
    for s_ in sorties:
        print(f"{s_['nom']:>14} {s_['r0']*1e6:>8.0f}u {s_['r_mes'][-1]*1e6:>9.0f}u "
              f"{s_['r_effectif'][-1]*1e6:>18.0f}u "
              f"{s_['r_laplace'][-1]*1e6:>15.0f}u")

    moy_e = np.mean([s_['r_effectif'][-1] for s_ in sorties])
    moy_l = np.mean([s_['r_laplace'][-1] for s_ in sorties])
    moy_m = np.mean([s_['r_mes'][-1] for s_ in sorties])
    print("-" * 72)
    print(f"{'moyenne':>14} {'':>9} {moy_m*1e6:>9.0f}u {moy_e*1e6:>18.0f}u "
          f"{moy_l*1e6:>15.0f}u")

    # Ecart quadratique moyen entre mesure et simulation, par champ
    print(f"\n  ECART A LA MESURE, moyenne sur toute la trajectoire")
    for cle, lib in [('r_effectif', 'champ reconstruit'),
                     ('r_laplace', 'Laplace impose')]:
        ecarts = []
        for s_ in sorties:
            r_sim = np.interp(s_['t_mes'], s_['t'], s_[cle])
            ecarts.append(np.sqrt(np.mean((r_sim - s_['r_mes']) ** 2)))
        print(f"    {lib:20s} : {np.mean(ecarts)*1e6:6.1f} um")

    rr = np.linspace(p['r_ext'], 320e-6, 3000)
    e2_eff = modele_decroissance(rr, ex['amplitude'], ex['longueur'],
                                 ex['r_bord'])
    e2_lap = np.interp(rr, ch['r'], ch['E2'][:, j])
    d = np.sign(e2_eff - e2_lap)
    croisement = rr[:-1][np.diff(d) != 0]
    print(f"\n  LECTURE")
    if len(croisement):
        rc = croisement[0]
        print(f"    les deux champs se croisent a r = {rc*1e6:.0f} um.")
        print(f"    en deca, le champ impose est le plus fort ; au dela, c'est")
        print(f"    le champ reconstruit, qui decroit "
              f"{ex['longueur']/(p['gap']/math.pi):.1f} fois plus lentement.")
    print("=" * 70)
    return sorties


# =============================================================================
# SECTION C : COMPARAISON
# =============================================================================

def section_c(profil, reco, simu, p):
    # Confrontation des deux chemins.
    #
    # Ils partent des memes particules et de la meme geometrie : leur ecart
    # mesure donc ce que le modele dielectrophoretique ne decrit pas, et sa
    # dependance radiale en indique la nature. Un ecart constant signalerait
    # une simple erreur d'amplitude, corrigeable par la tension ; un ecart qui
    # croit avec le rayon signale au contraire un second mecanisme, dont la
    # portee depasse celle du gradient de champ.
    print("\n" + "=" * 70)
    print("SECTION C : CONFRONTATION DES DEUX CHEMINS")
    print("=" * 70)

    r = profil['r']
    v_mes = profil['v']
    v_sim = simu['v_cible']
    ok = v_sim > 0
    rapport = np.full_like(v_mes, np.nan)
    rapport[ok] = v_mes[ok] / v_sim[ok]

    print(f"\n{'r (um)':>9} {'v mesuree':>12} {'v simulee':>12} {'rapport':>10}")
    print("-" * 46)
    pas = max(1, len(r) // 12)
    for i in range(0, len(r), pas):
        print(f"{r[i]*1e6:>9.1f} {v_mes[i]*1e6:>11.2f}u "
              f"{v_sim[i]*1e6:>11.3f}u {rapport[i]:>10.2f}")

    # Le rapport est-il constant ?
    fini = np.isfinite(rapport)
    r_min_rap = float(np.nanmin(rapport[fini]))
    r_max_rap = float(np.nanmax(rapport[fini]))
    print(f"\n  rapport de {r_min_rap:.2f} a {r_max_rap:.1f}")

    # Potentiel apparent selon la plage retenue
    print(f"\n  POTENTIEL APPARENT SELON LA PLAGE RETENUE")
    print(f"  {'plage (um)':>20} {'n':>5} {'potentiel (V)':>15}")
    print("  " + "-" * 42)
    bornes, potentiels = [], []
    for frac in [0.10, 0.20, 0.35, 0.5, 0.75, 1.0]:
        n = max(2, int(frac * len(r)))
        m = np.zeros(len(r), bool); m[:n] = True; m &= ok
        if int(m.sum()) < 2:
            continue
        fac = float(np.sum(v_mes[m] * v_sim[m]) / np.sum(v_sim[m] ** 2))
        V = p['amplitude'] * math.sqrt(max(fac, 0.0))
        bornes.append(r[n - 1]); potentiels.append(V)
        print(f"  {f'{r[0]*1e6:.0f} a {r[n-1]*1e6:.0f}':>20} "
              f"{int(m.sum()):>5} {V:>15.2f}")

    # Diagnostic
    print(f"\n  LECTURE")
    if r_max_rap / max(r_min_rap, 1e-12) < 2.0:
        print("    le rapport est sensiblement constant : l'ecart entre les deux")
        print("    chemins se ramene a une amplitude, donc a une tension")
        print("    effective inferieure a celle appliquee.")
    else:
        print("    le rapport n'est pas constant : il croit avec le rayon.")
        print("    aucune tension ne peut donc rendre compte de l'ecart.")
        i0 = int(np.nanargmin(np.abs(rapport - 1.0)))
        print(f"\n    les deux chemins coincident a r = {r[i0]*1e6:.0f} um "
              f"(rapport {rapport[i0]:.2f}),")
        print(f"    puis divergent : le modele y decroit en exp(-r/"
              f"{simu['longueur']*1e6:.1f} um) tandis que la")
        print(f"    mesure decroit bien plus lentement.")
        print(f"\n    interpretation : pres du bord, la dielectrophorese domine")
        print(f"    et le modele suffit. plus loin, le gradient de champ devient")
        print(f"    negligeable et un second mecanisme prend le relais, dont la")
        print(f"    portee depasse largement celle du champ. les candidats sont")
        print(f"    un ecoulement electrothermique, entretenu par l'echauffement")
        print(f"    du milieu conducteur, ou une electro-osmose alternative.")
        print(f"\n    CONSEQUENCE : le champ reconstruit en section A n'est")
        print(f"    utilisable que sur la portion ou les deux chemins")
        print(f"    concordent. au dela, il attribue a la dielectrophorese un")
        print(f"    mouvement qui ne lui revient pas.")
    print("=" * 70)

    return dict(r=r, v_mesuree=v_mes, v_simulee=v_sim, rapport=rapport,
                bornes=np.array(bornes), potentiels=np.array(potentiels))


# =============================================================================
# TRACES ELEMENTAIRES
# =============================================================================
# Chaque fonction dessine une vignette dans un axe fourni. Les panneaux les
# assemblent, et les figures individuelles les reprennent telles quelles : le
# trace n'est donc ecrit qu'une fois.

def _couleurs(n):
    return plt.cm.tab10(np.linspace(0, 1, max(n, 1)))


def _sous_ech(p, cible=400):
    return p.iloc[::max(1, len(p) // cible)]


def _anneau(ax, p, etiquette=True):
    ax.axvspan(p['r_int'] * 1e6, p['r_ext'] * 1e6, alpha=0.22, color=COUL_OR,
               label="anneau d'or" if etiquette else None)


def t_trajectoires(ax, brutes, p):
    coul = _couleurs(brutes['particule'].nunique())
    for pid, c in zip(sorted(brutes['particule'].unique()), coul):
        q = _sous_ech(brutes[brutes['particule'] == pid].sort_values('temps_s'))
        th = (np.deg2rad(q['theta_deg'].values) if q['theta_deg'].notna().any()
              else np.zeros(len(q)))
        ax.plot(th, q['r_um'].values, '-', color=c, lw=1.4, alpha=0.85)
        ax.scatter([th[-1]], [q['r_um'].values[-1]], color=c, s=60,
                   edgecolors='black', linewidths=1.3, zorder=5)
    tt = np.linspace(0, 2 * np.pi, 200)
    ax.fill_between(tt, p['r_int'] * 1e6, p['r_ext'] * 1e6, alpha=0.3,
                    color=COUL_OR)
    ax.set_title("Trajectoires des billes", fontsize=11, fontweight='bold',
                 pad=16)
    ax.grid(alpha=0.3)


def t_position(ax, sync, instants, p):
    # Position radiale au cours du temps, sur l'axe RECALE.
    #
    # Chaque bille ayant ete recalee sur son propre instant d'activation, le
    # mouvement demarre au meme instant pour toutes : c'est ce qui rend les
    # trajectoires directement comparables. La phase anterieure, en temps
    # negatif, montre l'etat de repos et vaut controle du recalage.
    coul = _couleurs(sync['particule'].nunique())
    for pid, c in zip(sorted(sync['particule'].unique()), coul):
        q = sync[sync['particule'] == pid].sort_values('temps_s')
        avant = q['temps_s'] < 0
        if np.any(avant):
            qa = _sous_ech(q[avant])
            ax.plot(qa['temps_s'], qa['r_um'], '-', color=c, lw=1.2,
                    alpha=0.35)
        qb = _sous_ech(q[~avant])
        ax.plot(qb['temps_s'], qb['r_um'], '-', color=c, lw=1.8, alpha=0.9)

    ax.axvline(0, color='red', ls='--', lw=2,
               label="activation, origine commune")
    xg = ax.get_xlim()[0]
    if xg < 0:
        ax.axvspan(xg, 0, color='gray', alpha=0.10, lw=0)
    ax.axhspan(p['r_int'] * 1e6, p['r_ext'] * 1e6, alpha=0.22, color=COUL_OR)
    ax.set_xlabel("Temps depuis l'activation (s)")
    ax.set_ylabel("Position radiale (um)")
    n_acq = len(instants) if isinstance(instants, dict) else 1
    ax.set_title("Position radiale, origine recalee sur l'activation"
                 + (f"\n{n_acq} acquisitions" if n_acq > 1 else ""),
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)


def t_vitesse_temps(ax, don):
    # La phase anterieure a l'activation apparait en temps negatif : la vitesse
    # y doit etre nulle, ce qui vaut controle du recalage.
    coul = _couleurs(don['particule'].nunique())
    for pid, c in zip(sorted(don['particule'].unique()), coul):
        q = _sous_ech(don[don['particule'] == pid].sort_values('temps_s'))
        ax.plot(q['temps_s'], q['v_r'] * 1e6, '-', color=c, lw=1.4, alpha=0.8)
    ax.axvline(0, color='red', ls='--', lw=1.8, label="activation")
    ax.axhline(0, color='k', ls=':', alpha=0.6)
    xg = ax.get_xlim()[0]
    if xg < 0:
        ax.axvspan(xg, 0, color='gray', alpha=0.10, lw=0)
    ax.set_xlabel("Temps recale sur l'activation (s)"); ax.set_ylabel("Vitesse radiale (um/s)")
    ax.set_title("Vitesse apres recalage", fontsize=11, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)


def t_vitesse_par_bille(ax, don):
    coul = _couleurs(don['particule'].nunique())
    for pid, c in zip(sorted(don['particule'].unique()), coul):
        q = _sous_ech(don[don['particule'] == pid], 250)
        ax.scatter(q['r_um'], q['v_r'] * 1e6, s=9, alpha=0.35, color=c,
                   label=q['nom'].iloc[0] if len(q) else None)
    ax.axhline(0, color='k', ls=':', alpha=0.6)
    ax.set_xlabel("Position radiale (um)"); ax.set_ylabel("Vitesse (um/s)")
    ax.set_title("Vitesse de chaque bille", fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)


def t_profil_moyen(ax, comp, ret, p, zoom=False):
    exc = comp['r'] < p['r_min']
    if np.any(exc):
        ax.errorbar(comp['r'][exc] * 1e6, comp['v'][exc] * 1e6,
                    yerr=comp['ecart'][exc] * 1e6, fmt='o', color=COUL_GRIS,
                    ms=5, alpha=0.55, capsize=2, label="exclu du calcul")
    ax.errorbar(ret['r'] * 1e6, ret['v'] * 1e6, yerr=ret['ecart'] * 1e6,
                fmt='o-', color=COUL_V, ms=6, lw=2, capsize=3,
                markeredgecolor='k', markeredgewidth=0.8, label="profil moyen")
    _anneau(ax, p)
    if zoom:
        # Agrandissement sur la queue du profil, ou les vitesses sont faibles
        # et invisibles a l'echelle du maximum.
        m = ret['r'] > np.median(ret['r'])
        if int(np.sum(m)) > 2:
            ax.set_ylim(0, float(np.max(ret['v'][m]) * 1e6) * 1.6)
            ax.set_xlim(float(np.median(ret['r']) * 1e6) * 0.95,
                        float(ret['r'].max() * 1e6) * 1.02)
    ax.set_xlabel("Position radiale (um)")
    ax.set_ylabel("Vitesse moyenne (um/s)")
    ax.set_title(f"Profil moyen{', agrandissement sur la queue' if zoom else ''}",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)


def t_etape_vitesse(ax, ret):
    ax.errorbar(ret['r'] * 1e6, ret['v'] * 1e6, yerr=ret['ecart'] * 1e6,
                fmt='o-', color=COUL_V, ms=6, lw=2, capsize=3,
                markeredgecolor='k', markeredgewidth=0.7)
    ax.set_xlabel("Position radiale (um)")
    ax.set_ylabel("Vitesse moyenne (um/s)")
    ax.set_title("1. Vitesse mesuree", fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)


def t_etape_force(ax, ret, reco):
    ax.errorbar(ret['r'] * 1e6, reco['force'] * 1e12,
                yerr=reco['force_ecart'] * 1e12, fmt='o-', color=COUL_F, ms=6,
                lw=2, capsize=3, markeredgecolor='k', markeredgewidth=0.7)
    ax.set_xlabel("Position radiale (um)")
    ax.set_ylabel("Force dielectrophoretique (pN)")
    ax.set_title("2. Force, F = gamma v", fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)


def t_etape_gradient(ax, ret, reco):
    ax.plot(ret['r'] * 1e6, reco['gradient'], 'o-', color=COUL_GRAD, ms=6,
            lw=2, markeredgecolor='k', markeredgewidth=0.7)
    ax.axhline(0, color='k', lw=0.9)
    ax.set_xlabel("Position radiale (um)")
    ax.set_ylabel("Gradient du carre du champ (V2/m3)")
    ax.set_title("3. Gradient, force divisee par le prefacteur", fontsize=11,
                 fontweight='bold')
    ax.grid(alpha=0.3)


def t_etape_champ(ax, reco):
    ax.plot(reco['r_e2'] * 1e6, reco['e2'], 'o-', color=COUL_E, ms=6, lw=2,
            markeredgecolor='k', markeredgewidth=0.7)
    ax.set_xlabel("Position radiale (um)")
    ax.set_ylabel("Carre du champ (V2/m2)")
    ax.set_title("4. Carre du champ, gradient integre", fontsize=11,
                 fontweight='bold')
    ax.grid(alpha=0.3, which='both')


def t_champ_extrapole(ax, reco, p, zoom=False):
    ex = reco['extrapolation']
    r_ext = np.linspace(p['r_ext'], reco['r_e2'].max() * 1.03, 400)
    if ex:
        ax.plot(r_ext * 1e6,
                modele_decroissance(r_ext, ex['amplitude'], ex['longueur'],
                                    ex['r_bord']),
                '-', color=COUL_FIT, lw=2.2,
                label="extrapolation, maximum au bord")
    ax.plot(reco['r_e2'] * 1e6, np.maximum(reco['e2'], 1e-30), 'o',
            color=COUL_E, ms=6, markeredgecolor='k', markeredgewidth=0.7,
            label="reconstruit")
    _anneau(ax, p)
    if zoom:
        m = reco['r_e2'] > np.median(reco['r_e2'])
        if int(np.sum(m)) > 2:
            ax.set_ylim(0, float(np.max(reco['e2'][m])) * 1.6)
            ax.set_xlim(float(np.median(reco['r_e2']) * 1e6) * 0.95,
                        reco['r_e2'].max() * 1e6 * 1.02)
    else:
        ax.set_xlim(p['r_ext'] * 1e6 * 0.9, reco['r_e2'].max() * 1e6 * 1.03)
    ax.set_xlabel("Position radiale (um)")
    ax.set_ylabel("Carre du champ (V2/m2)")
    ax.set_title(f"Champ reconstruit"
                 f"{', agrandissement sur la queue' if zoom else ''}",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)


def t_champ_electrique(ax, reco, p):
    ex = reco['extrapolation']; pot = reco['potentiel']
    r_ext = np.linspace(p['r_ext'], reco['r_e2'].max() * 1.03, 400)
    if ex:
        ax.plot(r_ext * 1e6,
                np.sqrt(modele_decroissance(r_ext, ex['amplitude'],
                                            ex['longueur'], ex['r_bord'])),
                '-', color=COUL_FIT, lw=2.2, label="extrapolation")
    ax.plot(reco['r_e2'] * 1e6, reco['e'], 'o', color=COUL_E, ms=6,
            markeredgecolor='k', markeredgewidth=0.7, label="reconstruit")
    ax.axhline(p['amplitude'] / p['gap'], color='k', ls=':', lw=1.6,
               label=f"amplitude / hauteur = {p['amplitude']/p['gap']:.2e} V/m")
    _anneau(ax, p)
    ax.set_xlim(p['r_ext'] * 1e6 * 0.9, reco['r_e2'].max() * 1e6 * 1.03)
    ax.set_xlabel("Position radiale (um)")
    ax.set_ylabel("Champ electrique (V/m)")
    ax.set_title("Champ electrique et potentiel", fontsize=11,
                 fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)


def t_recapitulatif(ax, reco, p):
    ax.axis('off')
    ex = reco['extrapolation']; pot = reco['potentiel']
    li = []
    if ex:
        li += [["Longueur de decroissance", f"{ex['longueur']*1e6:.2f} um"],
               ["Cavite que cela implique",
                f"{ex['longueur']*math.pi*1e6:.0f} um"],
               ["Cavite mesuree", f"{p['gap']*1e6:.1f} um"],
               ["Carre du champ au bord", f"{ex['amplitude']:.3e} V2/m2"],
               ["Qualite de l'extrapolation", f"{ex['r2']:.4f}"]]
    if pot:
        li += [["Champ au bord", f"{pot['e_bord']:.3e} V/m"],
               ["Potentiel estime", f"{pot['v_estime']:.3f} V"],
               ["Amplitude imposee", f"{pot['v_amplitude']:.2f} V"],
               ["Rapport", f"{100*pot['rapport']:.0f} %"]]
    t = ax.table(cellText=li, colLabels=["Grandeur", "Valeur"], loc='center',
                 cellLoc='left')
    t.auto_set_font_size(False); t.set_fontsize(9.5); t.scale(1, 1.7)
    ax.set_title("Recapitulatif de la reconstruction", fontsize=11,
                 fontweight='bold', pad=20)


def t_carte_potentiel(ax, simu, p, figure=None):
    # Potentiel dans le plan meridien.
    #
    # La contre-electrode d'ITO couvre TOUTE la face superieure et y impose un
    # potentiel nul, quel que soit le rayon. L'anneau d'or n'occupe en revanche
    # qu'une couronne de la face inferieure, le reste etant du verre isolant.
    # C'est cette dissymetrie qui rend le champ non uniforme : au dessus du
    # metal la cavite se comporte comme un condensateur plan, ailleurs le
    # potentiel doit s'etaler lateralement.
    ch = simu['champ']
    cf = ax.contourf(ch['r'] * 1e6, ch['z'] * 1e6, ch['V'].T, levels=60,
                     cmap="plasma", vmin=0, vmax=p['amplitude'])
    ax.contour(ch['r'] * 1e6, ch['z'] * 1e6, ch['V'].T, levels=12, colors='k',
               linewidths=0.4, alpha=0.45)
    if figure is not None:
        figure.colorbar(cf, ax=ax, label="Potentiel (V)")
    ax.axhline(p['gap'] * 1e6, color='cyan', lw=3, zorder=6,
               label="ITO, conducteur sur toute la surface")
    ax.plot([p['r_int'] * 1e6, p['r_ext'] * 1e6], [0, 0], color=COUL_OR, lw=5,
            zorder=6, label="anneau d'or")
    ax.axhline(p['z'] * 1e6, color='white', lw=1.4, ls='--',
               label=f"plan des billes, z = {p['z']*1e6:.1f} um")
    ax.set_xlim(0, R_AFFICHAGE_UM)
    ax.set_ylim(0, p['gap'] * 1e6)
    ax.set_xlabel("r (um)"); ax.set_ylabel("z (um)")
    ax.set_title("Potentiel, coupe meridienne", fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right', ncol=1)


def t_carte_champ(ax, simu, p, figure=None):
    # Norme du champ dans le plan meridien, echelle lineaire.
    ch = simu['champ']
    E = np.sqrt(ch['E2'])
    haut = float(np.percentile(E, 99.5))
    cf = ax.contourf(ch['r'] * 1e6, ch['z'] * 1e6, E.T,
                     levels=np.linspace(0, haut, 60), cmap="inferno",
                     extend='max')
    if figure is not None:
        figure.colorbar(cf, ax=ax, label="Champ (V/m)")
    ax.axhline(p['gap'] * 1e6, color='cyan', lw=3, zorder=6)
    ax.plot([p['r_int'] * 1e6, p['r_ext'] * 1e6], [0, 0], color=COUL_OR, lw=5,
            zorder=6)
    ax.axhline(p['z'] * 1e6, color='white', lw=1.4, ls='--')
    ax.set_xlim(0, R_AFFICHAGE_UM)
    ax.set_ylim(0, p['gap'] * 1e6)
    ax.set_xlabel("r (um)"); ax.set_ylabel("z (um)")
    ax.set_title("Norme du champ, coupe meridienne", fontsize=11,
                 fontweight='bold')


def t_carte_dessus(ax, simu, p, figure=None):
    # Champ dans le plan d'observation, reconstruit par symetrie de revolution.
    ch = simu['champ']
    j = simu['j']
    n = 400
    x = np.linspace(-R_AFFICHAGE_UM * 1e-6, R_AFFICHAGE_UM * 1e-6, n)
    X, Y = np.meshgrid(x, x)
    RR = np.sqrt(X ** 2 + Y ** 2)
    E_prof = np.sqrt(ch['E2'][:, j])
    Zc = np.interp(RR.ravel(), ch['r'], E_prof).reshape(RR.shape)
    haut = float(np.percentile(Zc, 99.5))
    cf = ax.contourf(X * 1e6, Y * 1e6, Zc, levels=np.linspace(0, haut, 60),
                     cmap="inferno", extend='max')
    if figure is not None:
        figure.colorbar(cf, ax=ax, label="Champ (V/m)")
    th = np.linspace(0, 2 * np.pi, 400)
    for rr, lw in [(p['r_int'], 1.4), (p['r_ext'], 1.4)]:
        ax.plot(rr * 1e6 * np.cos(th), rr * 1e6 * np.sin(th), color=COUL_OR,
                lw=lw)
    ax.set_aspect('equal')
    ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
    ax.set_title(f"Champ vu de dessus, z = {p['z']*1e6:.1f} um", fontsize=11,
                 fontweight='bold')


def t_profil_potentiel(ax, simu, p):
    # Potentiel le long du rayon, a plusieurs hauteurs.
    #
    # Au dessus du metal, le potentiel decroit lineairement avec z, comme dans
    # un condensateur plan. Au centre et au dela du bord, il s'ecarte de cette
    # loi : c'est la signature du champ non uniforme.
    ch = simu['champ']
    for z_um, coul, style in [(2, COUL_GRAD, '--'), (5, COUL_E, '-'),
                              (10, COUL_F, '--'), (18, COUL_SIMU, ':')]:
        if z_um * 1e-6 >= p['gap']:
            continue
        j = int(np.argmin(np.abs(ch['z'] - z_um * 1e-6)))
        ax.plot(ch['r'] * 1e6, ch['V'][:, j], style, color=coul, lw=2,
                label=f"z = {ch['z'][j]*1e6:.1f} um")
    _anneau(ax, p)
    ax.set_xlim(0, R_AFFICHAGE_UM)
    ax.set_xlabel("Position radiale (um)")
    ax.set_ylabel("Potentiel (V)")
    ax.set_title("Potentiel le long du rayon", fontsize=11, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)


def t_profil_simule(ax, simu, p):
    # Carre du champ le long du rayon, echelle lineaire.
    ax.plot(simu['r'] * 1e6, simu['e2'], '-', color=COUL_SIMU, lw=2.4,
            label="calcul de Laplace")
    _anneau(ax, p)
    ax.set_xlim(0, R_AFFICHAGE_UM)
    ax.set_xlabel("Position radiale (um)")
    ax.set_ylabel("Carre du champ (V2/m2)")
    ax.set_title(f"Carre du champ, decroissance sur "
                 f"{simu['longueur']*1e6:.1f} um", fontsize=11,
                 fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)


def t_gradient_simule(ax, simu, p):
    # Gradient radial, la grandeur qui porte la force.
    #
    # Il s'annule au dessus du metal, ou le champ est uniforme, et culmine de
    # part et d'autre des deux bords. Son signe indique le sens de la force
    # pour une particule en dielectrophorese positive ; il s'inverse pour une
    # dielectrophorese negative.
    ax.plot(simu['r'] * 1e6, simu['gradient'], '-', color=COUL_GRAD, lw=2.4)
    ax.axhline(0, color='k', lw=0.9)
    _anneau(ax, p)
    ax.set_xlim(0, R_AFFICHAGE_UM)
    ax.set_xlabel("Position radiale (um)")
    ax.set_ylabel("Gradient du carre du champ (V2/m3)")
    ax.set_title("Gradient radial, moteur de la force", fontsize=11,
                 fontweight='bold')
    ax.grid(alpha=0.3)


def t_vitesse_simulee(ax, simu, p, profil=None):
    # Vitesse predite par le modele, sans aucun parametre ajuste.
    ax.plot(simu['r'] * 1e6, simu['v'] * 1e6, '-', color=COUL_SIMU, lw=2.4,
            label=f"predite a {p['amplitude']:.1f} V")
    if profil is not None:
        ax.errorbar(profil['r'] * 1e6, profil['v'] * 1e6,
                    yerr=profil['ecart'] * 1e6, fmt='o', color=COUL_V, ms=5,
                    capsize=2, label="mesuree")
    ax.axhline(0, color='k', lw=0.9)
    _anneau(ax, p)
    # Le pic aux aretes depasse de plusieurs ordres de grandeur les vitesses
    # mesurees, qui sont toutes hors du metal. L'axe est donc borne sur la
    # plage utile, et le depassement annonce.
    if profil is not None:
        haut = float(np.max(profil['v'])) * 1e6 * 1.25
        ax.set_xlim(p['r_ext'] * 1e6 * 0.85, R_AFFICHAGE_UM)
        ax.set_ylim(-0.05 * haut, haut)
        pic = float(np.max(simu['v'])) * 1e6
        if pic > haut:
            ax.text(0.97, 0.94,
                    f"le pic aux aretes atteint {pic:.0f} um/s,\n"
                    f"hors du cadre et hors de la zone mesuree",
                    transform=ax.transAxes, ha='right', va='top', fontsize=8.5,
                    color=COUL_SIMU,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    else:
        ax.set_xlim(0, R_AFFICHAGE_UM)
    ax.set_xlabel("Position radiale (um)")
    ax.set_ylabel("Vitesse radiale (um/s)")
    ax.set_title("Vitesse predite et vitesse mesuree", fontsize=11,
                 fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)


def _tracer_comparaison(ax, trajs, p, cle, libelle, couleur_sim):
    # Mesure et simulation superposees, une couleur par bille.
    #
    # Le trait plein porte la mesure, le pointille de la MEME couleur la
    # simulation de la meme bille, partie de la meme position au meme instant.
    # Un seul champ est represente par figure : superposer les deux rendrait
    # illisible l'ecart de chacun a la mesure, qui est precisement ce que l'on
    # cherche a juger.
    if trajs is None:
        ax.axis('off')
        ax.text(0.5, 0.5, "trajectoires non disponibles", ha='center',
                va='center', color=COUL_GRIS, transform=ax.transAxes)
        return
    coul = _couleurs(len(trajs))
    ecarts = []
    for s_, c in zip(trajs, coul):
        # La mesure est en trait PLEIN et FIN, la simulation en POINTILLE plus
        # marque : l'oeil distingue ainsi immediatement ce qui est mesure de ce
        # qui est calcule, sans avoir a consulter la legende.
        #
        # La portion anterieure a l'activation, en temps negatif, est tracee
        # plus discretement : elle ne participe a aucun calcul, et sert
        # uniquement a verifier que la rupture de pente tombe bien a zero.
        avant = s_['t_mes'] < 0
        if np.any(avant):
            ax.plot(s_['t_mes'][avant], s_['r_mes'][avant] * 1e6, '-',
                    color=c, lw=1.0, alpha=0.30, zorder=1)
        apres = s_['t_mes'] >= 0
        ax.plot(s_['t_mes'][apres], s_['r_mes'][apres] * 1e6, '-', color=c,
                lw=1.3, alpha=0.95, zorder=4)
        ax.plot(s_['t'], s_[cle] * 1e6, '--', color=c, lw=2.2, alpha=0.85,
                dashes=(5, 3), zorder=3)
        ax.plot([0], [s_['r0'] * 1e6], 'o', color=c, ms=7,
                markeredgecolor='k', markeredgewidth=0.9, zorder=6)
        # L'ecart n'est evalue que sur la phase sous champ.
        if np.any(apres):
            r_sim = np.interp(s_['t_mes'][apres], s_['t'], s_[cle])
            ecarts.append(np.sqrt(np.mean(
                (r_sim - s_['r_mes'][apres]) ** 2)))

    ax.plot([], [], '-', color='k', lw=1.3, label="mesure")
    ax.plot([], [], '--', color='k', lw=2.2, dashes=(5, 3),
            label=f"simule, {libelle}")
    ax.plot([], [], 'o', color='w', ms=7, markeredgecolor='k',
            label="position a l'activation")
    ax.axhspan(p['r_int'] * 1e6, p['r_ext'] * 1e6, alpha=0.22, color=COUL_OR)
    ax.axvline(0, color='red', ls='--', lw=1.8, alpha=0.85,
               label="activation du champ")
    # La zone de temps negatif est grisee : rien n'y est calcule.
    xg = ax.get_xlim()[0]
    if xg < 0:
        ax.axvspan(xg, 0, color='gray', alpha=0.10, lw=0)
    ax.set_xlabel("Temps depuis l'activation du champ (s)")
    ax.set_ylabel("Position radiale (um)")
    n_acq = len(set(s_.get('acquisition', 0) for s_ in trajs))
    mention = (f", {n_acq} acquisitions recalees" if n_acq > 1 else "")
    ax.set_title(f"Mesure et {libelle}{mention}\n"
                 f"ecart quadratique moyen {np.mean(ecarts)*1e6:.1f} um",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)


def t_comparaison_reconstruit(ax, trajs, p):
    # Position mesuree confrontee a la simulation dans le champ RECONSTRUIT.
    _tracer_comparaison(ax, trajs, p, 'r_effectif', "champ reconstruit",
                        COUL_E)


def t_comparaison_laplace(ax, trajs, p):
    # Position mesuree confrontee a la simulation dans le champ de LAPLACE,
    # a la tension reellement imposee et sans aucun parametre ajuste.
    _tracer_comparaison(ax, trajs, p, 'r_laplace',
                        "champ de Laplace impose", COUL_SIMU)


def t_arrivees(ax, trajs, p):
    # Position finale atteinte, selon le champ employe.
    if trajs is None:
        ax.axis('off'); return
    noms = [s_['nom'] for s_ in trajs]
    x = np.arange(len(trajs))
    r0 = np.array([s_['r0'] for s_ in trajs]) * 1e6
    rm = np.array([s_['r_mes'][-1] for s_ in trajs]) * 1e6
    re = np.array([s_['r_effectif'][-1] for s_ in trajs]) * 1e6
    rl = np.array([s_['r_laplace'][-1] for s_ in trajs]) * 1e6
    ax.bar(x - 0.26, r0, 0.24, color=COUL_GRIS, label="depart")
    ax.bar(x, rm, 0.24, color=COUL_V, label="arrivee mesuree")
    ax.bar(x + 0.26, rl, 0.24, color=COUL_SIMU,
           label=f"arrivee predite a {p['amplitude']:.1f} V")
    ax.plot(x, re, 'o', color=COUL_E, ms=8, markeredgecolor='k',
            markeredgewidth=0.9, label="arrivee, champ reconstruit")
    ax.set_xticks(x); ax.set_xticklabels(noms, rotation=25, ha='right')
    ax.set_ylabel("Position radiale (um)")
    ax.set_title("Position finale selon le champ employe", fontsize=11,
                 fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')


def t_deux_champs(ax, reco, simu, p):
    # Les deux champs superposes sur toute la plage, avec leur croisement.
    ex = reco['extrapolation']
    ch = simu['champ']; j = simu['j']
    rr = np.linspace(p['r_ext'], reco['r_e2'].max() * 1.05, 600)
    if ex:
        e2e = modele_decroissance(rr, ex['amplitude'], ex['longueur'],
                                  ex['r_bord'])
        ax.plot(rr * 1e6, e2e, '-', color=COUL_E, lw=2.4,
                label=f"reconstruit, decroissance {ex['longueur']*1e6:.1f} um")
    e2l = np.interp(rr, ch['r'], ch['E2'][:, j])
    ax.plot(rr * 1e6, e2l, '--', color=COUL_SIMU, lw=2.4,
            label=f"impose a {p['amplitude']:.1f} V, decroissance "
                  f"{simu['longueur']*1e6:.1f} um")
    if ex:
        d = np.sign(e2e - e2l)
        cr = rr[:-1][np.diff(d) != 0]
        if len(cr):
            ax.axvline(cr[0] * 1e6, color='k', ls=':', lw=1.6,
                       label=f"croisement, {cr[0]*1e6:.0f} um")
    ax.plot(reco['r_e2'] * 1e6, reco['e2'], 'o', color=COUL_E, ms=5,
            markeredgecolor='k', markeredgewidth=0.6, label="points mesures")
    _anneau(ax, p)
    ax.set_xlim(p['r_ext'] * 1e6 * 0.97, reco['r_e2'].max() * 1e6 * 1.03)
    ax.set_ylim(0, max(float(np.max(e2l[:50])), float(np.max(reco['e2']))) * 1.1)
    ax.set_xlabel("Position radiale (um)")
    ax.set_ylabel("Carre du champ (V2/m2)")
    ax.set_title("Champ reconstruit et champ impose", fontsize=11,
                 fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)


def t_comparaison_vitesse(ax, comp, p):
    ax.plot(comp['r'] * 1e6, comp['v_mesuree'] * 1e6, 'o-', color=COUL_V,
            ms=6, lw=2, markeredgecolor='k', markeredgewidth=0.7,
            label="mesuree, chemin ascendant")
    ax.plot(comp['r'] * 1e6, comp['v_simulee'] * 1e6, 's--', color=COUL_SIMU,
            ms=5, lw=2,
            label=f"simulee a {p['amplitude']:.1f} V, chemin descendant")
    ax.set_xlabel("Position radiale (um)")
    ax.set_ylabel("Vitesse radiale (um/s)")
    ax.set_title("Les deux chemins confrontes", fontsize=11, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')


def t_comparaison_rapport(ax, comp, plafond=25.0):
    # Rapport entre les deux chemins, en echelle lineaire.
    #
    # Le rapport pouvant atteindre plusieurs ordres de grandeur, l'axe est
    # borne pour rester lisible. Les points qui sortent du cadre sont comptes
    # et annonces : ils font partie du resultat, et c'est meme eux qui le
    # portent.
    ax.plot(comp['r'] * 1e6, comp['rapport'], 'o-', color=COUL_GRAD, ms=6,
            lw=2, markeredgecolor='k', markeredgewidth=0.7)
    ax.axhline(1.0, color='k', ls='--', lw=1.8, label="accord parfait")
    ax.axhspan(0.5, 2.0, color='green', alpha=0.12, lw=0,
               label="accord a un facteur deux")
    fini = np.isfinite(comp['rapport'])
    hors = int(np.sum(comp['rapport'][fini] > plafond))
    ax.set_ylim(0, plafond)
    if hors:
        ax.text(0.97, 0.94,
                f"{hors} points au dessus de {plafond:.0f},\n"
                f"jusqu'a {np.nanmax(comp['rapport']):.0f}",
                transform=ax.transAxes, ha='right', va='top', fontsize=8.5,
                color=COUL_GRAD,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    ax.set_xlabel("Position radiale (um)")
    ax.set_ylabel("Vitesse mesuree / vitesse simulee")
    ax.set_title("Rapport des deux chemins\n"
                 "un rapport constant signalerait une simple erreur "
                 "d'amplitude", fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left'); ax.grid(alpha=0.3)


def t_potentiel_apparent(ax, comp, p):
    ax.plot(comp['bornes'] * 1e6, comp['potentiels'], 'o-', color=COUL_E, ms=7,
            lw=2, markeredgecolor='k', markeredgewidth=0.8)
    ax.axhline(p['amplitude'], color='k', ls='--', lw=1.8,
               label=f"amplitude imposee, {p['amplitude']:.1f} V")
    ax.set_xlabel("Rayon maximal inclus dans l'ajustement (um)")
    ax.set_ylabel("Potentiel apparent (V)")
    ax.set_title("Potentiel apparent selon la plage retenue\n"
                 "sa derive signale un mecanisme non decrit par le modele",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)


def t_comparaison_champ(ax, reco, simu, p):
    ax.plot(reco['r_e2'] * 1e6, reco['e2'], 'o', color=COUL_E, ms=6,
            markeredgecolor='k', markeredgewidth=0.7,
            label="reconstruit par les billes")
    ax.plot(simu['r'] * 1e6, simu['e2'], '-', color=COUL_SIMU, lw=2.2,
            label=f"calcule a {p['amplitude']:.1f} V")
    _anneau(ax, p)
    ax.set_xlim(p['r_ext'] * 1e6 * 0.9, reco['r_e2'].max() * 1e6 * 1.03)
    haut = max(float(np.max(reco['e2'])),
               float(np.max(simu['e2'][simu['r'] >= p['r_ext']])))
    ax.set_ylim(0, haut * 1.1)
    ax.set_xlabel("Position radiale (um)")
    ax.set_ylabel("Carre du champ (V2/m2)")
    ax.set_title("Champ reconstruit et champ calcule", fontsize=11,
                 fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)


# =============================================================================
# ASSEMBLAGE DES PANNEAUX ET DES VIGNETTES
# =============================================================================

def figure_seule(trace, largeur=9.0, hauteur=5.5, polaire=False):
    # Cree une figure ne contenant qu'une vignette.
    figure = plt.figure(figsize=(largeur, hauteur))
    ax = (figure.add_subplot(111, projection='polar') if polaire
          else figure.add_subplot(111))
    trace(ax, figure)
    figure.tight_layout()
    return figure


def enregistrer(figure, dossier, nom):
    chemin = os.path.join(dossier, f"{nom}.svg")
    figure.savefig(chemin, format='svg', bbox_inches='tight', facecolor='white')
    print(f"[ENREGISTRE] {chemin}", flush=True)
    return chemin


def produire(dossier, numero, titre_panneau, vignettes, disposition,
             taille, suptitre=None):
    # Produit un panneau et, si demande, chacune de ses vignettes separement.
    #
    # Les traces sont decrits une seule fois, dans la liste vignettes : le
    # panneau et les figures individuelles partagent donc rigoureusement le
    # meme code, et ne peuvent pas diverger.
    nl, nc = disposition
    figure = plt.figure(figsize=taille)
    gs = GridSpec(nl, nc, figure=figure, hspace=0.34, wspace=0.30)
    if suptitre:
        figure.suptitle(suptitre, fontsize=12, fontweight='bold')

    for k, (nom, trace, polaire) in enumerate(vignettes):
        i, j = divmod(k, nc)
        ax = (figure.add_subplot(gs[i, j], projection='polar') if polaire
              else figure.add_subplot(gs[i, j]))
        trace(ax, figure)

    enregistrer(figure, dossier, f"{numero:02d}-{titre_panneau}")

    if ENREGISTRER_VIGNETTES:
        for k, (nom, trace, polaire) in enumerate(vignettes):
            f = figure_seule(trace, polaire=polaire)
            enregistrer(f, dossier, f"{numero:02d}{chr(97+k)}-{nom}")
            plt.close(f)

    if not AFFICHER_PANNEAUX:
        plt.close(figure)
    return figure


# =============================================================================
# EXPORT
# =============================================================================

def exporter(dossier, ret, reco, simu, comp, p, re_k, gamma, fconf,
             instants):
    ex = reco['extrapolation']; pot = reco['potentiel']

    r_sortie = np.linspace(p['r_ext'], reco['r_e2'].max(), 400)
    if ex:
        e2s = modele_decroissance(r_sortie, ex['amplitude'], ex['longueur'],
                                  ex['r_bord'])
    else:
        e2s = np.interp(r_sortie, reco['r_e2'], reco['e2'])
    grads = np.gradient(e2s, r_sortie)
    ch = os.path.join(dossier, "champ_reconstruit.csv")
    np.savetxt(ch, np.column_stack([r_sortie, e2s, grads]), delimiter=',',
               header="r_m,E2_V2_par_m2,gradE2_V2_par_m3", comments='')
    print(f"[ENREGISTRE] {ch}")

    ch2 = os.path.join(dossier, "champ_simule.csv")
    np.savetxt(ch2, np.column_stack([simu['r'], simu['e2'], simu['gradient'],
                                     simu['v']]), delimiter=',',
               header="r_m,E2_V2_par_m2,gradE2_V2_par_m3,v_m_par_s",
               comments='')
    print(f"[ENREGISTRE] {ch2}")

    ch3 = os.path.join(dossier, "comparaison.csv")
    np.savetxt(ch3, np.column_stack([comp['r'], comp['v_mesuree'],
                                     comp['v_simulee'], comp['rapport'],
                                     reco['force'], reco['gradient']]),
               delimiter=',',
               header="r_m,v_mesuree,v_simulee,rapport,force_N,gradE2",
               comments='')
    print(f"[ENREGISTRE] {ch3}")

    meta = dict(
        origine="Champ_DEP_Trois_Sections.py",
        geometrie=dict(r_interieur_um=p['r_int'] * 1e6,
                       r_exterieur_um=p['r_ext'] * 1e6,
                       hauteur_cavite_um=p['gap'] * 1e6,
                       hauteur_mesure_um=p['z'] * 1e6),
        signal=dict(tension_affichee_V=p['tension_affichee'],
                    convention=p['convention'], amplitude_V=p['amplitude'],
                    frequence_Hz=p['frequence'], forme=p['forme'],
                    facteur_forme=facteur_forme_onde(p['forme'])),
        milieu=dict(eps_relative=p['eps_milieu_rel'],
                    sigma_S_par_m=p['sigma_milieu'],
                    viscosite_Pa_s=p['viscosite'],
                    rho_kg_par_m3=p['rho_milieu']),
        billes=dict(rayon_um=p['r_bille'] * 1e6, re_k=re_k,
                    gamma_N_s_par_m=gamma, facteur_confinement=fconf),
        section_a=dict(
            instants_activation_s=({str(k): float(v)
                                    for k, v in instants.items()}
                                   if isinstance(instants, dict) else instants),
            rayon_minimal_um=p['r_min'] * 1e6,
            n_bandes=len(ret['r']),
            longueur_decroissance_um=(ex['longueur'] * 1e6 if ex else None),
            cavite_impliquee_um=(ex['longueur'] * math.pi * 1e6 if ex else None),
            potentiel_estime_V=(pot['v_estime'] if pot else None),
            rapport_pourcent=(100 * pot['rapport'] if pot else None)),
        section_b=dict(longueur_decroissance_um=simu['longueur'] * 1e6,
                       longueur_attendue_um=p['gap'] / math.pi * 1e6,
                       champ_maximal_V_par_m=float(
                           math.sqrt(np.max(simu['e2'])))),
        section_c=dict(
            rapport_min=float(np.nanmin(comp['rapport'])),
            rapport_max=float(np.nanmax(comp['rapport'])),
            potentiels_apparents_V=comp['potentiels'].tolist(),
            bornes_um=(comp['bornes'] * 1e6).tolist()))
    chj = os.path.join(dossier, "parametres.json")
    with open(chj, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[ENREGISTRE] {chj}")


# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================

def principal():
    print("=" * 70)
    print("CHAMP DIELECTROPHORETIQUE : RECONSTRUCTION, SIMULATION, COMPARAISON")
    print("=" * 70)
    if not TK_DISPO:
        print("[INFO] mode non interactif")
        return

    racine_tk()
    chemin = filedialog.askopenfilename(
        title="Choisir le fichier de trajectoires des billes",
        filetypes=[("Classeurs Excel", "*.xlsx *.xls")])
    if not chemin:
        print("[INFO] aucun fichier choisi, arret.")
        return
    dossier = filedialog.askdirectory(
        title="Choisir le dossier ou enregistrer les resultats")
    if not dossier:
        dossier = os.path.dirname(os.path.abspath(chemin))
    print(f"[INFO] resultats dans : {dossier}")

    print(f"\n[LECTURE] {os.path.basename(chemin)}")
    brutes = lire_trajectoires(chemin)
    if brutes.empty:
        print("[ERREUR] aucune trajectoire exploitable.")
        return
    print(f"  {len(brutes)} points, {brutes['particule'].nunique()} billes")

    p = saisir_parametres()
    if p is None:
        print("[INFO] saisie annulee, arret.")
        return

    re_k = facteur_clausius_mossotti(p)
    gamma, fconf, lam = coefficient_trainee(p['r_bille'], p['gap'],
                                            p['viscosite'])
    print(f"\n[PARAMETRES DERIVES]")
    print(f"  Re[K] = {re_k:+.5f}")
    print(f"  confinement : rapport {lam:.3f}, facteur {fconf:.4f}")
    print(f"  gamma = {gamma:.4e} N.s/m")
    print(f"  amplitude = {p['amplitude']:.2f} V "
          f"({p['tension_affichee']:.0f} V {p['convention']})")

    sync, instants, groupes = synchroniser(brutes, p)
    if sync.empty:
        return
    # SEPARATION STRICTE entre ce qui est affiche et ce qui est calcule.
    #
    # Les points anterieurs a l'activation sont conservés dans sync pour
    # l'affichage, mais ils n'ont AUCUNE raison d'entrer dans le calcul des
    # vitesses ni du champ : la particule y est au repos, et les inclure
    # ajouterait des vitesses nulles au profil moyen, ce qui biaiserait la
    # longueur de decroissance et donc le potentiel reconstruit.
    sync_calcul = sync[sync['temps_s'] >= 0].copy()
    n_hors = len(sync) - len(sync_calcul)
    if n_hors:
        print(f"\n[CALCUL] {n_hors} points anterieurs a l'activation ecartes "
              f"des calculs,")
        print(f"         ils restent traces sur les figures.")
    don = calculer_vitesses(sync_calcul)
    comp_prof, ret = profil_moyen(don, p['bandes'], p['r_min'])
    if len(ret['r']) < 5:
        print("[ERREUR] profil trop pauvre.")
        return

    reco = section_a(ret, p, re_k, gamma)
    simu = section_b(p, re_k, gamma, ret['r'])
    comp = section_c(ret, reco, simu, p)

    print("\n[FIGURES]")
    produire(dossier, 1, "cinematique",
             [("trajectoires", lambda a, f: t_trajectoires(a, brutes, p), True),
              ("position-radiale",
               lambda a, f: t_position(a, sync, instants, p), False),
              ("vitesse-temps", lambda a, f: t_vitesse_temps(a, don), False),
              ("vitesse-par-bille", lambda a, f: t_vitesse_par_bille(a, don), False),
              ("profil-moyen", lambda a, f: t_profil_moyen(a, comp_prof, ret, p), False),
              ("profil-moyen-zoom", lambda a, f: t_profil_moyen(a, comp_prof, ret, p, True), False)],
             (2, 3), (17, 10),
             (f"Cinematique des billes, {len(instants)} acquisitions"
              if len(instants) > 1 else
              f"Cinematique des billes, activation a t = "
              f"{list(instants.values())[0]:.2f} s"))

    produire(dossier, 2, "chaine-ascendante",
             [("vitesse", lambda a, f: t_etape_vitesse(a, ret), False),
              ("force", lambda a, f: t_etape_force(a, ret, reco), False),
              ("gradient", lambda a, f: t_etape_gradient(a, ret, reco), False),
              ("carre-du-champ", lambda a, f: t_etape_champ(a, reco), False)],
             (2, 2), (14, 9),
             f"Section A, chaine ascendante   Re[K] = {re_k:+.4f}, "
             f"gamma = {gamma:.3e} N.s/m")

    produire(dossier, 3, "champ-et-potentiel",
             [("champ-complet", lambda a, f: t_champ_extrapole(a, reco, p, False), False),
              ("champ-zoom", lambda a, f: t_champ_extrapole(a, reco, p, True), False),
              ("champ-electrique", lambda a, f: t_champ_electrique(a, reco, p), False),
              ("recapitulatif", lambda a, f: t_recapitulatif(a, reco, p), False)],
             (2, 2), (14, 9),
             "Section A, champ reconstruit et potentiel effectif")

    produire(dossier, 4, "simulation-directe",
             [("potentiel-coupe", lambda a, f: t_carte_potentiel(a, simu, p, f), False),
              ("champ-coupe", lambda a, f: t_carte_champ(a, simu, p, f), False),
              ("champ-vue-dessus", lambda a, f: t_carte_dessus(a, simu, p, f), False),
              ("potentiel-profil", lambda a, f: t_profil_potentiel(a, simu, p), False),
              ("champ-profil", lambda a, f: t_profil_simule(a, simu, p), False),
              ("gradient-profil", lambda a, f: t_gradient_simule(a, simu, p), False),
              ("vitesse-simulee", lambda a, f: t_vitesse_simulee(a, simu, p, ret), False)],
             (3, 3), (19, 14),
             f"Section B, simulation directe a {p['amplitude']:.1f} V "
             f"({p['tension_affichee']:.0f} V {p['convention']}), "
             f"aucun parametre ajuste")

    produire(dossier, 5, "comparaison",
             [("vitesses", lambda a, f: t_comparaison_vitesse(a, comp, p), False),
              ("rapport", lambda a, f: t_comparaison_rapport(a, comp), False),
              ("champs", lambda a, f: t_comparaison_champ(a, reco, simu, p), False),
              ("potentiel-apparent", lambda a, f: t_potentiel_apparent(a, comp, p), False)],
             (2, 2), (15, 10),
             "Section C, confrontation des deux chemins")

    trajs = trajectoires_comparees(sync, reco, simu, p, re_k, gamma,
                                   instants)
    produire(dossier, 6, "trajectoires-comparees",
             [("mesure-vs-reconstruit",
               lambda a, f: t_comparaison_reconstruit(a, trajs, p), False),
              ("mesure-vs-laplace",
               lambda a, f: t_comparaison_laplace(a, trajs, p), False),
              ("arrivees", lambda a, f: t_arrivees(a, trajs, p), False),
              ("deux-champs", lambda a, f: t_deux_champs(a, reco, simu, p), False)],
             (2, 2), (15, 11),
             ("Trajectoires mesurees et simulees, chaque acquisition recalee "
              "sur sa propre activation" if len(instants) > 1 else
              f"Trajectoires mesurees et simulees, origine a l'activation "
              f"(t = {list(instants.values())[0]:.2f} s)"))

    print("\n[EXPORT]")
    exporter(dossier, ret, reco, simu, comp, p, re_k, gamma, fconf,
             instants)

    if AFFICHER_PANNEAUX:
        print(f"\n[AFFICHAGE] {len(plt.get_fignums())} panneaux, fermer les "
              f"fenetres pour terminer", flush=True)
        plt.show()

    print("\n" + "=" * 70)
    print("TERMINE.")
    print("=" * 70)


if __name__ == "__main__":
    principal()