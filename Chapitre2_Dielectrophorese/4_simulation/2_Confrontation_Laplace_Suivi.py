# -*- coding: utf-8 -*-
# =============================================================================
# CONFRONTATION DU MODELE DIRECT AUX TRAJECTOIRES MESUREES
# Champ issu de la resolution de Laplace contre suivi de cellules
# =============================================================================
#
# CE QUE FAIT LE PROGRAMME
#
#   Il repond a une seule question : le champ calcule par la resolution de
#   Laplace rend-il compte du mouvement reellement observe ?
#
#   Deux confrontations sont produites, qui ne disent pas la meme chose.
#
#     VITESSE EN FONCTION DE LA POSITION. C'est la comparaison la plus
#     directe, puisque le modele relie precisement ces deux grandeurs. Elle est
#     indifferente a l'instant d'activation et a la position de depart, et fait
#     apparaitre ou le modele se trompe le long du rayon.
#
#     POSITION EN FONCTION DU TEMPS. C'est la comparaison la plus parlante,
#     puisqu'elle montre ce qu'on observe a l'oeil. Elle cumule en revanche les
#     ecarts le long de la trajectoire, de sorte qu'une erreur locale de vitesse
#     s'y traduit par un decalage durable.
#
#   Le programme n'ajuste RIEN. Le facteur de Clausius-Mossotti est impose par
#   l'utilisateur, et la simulation part de la position reellement occupee par
#   chaque cellule a l'instant d'activation. C'est ce qui donne son sens a la
#   confrontation : tout ecart constate est imputable au modele, non a un
#   parametre libre.
#
# -----------------------------------------------------------------------------
# ENTREES
# -----------------------------------------------------------------------------
#   un champ, sous forme de fichier CSV a colonnes r_m et E2_V2_par_m2,
#   tel que produit par le programme de reconstruction ;
#   un classeur de suivi, un onglet par cellule, avec temps et rayon.
#
# -----------------------------------------------------------------------------
# SORTIES, dans le dossier choisi
# -----------------------------------------------------------------------------
#   01-vitesse-position.svg    vitesse mesuree et vitesse predite le long du rayon
#   02-position-temps.svg      trajectoires mesurees et simulees
#   03-diagnostic.svg          rapport des vitesses et bilan chiffre
#   confrontation.csv          toutes les grandeurs point par point
#   bilan.json                 chiffres de synthese, pour la redaction
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

from scipy.signal import savgol_filter

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    TK_DISPO = True
except ImportError:
    TK_DISPO = False
    print("[ALERTE] tkinter indisponible : les fonctions restent importables.")


# =============================================================================
# CONFIGURATION
# =============================================================================

R_INT_UM       = 50.0
R_EXT_UM       = 140.0
HAUTEUR_CAV_UM = 20.0

# --- Cellules ----------------------------------------------------------------
# Le rayon est lu dans le fichier de suivi s'il y figure. La valeur ci-dessous
# ne sert que de repli.
RAYON_CELLULE_UM = 5.0
SOURCE_RAYON = "fichier"

# --- Facteur de Clausius-Mossotti impose -------------------------------------
# C'est le point essentiel de cette confrontation : la valeur n'est PAS
# ajustee. Nous employons celle que la litterature conduit a attendre pour une
# microalgue en dielectrophorese negative franche, de sorte que l'ecart
# constate ne puisse pas etre attribue a un choix de parametre.
RE_K_IMPOSE = -0.30

# Bornes de la bande de sûreté.
#
# Plutot qu'une valeur unique, dont le choix pourrait etre discute, nous
# encadrons la prediction par les deux valeurs extremes plausibles. La
# conclusion ne depend alors plus d'un parametre : si l'ecart subsiste sur
# toute la bande, il ne peut pas etre impute au choix de Re[K].
#
# La borne basse correspond a la limite theorique d'une sphere homogene, qui
# donne la force la plus intense en dielectrophorese negative. La borne haute
# correspond a la reponse la plus faible qu'on puisse raisonnablement envisager
# pour une cellule dont les indicateurs signalent une repulsion franche.
RE_K_BANDE_MIN = -0.50
RE_K_BANDE_MAX = -0.10

# Mode de construction de la bande.
#   "cellule"  chaque cellule recoit sa propre bande, tracee autour de sa
#              trajectoire. C'est le plus rigoureux, puisque la bande depend de
#              la position de depart et du rayon propres a chaque cellule.
#   "globale"  une bande unique enveloppe toutes les cellules. Plus lisible
#              quand les departs sont voisins, mais elle melange des cellules
#              de tailles differentes.
MODE_BANDE = "cellule"

# --- Milieu ------------------------------------------------------------------
EPS_MILIEU_REL  = 78.0
SIGMA_MILIEU    = 0.120
VISCOSITE_MPA_S = 1.0

# --- Recalage temporel -------------------------------------------------------
SEUIL_ACTIVATION_UM_S = 2.0
POINTS_CONSECUTIFS = 3
LISSAGE_DETECTION_S = 0.30
ECART_MAX_ACQUISITION = 1.5

# --- Derivation a fenetre adaptative -----------------------------------------
# La position est quantifiee au pixel. Loin de l'electrode la cellule ralentit,
# et sur une fenetre courte elle ne parcourt qu'une fraction de pixel : la
# vitesse calculee est alors dominee par les sauts de quantification. La
# fenetre est donc elargie localement jusqu'a ce que le deplacement couvre
# plusieurs pas.
ECHELLE_PX_PAR_UM = 3.06
PAS_MINIMAUX = 3.0
FENETRES_CANDIDATES = (5, 9, 15, 25, 41, 65, 101, 151)
LISSAGE_ORDRE = 2

# --- Moyennage ---------------------------------------------------------------
NB_BANDES = 30
POINTS_MIN_PAR_BANDE = 3

# --- Integration -------------------------------------------------------------
PAS_TEMPS_MS = 5.0
DEPLACEMENT_MAX_UM = 0.2

# --- Affichage ---------------------------------------------------------------
R_AFFICHAGE_UM = 300.0

EPSILON_0 = 8.854187817e-12

matplotlib.rcParams['figure.facecolor']  = 'white'
matplotlib.rcParams['savefig.facecolor'] = 'white'
matplotlib.rcParams['axes.facecolor']    = 'white'

COUL_MES  = 'magenta'
COUL_SIM  = '#9467bd'
COUL_OR   = 'gold'
COUL_GRIS = '#7f7f7f'
COUL_ALER = '#d62728'

COLONNES_TEMPS = ['temps_s', 'time_s', 'temps', 'time', 't_s', 't']
COLONNES_RAYON = ['r_um', 'rayon_um', 'radius_um', 'r', 'rayon', 'distance_um']
COLONNES_TAILLE = ['r_part_um', 'rayon_particule_um', 'a_um', 'taille_um']
COLONNES_IDENT = ['particle_id', 'cellule', 'particule', 'id', 'track_id']
COLONNES_SUIVI = ['tracked', 'suivi', 'valide', 'valid']


# =============================================================================
# PHYSIQUE
# =============================================================================

def coefficient_trainee(rayon, hauteur, viscosite):
    # Trainee corrigee du confinement entre deux parois planes, modele de
    # Happel et Brenner.
    g0 = 6.0 * math.pi * viscosite * rayon
    lam = rayon / (hauteur / 2.0)
    if lam < 0.95:
        den = (1.0 - 1.004 * lam + 0.418 * lam ** 3
               + 0.21 * lam ** 4 - 0.169 * lam ** 5)
        f = 1.0 / den
    else:
        f = 1.0 + (9.0 / 8.0) * lam + lam ** 3
    return g0 * f, f, lam


def prefacteur(p, rayon, re_k):
    # 2 pi a^3 eps_m Re[K], le rayon etant celui de la PARTICULE.
    return 2.0 * math.pi * rayon ** 3 * p['eps_milieu'] * re_k


def charger_champ(chemin):
    # Lit un champ tabule et en deduit le gradient.
    #
    # Le fichier peut contenir une colonne de gradient, mais nous la
    # recalculons systematiquement par derivation du carre du champ : cela
    # garantit que les deux grandeurs sont coherentes entre elles, et que le
    # gradient employe correspond bien au champ affiche.
    d = np.genfromtxt(chemin, delimiter=',', names=True)
    r = np.asarray(d['r_m'], dtype=float)
    e2 = np.asarray(d['E2_V2_par_m2'], dtype=float)
    ordre = np.argsort(r)
    r, e2 = r[ordre], e2[ordre]
    grad = np.gradient(e2, r, edge_order=2)
    return dict(r=r, e2=e2, gradient=grad, source=os.path.basename(chemin),
                e_max=math.sqrt(float(np.max(e2))))


def vitesse_predite(r, champ, pref, gamma):
    g = np.interp(r, champ['r'], champ['gradient'], left=0.0, right=0.0)
    return pref * g / gamma


def integrer(r0, champ, pref, gamma, t, dt=None, deplacement_max=None):
    # Integre dr/dt = v(r) avec un pas adaptatif et un schema du point milieu.
    #
    # Le pas adaptatif est indispensable ici : au voisinage de l'arete, le champ
    # calcule produit des vitesses de plusieurs milliers de micrometres par
    # seconde, et un pas fixe ferait franchir a la particule, en une seule
    # etape, une distance superieure a celle sur laquelle le champ varie.
    if dt is None:
        dt = PAS_TEMPS_MS * 1e-3
    if deplacement_max is None:
        deplacement_max = DEPLACEMENT_MAX_UM * 1e-6
    duree = float(t[-1])
    n = max(int(duree / dt) + 1, 2)
    ts = np.linspace(0.0, duree, n)
    rs = np.zeros(n)
    rs[0] = r0
    r_min = float(champ['r'].min())
    r_max = float(champ['r'].max())

    def v(x):
        g = float(np.interp(x, champ['r'], champ['gradient'],
                            left=0.0, right=0.0))
        return pref * g / gamma

    for i in range(n - 1):
        courant = rs[i]
        restant = dt
        garde = 0
        while restant > 1e-12 and garde < 20000:
            garde += 1
            vv = v(courant)
            if abs(vv) < 1e-15:
                break
            sous_pas = min(restant, deplacement_max / abs(vv))
            v_mil = v(min(max(courant + 0.5 * vv * sous_pas, r_min), r_max))
            courant = min(max(courant + v_mil * sous_pas, r_min), r_max)
            restant -= sous_pas
        rs[i + 1] = courant
    return np.interp(t, ts, rs)


# =============================================================================
# LECTURE ET RECALAGE
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


def _derivee_lissee(t, r, duree=LISSAGE_DETECTION_S, ordre=2):
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
    if f < ordre + 2:
        return np.gradient(r, t)
    return savgol_filter(r, f, ordre, deriv=1, delta=dt)


def vitesse_adaptative(t, r, pas_q=None, n_pas=PAS_MINIMAUX,
                       fenetres=FENETRES_CANDIDATES, ordre=LISSAGE_ORDRE):
    # Derivee de la position sur une fenetre elargie localement jusqu'a ce que
    # le deplacement depasse plusieurs pas de quantification du suivi.
    n = len(t)
    if n < 5:
        return (np.gradient(r, t) if n > 1 else np.zeros(n),
                np.zeros(n, dtype=bool))
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        return np.gradient(r, t), np.ones(n, dtype=bool)
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
        v = np.gradient(r, t)
        fi = np.ones(n, dtype=bool)
        fi[0] = fi[-1] = False
        return v, fi

    idx = np.arange(n)
    v = np.full(n, np.nan)
    fen = np.zeros(n, dtype=int)
    for f in sorted(der):
        d = f // 2
        a = np.clip(idx - d, 0, n - 1)
        b = np.clip(idx + d, 0, n - 1)
        ok = np.isnan(v) & (np.abs(r[b] - r[a]) >= n_pas * pas_q)
        v[ok] = der[f][ok]
        fen[ok] = f
    reste = np.isnan(v)
    fmax = max(der)
    v[reste] = der[fmax][reste]
    fen[reste] = fmax
    demi = fen // 2
    fiable = (idx - demi >= 0) & (idx + demi <= n - 1)
    return v, fiable


def regrouper_par_acquisition(det, ecart_max=ECART_MAX_ACQUISITION):
    if not det:
        return {}, {}
    cles = sorted(det, key=lambda k: det[k])
    groupes = {cles[0]: 0}
    num = 0
    for prec, cour in zip(cles[:-1], cles[1:]):
        if det[cour] - det[prec] > ecart_max:
            num += 1
        groupes[cour] = num
    instants = {g: float(np.median([det[k] for k in groupes
                                    if groupes[k] == g]))
                for g in set(groupes.values())}
    return groupes, instants


def _tableaux(chemin):
    # Reconnait un classeur a plusieurs onglets, un tableau au format long ou
    # un tableau au format large, sans que l'utilisateur ait a le preciser.
    if chemin.lower().endswith(('.xlsx', '.xls')):
        cl = pd.ExcelFile(chemin)
        if len(cl.sheet_names) > 1:
            return [(o, cl.parse(o)) for o in cl.sheet_names]
        df = cl.parse(cl.sheet_names[0])
    else:
        try:
            df = pd.read_csv(chemin, sep=None, engine='python')
        except Exception:
            df = pd.read_csv(chemin, sep=';', decimal=',')
    ci = _colonne(df.columns, COLONNES_IDENT)
    if ci is not None and df[ci].nunique() > 1:
        return [(f"Cellule_{v}", df[df[ci] == v].copy())
                for v in df[ci].dropna().unique()]
    return [(os.path.splitext(os.path.basename(chemin))[0], df)]


def lire_cellules(chemin, p):
    # Lit le suivi, recale chaque trajectoire sur sa mise en mouvement et
    # calcule les vitesses.
    print("\n" + "=" * 72)
    print("LECTURE DES TRAJECTOIRES")
    print("=" * 72)
    tableaux = _tableaux(chemin)
    brutes, det = [], {}

    for k, (nom, df) in enumerate(tableaux):
        cr = _colonne(df.columns, COLONNES_RAYON)
        if cr is None or df.empty:
            continue
        ct = _colonne(df.columns, COLONNES_TEMPS)
        ca = _colonne(df.columns, COLONNES_TAILLE)
        cs = _colonne(df.columns, COLONNES_SUIVI)
        r = pd.to_numeric(df[cr], errors='coerce').values * 1e-6
        t = (pd.to_numeric(df[ct], errors='coerce').values if ct is not None
             else np.arange(len(df)) / 20.0)
        ok = np.isfinite(r) & np.isfinite(t)
        if cs is not None:
            suivi = df[cs].astype(str).str.strip().str.lower()
            ok &= suivi.isin(['true', '1', 'vrai', 'oui', 'yes']).values
        r, t = r[ok], t[ok]
        if len(r) < 15:
            continue
        o = np.argsort(t)
        r, t = r[o], t[o]

        rayon = None
        if ca is not None:
            vals = pd.to_numeric(df[ca], errors='coerce').values[ok]
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if len(vals):
                rayon = float(np.median(vals)) * 1e-6

        v = _derivee_lissee(t, r)
        indice = None
        for i in range(len(v) - POINTS_CONSECUTIFS + 1):
            if np.all(np.abs(v[i:i + POINTS_CONSECUTIFS]) > p['seuil']):
                indice = i
                break
        brutes.append(dict(nom=nom, t=t, r=r, cle=k, rayon=rayon))
        if indice is not None:
            det[k] = float(t[indice])

    if not brutes:
        print("  [ERREUR] aucune trajectoire exploitable.")
        return []

    if len(det) < max(2, 0.2 * len(brutes)):
        vmax = np.array([float(np.max(np.abs(_derivee_lissee(b['t'], b['r']))))
                         for b in brutes])
        print(f"  [ALERTE] {len(det)} cellule(s) sur {len(brutes)} franchissent "
              f"le seuil de {p['seuil']*1e6:.1f} um/s.")
        print(f"           vitesses maximales observees : "
              f"{vmax.min()*1e6:.2f} a {vmax.max()*1e6:.2f} um/s.")
        print(f"           un seuil de "
              f"{np.percentile(vmax, 25)*0.6*1e6:.1f} um/s serait adapte.")
        for b in brutes:
            det.setdefault(b['cle'], float(b['t'][0]))

    groupes, instants = regrouper_par_acquisition(det)
    print(f"  {len(brutes)} cellules, {len(instants)} acquisition(s)")
    for g in sorted(instants):
        n = sum(1 for kk in groupes if groupes[kk] == g)
        print(f"    acquisition {g + 1} : activation a {instants[g]:.3f} s, "
              f"{n} cellules")

    sorties = []
    for b in brutes:
        k = b['cle']
        t_ref = det.get(k, instants.get(groupes.get(k, 0), b['t'][0]))
        m = b['t'] >= t_ref
        if int(np.sum(m)) < 15:
            continue
        t_r, r_r = b['t'][m] - t_ref, b['r'][m]
        v_r, fiable = vitesse_adaptative(t_r, r_r)
        a = (b['rayon'] if (p['source_rayon'] == "fichier" and b['rayon'])
             else p['r_cellule'])
        sorties.append(dict(nom=b['nom'], t=t_r, r=r_r, v=v_r, fiable=fiable,
                            r0=float(r_r[0]), rayon=a, activation=t_ref,
                            rayon_fichier=b['rayon']))

    print(f"\n{'cellule':>12} {'points':>7} {'rayon':>8} {'depart':>9} "
          f"{'arrivee':>9} {'v max':>10}")
    print("-" * 60)
    for s in sorties:
        print(f"{s['nom']:>12} {len(s['t']):>7} {s['rayon']*1e6:>7.2f}u "
              f"{s['r0']*1e6:>8.1f}u {s['r'][-1]*1e6:>8.1f}u "
              f"{np.max(np.abs(s['v']))*1e6:>9.2f}u")
    print("=" * 72)
    return sorties


# =============================================================================
# CONFRONTATION
# =============================================================================

def profil_mesure(cellules, nb_bandes=NB_BANDES):
    # Profil de vitesse moyen par bandes radiales, toutes cellules confondues.
    r = np.concatenate([s['r'][s['fiable']] for s in cellules])
    v = np.concatenate([s['v'][s['fiable']] for s in cellules])
    bornes = np.linspace(r.min(), r.max(), nb_bandes + 1)
    ce, mo, ec, ef = [], [], [], []
    for i in range(nb_bandes):
        m = (r >= bornes[i]) & (r < bornes[i + 1])
        if int(np.sum(m)) < POINTS_MIN_PAR_BANDE:
            continue
        ce.append(0.5 * (bornes[i] + bornes[i + 1]))
        mo.append(float(np.mean(v[m])))
        ec.append(float(np.std(v[m])))
        ef.append(int(np.sum(m)))
    return dict(r=np.array(ce), v=np.array(mo), ecart=np.array(ec),
                n=np.array(ef), r_tous=r, v_tous=v)


def confronter(cellules, champ, p):
    # Compare vitesse mesuree et vitesse predite, puis trajectoire mesuree et
    # trajectoire simulee, sans ajuster aucun parametre.
    print("\n" + "=" * 72)
    print("CONFRONTATION AU MODELE DIRECT")
    print("=" * 72)
    print(f"  champ         : {champ['source']}")
    print(f"  champ maximal : {champ['e_max']:.4e} V/m")
    print(f"  Re[K] impose  : {p['re_k']:+.3f}, AUCUN parametre ajuste\n")

    prof = profil_mesure(cellules, p['bandes'])

    # Rayon typique, pour le profil de vitesse predite
    a_typ = float(np.median([s['rayon'] for s in cellules]))
    gamma_typ, fconf, lam = coefficient_trainee(a_typ, p['gap'], p['viscosite'])
    pref_typ = prefacteur(p, a_typ, p['re_k'])
    v_pred = vitesse_predite(prof['r'], champ, pref_typ, gamma_typ)

    print(f"  rayon median  : {a_typ*1e6:.2f} um")
    print(f"  confinement   : rapport {lam:.3f}, facteur {fconf:.3f}")
    print(f"  trainee       : {gamma_typ:.4e} N.s/m\n")

    print(f"  VITESSE EN FONCTION DE LA POSITION")
    print(f"  {'r (um)':>9} {'mesuree':>11} {'predite':>13} {'rapport':>12}")
    print("  " + "-" * 48)
    rapports = []
    pas = max(1, len(prof['r']) // 12)
    for i in range(0, len(prof['r']), pas):
        rap = (prof['v'][i] / v_pred[i]) if v_pred[i] != 0 else float('nan')
        rapports.append(rap)
        print(f"  {prof['r'][i]*1e6:>9.1f} {prof['v'][i]*1e6:>10.2f}u "
              f"{v_pred[i]*1e6:>12.3f}u {rap:>12.2f}")

    # Rapport sur tout le profil, et rayon ou les deux se croisent
    rap_tous = np.where(v_pred != 0, prof['v'] / v_pred, np.nan)
    fini = np.isfinite(rap_tous)
    croisement = None
    if np.any(fini):
        idx = np.where(np.diff(np.sign(rap_tous[fini] - 1.0)) != 0)[0]
        if len(idx):
            croisement = float(prof['r'][fini][idx[0]])

    # Trajectoires simulees, depuis la position reelle a l'activation
    print(f"\n  POSITION EN FONCTION DU TEMPS")
    print(f"  {'cellule':>12} {'depart':>9} {'mesuree':>10} {'simulee':>10} "
          f"{'ecart':>9}")
    print("  " + "-" * 54)
    trajs, ecarts = [], []
    for s in cellules:
        gamma, _, _ = coefficient_trainee(s['rayon'], p['gap'], p['viscosite'])
        pref = prefacteur(p, s['rayon'], p['re_k'])
        r_sim = integrer(s['r0'], champ, pref, gamma, s['t'])
        # Bornes de la bande, obtenues en integrant aux deux valeurs extremes
        r_bas = integrer(s['r0'], champ,
                         prefacteur(p, s['rayon'], p['re_k_min']), gamma,
                         s['t'])
        r_haut = integrer(s['r0'], champ,
                          prefacteur(p, s['rayon'], p['re_k_max']), gamma,
                          s['t'])
        ecart = float(np.sqrt(np.mean((r_sim - s['r']) ** 2)))
        # Ecart a la borne la plus favorable, celle qui s'ecarte le moins
        ecart_favorable = min(
            float(np.sqrt(np.mean((r_bas - s['r']) ** 2))),
            float(np.sqrt(np.mean((r_haut - s['r']) ** 2))))
        ecarts.append(ecart)
        trajs.append(dict(nom=s['nom'], t=s['t'], r_mes=s['r'], r_sim=r_sim,
                          r_bas=r_bas, r_haut=r_haut, r0=s['r0'],
                          rayon=s['rayon'], ecart=ecart,
                          ecart_favorable=ecart_favorable))
        print(f"  {s['nom']:>12} {s['r0']*1e6:>8.1f}u {s['r'][-1]*1e6:>9.1f}u "
              f"{r_sim[-1]*1e6:>9.1f}u {ecart*1e6:>8.1f}u")

    depl_mes = float(np.mean([s['r'][-1] - s['r0'] for s in cellules]))
    depl_sim = float(np.mean([t['r_sim'][-1] - t['r0'] for t in trajs]))
    depl_bas = float(np.mean([t['r_bas'][-1] - t['r0'] for t in trajs]))
    depl_haut = float(np.mean([t['r_haut'][-1] - t['r0'] for t in trajs]))
    # La bande recouvre-t-elle les mesures ?
    recouvre = sum(1 for s, t in zip(cellules, trajs)
                   if min(t['r_bas'][-1], t['r_haut'][-1]) <= s['r'][-1]
                   <= max(t['r_bas'][-1], t['r_haut'][-1]))
    print("  " + "-" * 54)
    print(f"  {'moyenne':>12} {'':>9} {depl_mes*1e6:>+9.1f}u "
          f"{depl_sim*1e6:>+9.1f}u {np.mean(ecarts)*1e6:>8.1f}u")

    print(f"\n  LECTURE")
    if croisement is not None:
        print(f"    les deux vitesses se croisent a r = {croisement*1e6:.0f} um.")
        m_avant = prof['r'] < croisement
        m_apres = prof['r'] > croisement
        if np.any(m_avant):
            print(f"    en deca, le modele predit des vitesses "
                  f"{1/np.nanmedian(rap_tous[m_avant]):.0f} fois trop elevees")
        if np.any(m_apres):
            print(f"    au dela, il les predit "
                  f"{np.nanmedian(rap_tous[m_apres]):.0f} fois trop faibles")
    else:
        print(f"    le rapport ne traverse pas l'unite sur la plage mesuree :")
        print(f"    le modele se trompe dans le meme sens partout, avec un")
        print(f"    rapport median de {np.nanmedian(rap_tous):.2f}.")
    print(f"\n    deplacement moyen mesure  : {depl_mes*1e6:+.1f} um")
    print(f"    bande de surete           : {depl_haut*1e6:+.1f} a "
          f"{depl_bas*1e6:+.1f} um")
    print(f"      borne Re[K] = {p['re_k_max']:+.2f} : {depl_haut*1e6:+.1f} um")
    print(f"      valeur centrale {p['re_k']:+.2f}   : {depl_sim*1e6:+.1f} um")
    print(f"      borne Re[K] = {p['re_k_min']:+.2f} : {depl_bas*1e6:+.1f} um")
    print(f"\n    {recouvre} cellule(s) sur {len(cellules)} tombent dans la "
          f"bande")
    if recouvre == 0:
        print(f"    AUCUNE mesure n'est atteinte, meme par la borne la plus")
        print(f"    favorable : l'ecart ne peut donc pas etre impute au choix")
        print(f"    du facteur de Clausius-Mossotti.")
    if abs(depl_haut) > 1e-12:
        print(f"\n    rapport a la borne la plus favorable : "
              f"{depl_mes/depl_haut:.2f}")
    print("=" * 72)

    return dict(profil=prof, v_predite=v_pred, rapport=rap_tous,
                croisement=croisement, trajectoires=trajs,
                a_typique=a_typ, gamma_typique=gamma_typ,
                confinement=fconf, lam=lam,
                deplacement_mesure=depl_mes, deplacement_simule=depl_sim,
                deplacement_bas=depl_bas, deplacement_haut=depl_haut,
                recouvre=recouvre,
                ecart_moyen=float(np.mean(ecarts)),
                ecart_favorable=float(np.mean([t['ecart_favorable']
                                               for t in trajs])))


# =============================================================================
# FIGURES
# =============================================================================

def _anneau_v(ax, p):
    ax.axvspan(p['r_int'] * 1e6, p['r_ext'] * 1e6, alpha=0.22, color=COUL_OR,
               label="anneau d'or")


def _anneau_h(ax, p):
    ax.axhspan(p['r_int'] * 1e6, p['r_ext'] * 1e6, alpha=0.22, color=COUL_OR)


def figure_vitesse_position(cellules, res, champ, p):
    # Vitesse en fonction de la position, mesuree et predite.
    fig = plt.figure(figsize=(15, 6))
    gs = GridSpec(1, 2, figure=fig, wspace=0.26)
    fig.suptitle(
        f"Vitesse en fonction de la position, bande de surete Re[K] de "
        f"{p['re_k_min']:+.2f} a {p['re_k_max']:+.2f}, aucun parametre ajuste",
        fontsize=12, fontweight='bold')
    prof = res['profil']

    for k, zoom in enumerate([False, True]):
        ax = fig.add_subplot(gs[0, k])
        coul = plt.cm.tab10(np.linspace(0, 1, len(cellules)))
        for s, c in zip(cellules, coul):
            m = s['fiable']
            ax.scatter(s['r'][m] * 1e6, s['v'][m] * 1e6, s=7, alpha=0.25,
                       color=c)
        ax.errorbar(prof['r'] * 1e6, prof['v'] * 1e6,
                    yerr=prof['ecart'] * 1e6, fmt='o-', color=COUL_MES, ms=6,
                    lw=2, capsize=3, markeredgecolor='k', markeredgewidth=0.8,
                    label="mesuree, moyenne par bandes")
        # Bande de surete : la vitesse etant proportionnelle a Re[K], les deux
        # bornes se deduisent l'une de l'autre par un simple rapport, et la
        # bande est exactement l'ensemble des predictions possibles.
        rr = np.linspace(prof['r'].min() * 0.95, prof['r'].max() * 1.05, 600)
        gamma = res['gamma_typique']
        v_min = vitesse_predite(rr, champ,
                                prefacteur(p, res['a_typique'],
                                           p['re_k_min']), gamma) * 1e6
        v_max = vitesse_predite(rr, champ,
                                prefacteur(p, res['a_typique'],
                                           p['re_k_max']), gamma) * 1e6
        v_cen = vitesse_predite(rr, champ,
                                prefacteur(p, res['a_typique'],
                                           p['re_k']), gamma) * 1e6
        ax.fill_between(rr * 1e6, v_max, v_min, color=COUL_GRIS, alpha=0.22,
                        lw=0, label="bande de surete")
        ax.plot(rr * 1e6, v_min, '--', color=COUL_ALER, lw=1.8, dashes=(6, 3),
                label=f"Re[K] = {p['re_k_min']:+.2f}")
        ax.plot(rr * 1e6, v_max, '--', color='#2ca02c', lw=1.8, dashes=(6, 3),
                label=f"Re[K] = {p['re_k_max']:+.2f}")
        ax.plot(rr * 1e6, v_cen, '-', color=COUL_SIM, lw=2.4,
                label=f"Re[K] = {p['re_k']:+.2f}, valeur centrale")
        if res['croisement'] is not None:
            ax.axvline(res['croisement'] * 1e6, color='k', ls=':', lw=1.6,
                       label=f"croisement, {res['croisement']*1e6:.0f} um")
        ax.axhline(0, color='k', lw=0.8)
        _anneau_v(ax, p)
        if zoom:
            haut = float(np.percentile(prof['v'], 90)) * 1e6
            ax.set_ylim(-0.1 * haut, haut * 1.5)
            ax.set_xlim(prof['r'].min() * 1e6 * 0.98,
                        prof['r'].max() * 1e6 * 1.02)
            titre = "Agrandissement sur la plage mesuree"
        else:
            ax.set_xlim(p['r_ext'] * 1e6 * 0.95, R_AFFICHAGE_UM)
            titre = "Vue complete"
        ax.set_xlabel("Position radiale (um)")
        ax.set_ylabel("Vitesse radiale (um/s)")
        ax.set_title(titre, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    return fig


def figure_position_temps(res, p):
    # Trajectoires mesurees et simulees, cellule par cellule.
    fig = plt.figure(figsize=(15, 6))
    gs = GridSpec(1, 2, figure=fig, wspace=0.26)
    fig.suptitle(
        f"Position en fonction du temps, bande de surete Re[K] de "
        f"{p['re_k_min']:+.2f} a {p['re_k_max']:+.2f}",
        fontsize=12, fontweight='bold')

    ax = fig.add_subplot(gs[0, 0])
    coul = plt.cm.tab10(np.linspace(0, 1, len(res['trajectoires'])))
    mode = p.get('mode_bande', MODE_BANDE)

    if mode == "globale":
        # Une seule bande, enveloppant l'ensemble des cellules
        t_ref = res['trajectoires'][0]['t']
        bas = np.max([np.interp(t_ref, t['t'], t['r_bas'])
                      for t in res['trajectoires']], axis=0)
        haut = np.min([np.interp(t_ref, t['t'], t['r_haut'])
                       for t in res['trajectoires']], axis=0)
        ax.fill_between(t_ref, haut * 1e6, bas * 1e6, color=COUL_GRIS,
                        alpha=0.22, lw=0, label="bande de surete")
        ax.plot(t_ref, bas * 1e6, '--', color=COUL_ALER, lw=1.8, dashes=(6, 3),
                label=f"Re[K] = {p['re_k_min']:+.2f}")
        ax.plot(t_ref, haut * 1e6, '--', color='#2ca02c', lw=1.8,
                dashes=(6, 3), label=f"Re[K] = {p['re_k_max']:+.2f}")
    else:
        # Une bande par cellule, autour de sa propre trajectoire
        for t, c in zip(res['trajectoires'], coul):
            ax.fill_between(t['t'], t['r_haut'] * 1e6, t['r_bas'] * 1e6,
                            color=COUL_GRIS, alpha=0.18, lw=0)
            ax.plot(t['t'], t['r_bas'] * 1e6, '--', color=COUL_ALER, lw=1.2,
                    dashes=(6, 3), alpha=0.75)
            ax.plot(t['t'], t['r_haut'] * 1e6, '--', color='#2ca02c', lw=1.2,
                    dashes=(6, 3), alpha=0.75)
        ax.fill_between([], [], [], color=COUL_GRIS, alpha=0.18,
                        label="bande de surete")
        ax.plot([], [], '--', color=COUL_ALER, lw=1.2, dashes=(6, 3),
                label=f"Re[K] = {p['re_k_min']:+.2f}")
        ax.plot([], [], '--', color='#2ca02c', lw=1.2, dashes=(6, 3),
                label=f"Re[K] = {p['re_k_max']:+.2f}")

    for t, c in zip(res['trajectoires'], coul):
        ax.plot(t['t'], t['r_mes'] * 1e6, '-', color=c, lw=1.6, alpha=0.95,
                zorder=4)
        ax.plot(t['t'], t['r_sim'] * 1e6, '-', color=COUL_SIM, lw=1.4,
                alpha=0.55, zorder=3)
        ax.plot([0], [t['r0'] * 1e6], 'o', color=c, ms=7,
                markeredgecolor='k', markeredgewidth=0.9, zorder=6)
    ax.plot([], [], '-', color='k', lw=1.6, label="mesuree")
    ax.plot([], [], '-', color=COUL_SIM, lw=1.4,
            label=f"Re[K] = {p['re_k']:+.2f}, centrale")
    _anneau_h(ax, p)
    ax.set_xlabel("Temps depuis l'activation (s)")
    ax.set_ylabel("Position radiale (um)")
    ax.set_title("Trajectoires", fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    for t, c in zip(res['trajectoires'], coul):
        ax.fill_between(t['t'], (t['r_haut'] - t['r0']) * 1e6,
                        (t['r_bas'] - t['r0']) * 1e6, color=COUL_GRIS,
                        alpha=0.18, lw=0)
        ax.plot(t['t'], (t['r_bas'] - t['r0']) * 1e6, '--', color=COUL_ALER,
                lw=1.2, dashes=(6, 3), alpha=0.75)
        ax.plot(t['t'], (t['r_haut'] - t['r0']) * 1e6, '--', color='#2ca02c',
                lw=1.2, dashes=(6, 3), alpha=0.75)
    for t, c in zip(res['trajectoires'], coul):
        ax.plot(t['t'], (t['r_mes'] - t['r0']) * 1e6, '-', color=c, lw=1.6,
                alpha=0.95, zorder=4)
    ax.axhline(0, color='k', lw=0.9)
    ax.fill_between([], [], [], color=COUL_GRIS, alpha=0.18,
                    label="bande de surete")
    ax.plot([], [], '--', color=COUL_ALER, lw=1.2, dashes=(6, 3),
            label=f"Re[K] = {p['re_k_min']:+.2f}")
    ax.plot([], [], '--', color='#2ca02c', lw=1.2, dashes=(6, 3),
            label=f"Re[K] = {p['re_k_max']:+.2f}")
    ax.plot([], [], '-', color='k', lw=1.6, label="mesuree")
    ax.set_xlabel("Temps depuis l'activation (s)")
    ax.set_ylabel("Deplacement accompli (um)")
    ax.set_title("Deplacement depuis la position initiale\n"
                 "l'ecart entre plein et pointille mesure l'erreur du modele",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return fig


def figure_diagnostic(res, champ, p, plafond=30.0):
    # Rapport des vitesses et bilan chiffre.
    fig = plt.figure(figsize=(15, 6))
    gs = GridSpec(1, 2, figure=fig, wspace=0.28)
    fig.suptitle("Diagnostic de l'ecart au modele", fontsize=12,
                 fontweight='bold')
    prof = res['profil']

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(prof['r'] * 1e6, res['rapport'], 'o-', color=COUL_ALER, ms=6,
            lw=2, markeredgecolor='k', markeredgewidth=0.7)
    ax.axhline(1.0, color='k', ls='--', lw=1.8, label="accord parfait")
    ax.axhspan(0.5, 2.0, color='green', alpha=0.12, lw=0,
               label="accord a un facteur deux")
    if res['croisement'] is not None:
        ax.axvline(res['croisement'] * 1e6, color='k', ls=':', lw=1.6,
                   label=f"croisement, {res['croisement']*1e6:.0f} um")
    fini = np.isfinite(res['rapport'])
    hors = int(np.sum(res['rapport'][fini] > plafond))
    ax.set_ylim(0, plafond)
    if hors:
        ax.text(0.97, 0.94,
                f"{hors} points au dessus de {plafond:.0f},\n"
                f"jusqu'a {np.nanmax(res['rapport']):.0f}",
                transform=ax.transAxes, ha='right', va='top', fontsize=8.5,
                color=COUL_ALER,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    ax.set_xlabel("Position radiale (um)")
    ax.set_ylabel("Vitesse mesuree / vitesse predite")
    ax.set_title("Rapport des vitesses\n"
                 "un rapport constant signalerait une simple erreur "
                 "d'amplitude", fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    ax.axis('off')
    li = [["Champ employe", champ['source']],
          ["Champ maximal", f"{champ['e_max']:.3e} V/m"],
          ["Re[K] central", f"{p['re_k']:+.3f}"],
          ["Bande de sûreté",
           f"{p['re_k_min']:+.2f} a {p['re_k_max']:+.2f}"],
          ["Rayon median", f"{res['a_typique']*1e6:.2f} um"],
          ["Confinement", f"{res['confinement']:.3f}"],
          ["", ""],
          ["Deplacement mesure", f"{res['deplacement_mesure']*1e6:+.1f} um"],
          ["Deplacement simule", f"{res['deplacement_simule']*1e6:+.1f} um"],
          ["Bande de deplacement",
           f"{res['deplacement_haut']*1e6:+.1f} a "
           f"{res['deplacement_bas']*1e6:+.1f} um"],
          ["Mesures dans la bande",
           f"{res['recouvre']} sur {len(res['trajectoires'])}"],
          ["Ecart quadratique moyen", f"{res['ecart_moyen']*1e6:.1f} um"]]
    if res['croisement'] is not None:
        li.append(["Croisement des vitesses",
                   f"{res['croisement']*1e6:.0f} um"])
    fini = np.isfinite(res['rapport'])
    if np.any(fini):
        li += [["Rapport minimal", f"{np.nanmin(res['rapport']):.2f}"],
               ["Rapport maximal", f"{np.nanmax(res['rapport']):.0f}"],
               ["Rapport median", f"{np.nanmedian(res['rapport']):.2f}"]]
    t = ax.table(cellText=li, colLabels=["Grandeur", "Valeur"], loc='center',
                 cellLoc='left')
    t.auto_set_font_size(False)
    t.set_fontsize(9.5)
    t.scale(1, 1.6)
    ax.set_title("Bilan chiffre", fontsize=11, fontweight='bold', y=1.04)
    return fig


# =============================================================================
# EXPORT
# =============================================================================

def exporter(dossier, cellules, res, champ, p):
    prof = res['profil']
    np.savetxt(os.path.join(dossier, "confrontation.csv"),
               np.column_stack([prof['r'], prof['v'], prof['ecart'],
                                prof['n'], res['v_predite'], res['rapport']]),
               delimiter=',',
               header="r_m,v_mesuree_m_par_s,ecart_type,n_points,"
                      "v_predite_m_par_s,rapport",
               comments='')
    print(f"[ENREGISTRE] {os.path.join(dossier, 'confrontation.csv')}")

    lignes = []
    for t in res['trajectoires']:
        for tt, rm, rs in zip(t['t'], t['r_mes'], t['r_sim']):
            lignes.append(dict(cellule=t['nom'], temps_s=tt,
                               r_mesure_um=rm * 1e6, r_simule_um=rs * 1e6))
    pd.DataFrame(lignes).to_csv(
        os.path.join(dossier, "trajectoires.csv"), index=False)

    fini = np.isfinite(res['rapport'])
    meta = dict(
        origine="Confrontation_Modele_Direct.py",
        champ=dict(source=champ['source'], e_max_V_par_m=champ['e_max']),
        parametres=dict(re_k_central=p['re_k'],
                        re_k_bande=[p['re_k_min'], p['re_k_max']],
                        rayon_median_um=res['a_typique'] * 1e6,
                        confinement=res['confinement'],
                        gap_um=p['gap'] * 1e6,
                        sigma_milieu=p['sigma_milieu']),
        cellules=dict(nombre=len(cellules),
                      noms=[s['nom'] for s in cellules],
                      departs_um=[s['r0'] * 1e6 for s in cellules],
                      rayons_um=[s['rayon'] * 1e6 for s in cellules]),
        vitesse=dict(
            plage_mesuree_um=[float(prof['r'].min() * 1e6),
                              float(prof['r'].max() * 1e6)],
            v_mesuree_max_um_s=float(prof['v'].max() * 1e6),
            v_predite_max_um_s=float(np.max(res['v_predite']) * 1e6),
            rapport_min=float(np.nanmin(res['rapport'][fini])),
            rapport_max=float(np.nanmax(res['rapport'][fini])),
            rapport_median=float(np.nanmedian(res['rapport'][fini])),
            croisement_um=(res['croisement'] * 1e6
                           if res['croisement'] else None)),
        position=dict(
            deplacement_mesure_um=res['deplacement_mesure'] * 1e6,
            deplacement_simule_um=res['deplacement_simule'] * 1e6,
            deplacement_borne_basse_um=res['deplacement_bas'] * 1e6,
            deplacement_borne_haute_um=res['deplacement_haut'] * 1e6,
            cellules_dans_la_bande=res['recouvre'],
            ecart_borne_favorable_um=res['ecart_favorable'] * 1e6,
            ecart_quadratique_moyen_um=res['ecart_moyen'] * 1e6,
            arrivees_mesurees_um=[float(t['r_mes'][-1] * 1e6)
                                  for t in res['trajectoires']],
            arrivees_simulees_um=[float(t['r_sim'][-1] * 1e6)
                                  for t in res['trajectoires']]))
    with open(os.path.join(dossier, "bilan.json"), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[ENREGISTRE] {os.path.join(dossier, 'bilan.json')}")


# =============================================================================
# INTERFACE DE SAISIE
# =============================================================================

_racine = None


def racine_tk():
    global _racine
    if _racine is None:
        _racine = tk.Tk()
        _racine.withdraw()
    return _racine


def saisir():
    racine = racine_tk()
    res = {'p': None}
    fen = tk.Toplevel(racine)
    fen.title("Confrontation au modele direct")
    fen.resizable(False, False)
    fen.lift(); fen.focus_force(); fen.grab_set()
    ttk.Label(fen, text="CONFRONTATION DU MODELE DIRECT AUX MESURES",
              font=('Arial', 12, 'bold')).grid(row=0, column=0, columnspan=2,
                                               pady=(14, 10))
    cadre = ttk.Frame(fen, padding="16")
    cadre.grid(row=1, column=0, columnspan=2)
    ch = {}
    entrees = [
        ('re_k', "Re[K] central", RE_K_IMPOSE),
        ('re_k_min', "Borne basse de la bande", RE_K_BANDE_MIN),
        ('re_k_max', "Borne haute de la bande", RE_K_BANDE_MAX),
        ('r_cellule', "Rayon de repli (um)", RAYON_CELLULE_UM),
        ('r_int', "Rayon interieur (um)", R_INT_UM),
        ('r_ext', "Rayon exterieur (um)", R_EXT_UM),
        ('gap', "Hauteur de cavite (um)", HAUTEUR_CAV_UM),
        ('eps_milieu', "Permittivite du milieu", EPS_MILIEU_REL),
        ('sigma_milieu', "Conductivite (S/m)", SIGMA_MILIEU),
        ('viscosite', "Viscosite (mPa.s)", VISCOSITE_MPA_S),
        ('seuil', "Seuil d'activation (um/s)", SEUIL_ACTIVATION_UM_S),
        ('bandes', "Nombre de bandes", NB_BANDES)]
    for i, (cle, lib, dft) in enumerate(entrees):
        ttk.Label(cadre, text=lib).grid(row=i, column=0, sticky='w', pady=3)
        e = ttk.Entry(cadre, width=14)
        e.insert(0, str(dft))
        e.grid(row=i, column=1, sticky='w', padx=(12, 0), pady=3)
        ch[cle] = e
    li = len(entrees)
    ttk.Label(cadre, text="Source du rayon").grid(row=li, column=0, sticky='w',
                                                  pady=3)
    v_src = tk.StringVar(master=fen, value=SOURCE_RAYON)
    ttk.Combobox(cadre, textvariable=v_src, values=["fichier", "impose"],
                 width=11, state="readonly").grid(row=li, column=1, sticky='w',
                                                  padx=(12, 0), pady=3)

    bt = ttk.Frame(fen)
    bt.grid(row=2, column=0, columnspan=2, pady=(8, 16))

    def valider():
        try:
            res['p'] = dict(
                re_k=float(ch['re_k'].get()),
                re_k_min=float(ch['re_k_min'].get()),
                re_k_max=float(ch['re_k_max'].get()),
                mode_bande=MODE_BANDE,
                r_cellule=float(ch['r_cellule'].get()) * 1e-6,
                source_rayon=v_src.get(),
                r_int=float(ch['r_int'].get()) * 1e-6,
                r_ext=float(ch['r_ext'].get()) * 1e-6,
                gap=float(ch['gap'].get()) * 1e-6,
                eps_milieu=float(ch['eps_milieu'].get()) * EPSILON_0,
                eps_milieu_rel=float(ch['eps_milieu'].get()),
                sigma_milieu=float(ch['sigma_milieu'].get()),
                viscosite=float(ch['viscosite'].get()) * 1e-3,
                seuil=float(ch['seuil'].get()) * 1e-6,
                bandes=int(ch['bandes'].get()))
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


def principal():
    print("=" * 72)
    print("CONFRONTATION DU MODELE DIRECT AUX TRAJECTOIRES MESUREES")
    print("=" * 72)
    if not TK_DISPO:
        return
    racine_tk()

    chemin_champ = filedialog.askopenfilename(
        title="Champ simule (champ_simule.csv)",
        filetypes=[("Fichiers CSV", "*.csv")])
    if not chemin_champ:
        print("[INFO] aucun champ choisi, arret.")
        return

    chemin_suivi = filedialog.askopenfilename(
        title="Trajectoires des cellules",
        filetypes=[("Fichiers de suivi", "*.xlsx *.xls *.csv"),
                   ("Tous les fichiers", "*.*")])
    if not chemin_suivi:
        print("[INFO] aucun suivi choisi, arret.")
        return

    p = saisir()
    if p is None:
        return

    dossier = filedialog.askdirectory(title="Dossier de sortie")
    if not dossier:
        dossier = os.path.dirname(os.path.abspath(chemin_suivi))
    print(f"[INFO] resultats dans : {dossier}")

    champ = charger_champ(chemin_champ)
    cellules = lire_cellules(chemin_suivi, p)
    if not cellules:
        return
    res = confronter(cellules, champ, p)

    print("\n[FIGURES]")
    for fig, nom in [
            (figure_vitesse_position(cellules, res, champ, p),
             "01-vitesse-position"),
            (figure_position_temps(res, p), "02-position-temps"),
            (figure_diagnostic(res, champ, p), "03-diagnostic")]:
        chemin = os.path.join(dossier, f"{nom}.svg")
        fig.savefig(chemin, format='svg', bbox_inches='tight',
                    facecolor='white')
        print(f"[ENREGISTRE] {chemin}")

    print("\n[EXPORT]")
    exporter(dossier, cellules, res, champ, p)

    print(f"\n[AFFICHAGE] fermer les fenetres pour terminer", flush=True)
    plt.show()
    print("\n" + "=" * 72)
    print("TERMINE.")
    print("=" * 72)


if __name__ == "__main__":
    principal()