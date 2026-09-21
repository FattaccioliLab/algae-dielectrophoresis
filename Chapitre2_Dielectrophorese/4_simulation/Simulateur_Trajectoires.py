# -*- coding: utf-8 -*-
# =============================================================================
# SIMULATEUR DE DEPLACEMENT DIELECTROPHORETIQUE
# Interface interactive, champ ideal et champ mesure, abaque et comparaison
# =============================================================================
#
# CE QUE FAIT LE PROGRAMME
#
#   Il integre le mouvement radial de particules dans deux champs distincts et
#   permet de les confronter a des trajectoires reelles.
#
#     CHAMP IDEAL    resolution de Laplace pour la geometrie et la tension
#                    declarees, c'est-a-dire le champ que le montage devrait
#                    produire.
#     CHAMP MESURE   profil deduit du deplacement de billes de reference, lu
#                    dans un fichier ou decrit par sa decroissance
#                    exponentielle depuis le bord de l'electrode.
#
#   L'ecart entre les deux n'est pas un defaut de mesure. Dans une cavite dont
#   le milieu conduit, le champ chauffe le liquide, ce qui cree des gradients
#   de conductivite et de permittivite, donc une force volumique et un
#   ecoulement. La dielectrophorese suit le GRADIENT du champ, qui s'eteint sur
#   une longueur de l'ordre de h sur pi. Un ecoulement est au contraire une
#   solution de Stokes dont la portee est celle de la cavite entiere. Le
#   mouvement observe loin de l'electrode est donc gouverne par l'ecoulement,
#   et le champ qu'on en deduit parait aplati.
#
#   References : Ramos, Morgan, Green, Castellanos, J. Phys. D 31, 2338 (1998) ;
#   Castellanos, Ramos, Gonzalez, Green, Morgan, J. Phys. D 36, 2584 (2003).
#
# -----------------------------------------------------------------------------
# ORGANISATION DE L'INTERFACE
# -----------------------------------------------------------------------------
#   A gauche  : parametres, choix des champs, diagnostics, boutons.
#   A droite  : quatre vues synchronisees par un curseur temporel.
#                 vue de dessus, positions des particules a l'instant choisi
#                 position radiale au cours du temps
#                 profil de force le long du rayon
#                 profil de vitesse le long du rayon
#
#   Trois fenetres secondaires s'ouvrent a la demande :
#                 abaque des temps d'atteinte
#                 comparaison a des trajectoires reelles
#                 champs et vitesses compares
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
from matplotlib.widgets import Slider

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    TK_DISPO = True
except ImportError:
    TK_DISPO = False
    print("[ALERTE] tkinter indisponible : les fonctions de calcul restent "
          "importables.")


# =============================================================================
# VALEURS PAR DEFAUT
# =============================================================================

# --- Geometrie, en micrometres -----------------------------------------------
# L'anneau est decrit par ses rayons interieur et exterieur, et non par un rayon
# median et une largeur : c'est ainsi qu'il est defini sur les masques de
# photolithographie, et cela evite toute ambiguite.
R_INT_UM        = 50.0
R_EXT_UM        = 140.0
HAUTEUR_CAV_UM  = 20.0
EPAISSEUR_OR_NM = 1000.0
Z_PARTICULE_UM  = 5.0

# --- Signal ------------------------------------------------------------------
# Un generateur affiche une tension crete a crete, alors que Laplace demande une
# amplitude. Pour 10 Vpp centres sur zero, l'amplitude vaut 5 V. Confondre les
# deux introduit un facteur deux sur le champ, donc quatre sur la force.
TENSION_AFFICHEE = 10.0
TENSION_EN_CRETE_A_CRETE = True
FREQUENCE_KHZ = 100.0
FORME_SIGNAL  = "creneau"

# --- Milieu ------------------------------------------------------------------
# Un TAP est un tampon dilue : sa masse volumique ne s'ecarte pas de celle de
# l'eau. Sa conductivite, en revanche, est elevee et c'est elle qui gouverne le
# facteur de Clausius-Mossotti.
EPS_MILIEU_REL  = 78.0
SIGMA_MILIEU    = 0.30
VISCOSITE_MPA_S = 1.0
RHO_MILIEU      = 1000.0

# --- Particule ---------------------------------------------------------------
# Pour une bille de polystyrene, la conductivite effective est dominee par la
# surface : sigma_p = 2 Ks / a, avec Ks de l'ordre du nanosiemens. Pour un rayon
# de 3 um et Ks = 1 nS, cela donne 6,7e-4 S/m. Les valeurs typiques d'une
# CELLULE seraient tout autres, de l'ordre de eps_r = 60 et sigma = 0,5 S/m, et
# conduiraient a une dielectrophorese positive.
RAYON_PART_UM = 3.0
EPS_PART_REL  = 2.55
SIGMA_PART    = 6.7e-4
RHO_PART      = 1050.0

# --- Champ mesure ------------------------------------------------------------
CHAMP_AMPLITUDE = 1.387e10     # carre du champ au bord, en V2/m2
CHAMP_LONGUEUR_UM = 19.4       # longueur de decroissance

# --- Simulation --------------------------------------------------------------
DUREE_S = 18.0
PAS_TEMPS_MS = 2.0
DEPARTS_UM = ("145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, "
              "205, 210, 215, 220, 225, 230, 235, 240, 245, 250")
CIBLES_UM = "160, 180, 200, 220, 240, 260"

# --- Maillage ----------------------------------------------------------------
NR = 500
NZ = 200
R_MAX_UM = 500.0

# --- Recalage temporel des trajectoires reelles ------------------------------
# Ecart au dela duquel deux instants de mise en mouvement sont attribues a des
# acquisitions differentes. Au sein d'une meme video le champ n'est applique
# qu'une fois, et les detections ne peuvent differer que d'une fraction de
# seconde ; un ecart de plusieurs secondes signale un changement de video.
ECART_MAX_ACQUISITION = 1.5
MODE_RECALAGE = "individuel"

# --- Affichage ---------------------------------------------------------------
R_AFFICHAGE_UM = 320.0

EPSILON_0 = 8.854187817e-12
GRAVITE   = 9.81

matplotlib.rcParams['figure.facecolor']  = 'white'
matplotlib.rcParams['savefig.facecolor'] = 'white'
matplotlib.rcParams['axes.facecolor']    = 'white'

COUL_IDEAL  = '#9467bd'
COUL_MESURE = '#1f77b4'
COUL_REEL   = 'magenta'
COUL_OR     = 'gold'
COUL_GRIS   = '#7f7f7f'

COLONNES_TEMPS = ['temps_s', 'time_s', 'temps', 'time', 't_s', 't']
COLONNES_RAYON = ['r_um', 'rayon_um', 'radius_um', 'r', 'rayon', 'distance_um']


# =============================================================================
# PHYSIQUE
# =============================================================================

def facteur_forme_onde(forme=FORME_SIGNAL):
    # Rapport entre la moyenne temporelle du carre du champ et le carre de la
    # valeur crete. Un demi pour une sinusoide, un pour un creneau : dans ce
    # dernier cas le carre du champ est constant a tout instant, et la
    # correction en valeur efficace, souvent appliquee mecaniquement, serait
    # fausse.
    if forme.lower().startswith("sin"):
        return 0.5
    if forme.lower().startswith("cre"):
        return 1.0
    raise ValueError("forme de signal inconnue")


def facteur_clausius_mossotti(p):
    # Partie reelle du facteur de Clausius-Mossotti, bornee entre -0,5 et 1.
    #
    # Elle est TOUJOURS recalculee a partir des permittivites et des
    # conductivites, jamais saisie a la main : cela supprime toute possibilite
    # d'incoherence entre la valeur employee pour la force et celle affichee.
    w = 2.0 * math.pi * p['frequence']
    ep = p['eps_part'] - 1j * p['sigma_part'] / w
    em = p['eps_milieu'] - 1j * p['sigma_milieu'] / w
    return float(np.real((ep - em) / (ep + 2.0 * em)))


def coefficient_trainee(rayon, hauteur, viscosite):
    # Trainee corrigee du confinement entre deux parois planes, modele de
    # Happel et Brenner. La correction ne depend que du rapport entre le rayon
    # de la particule et la demi-hauteur de cavite.
    g0 = 6.0 * math.pi * viscosite * rayon
    lam = rayon / (hauteur / 2.0)
    if lam < 0.95:
        den = (1.0 - 1.004 * lam + 0.418 * lam ** 3
               + 0.21 * lam ** 4 - 0.169 * lam ** 5)
        f = 1.0 / den
    else:
        f = 1.0 + (9.0 / 8.0) * lam + lam ** 3
    return g0 * f, f, lam


def prefacteur_dep(p, re_k):
    # 2 pi a^3 eps_m Re[K]. Le rayon eleve au cube est celui de la PARTICULE,
    # non sa position radiale : c'est une confusion classique.
    return 2.0 * math.pi * p['r_part'] ** 3 * p['eps_milieu'] * re_k


def diagnostic_ecoulements(p):
    # Ordres de grandeur des mecanismes qui concurrencent la dielectrophorese.
    # Lois d'echelle etablies par Castellanos et ses collaborateurs.
    s = p['sigma_milieu']; eps = p['eps_milieu']; V = p['amplitude']
    h = p['gap']; eta = p['viscosite']; f = p['frequence']
    k_th = 0.6
    dT = s * V ** 2 / (2.0 * k_th)
    u_et = 1e-3 * eps * dT * V ** 2 / (eta * h)
    c = s / 0.0150
    lam_D = 0.304e-9 / math.sqrt(max(c / 1000.0, 1e-12))
    f_rc = s * lam_D / (2.0 * math.pi * eps * h)
    amort = (f_rc / f) ** 2 if f > 0 else float('inf')
    u_grav = (p['rho_milieu'] * GRAVITE * 2.1e-4 * dT * h ** 2) / eta
    v_sed = (2.0 * p['r_part'] ** 2 * (p['rho_part'] - p['rho_milieu'])
             * GRAVITE / (9.0 * eta))
    return dict(dT=dT, u_et=u_et, f_rc=f_rc, amortissement=amort,
                u_grav=u_grav, v_sed=v_sed,
                t_sed=(h / v_sed if v_sed > 0 else float('inf')))


# =============================================================================
# CHAMPS
# =============================================================================

def champ_ideal(p, bavard=True):
    # Resolution de Laplace en coordonnees (r, z), assemblage vectorise.
    #
    # Remplir une matrice creuse element par element dans une double boucle
    # coute plusieurs secondes pour quatre-vingt mille points. La construction
    # par triplets ligne, colonne, valeur donne la meme matrice environ soixante
    # fois plus vite.
    #
    # Conditions aux limites : la contre-electrode d'ITO couvre TOUTE la face
    # superieure et y impose un potentiel nul ; l'anneau d'or n'occupe qu'une
    # couronne de la face inferieure, le reste etant du verre isolant traite en
    # Neumann. C'est cette dissymetrie qui rend le champ non uniforme.
    nr, nz = p['nr'], NZ
    r1 = np.linspace(0.0, p['r_max'], nr)
    z1 = np.linspace(0.0, p['gap'], nz)
    dr, dz = r1[1] - r1[0], z1[1] - z1[0]
    R, Z = np.meshgrid(r1, z1, indexing='ij')
    if bavard:
        print(f"  maillage {nr} x {nz}, dr = {dr*1e6:.3f} um", flush=True)

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
    L, C, Va = [], [], []
    b = np.zeros(n)

    def add(li, co, va):
        L.append(np.asarray(li).ravel()); C.append(np.asarray(co).ravel())
        Va.append(np.asarray(va).ravel())

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

    A = coo_matrix((np.concatenate(Va),
                    (np.concatenate(L), np.concatenate(C))),
                   shape=(n, n)).tocsr()
    V = spsolve(A, b).reshape(nr, nz)
    Er, Ez = np.gradient(-V, dr, dz)
    E2 = (Er ** 2 + Ez ** 2) * facteur_forme_onde(p['forme'])
    # Le gradient est calcule UNE FOIS ici, et non a chaque trajectoire : il ne
    # depend ni de la particule ni de sa position de depart.
    dE2, _ = np.gradient(E2, dr, dz, edge_order=2)
    j = int(np.argmin(np.abs(z1 - p['z'])))
    if bavard:
        print(f"  E maximal = {math.sqrt(np.max(E2)):.3e} V/m", flush=True)
    return dict(nom="ideal", r=r1, z=z1, V=V, E2_2d=E2, dE2_2d=dE2, j=j,
                e2=E2[:, j], gradient=dE2[:, j])


def champ_mesure(p, chemin=None):
    # Champ apparent, deduit du deplacement de billes de reference.
    #
    # Le gradient est obtenu analytiquement a partir du modele exponentiel, ce
    # qui evite d'introduire du bruit de derivation dans l'integration.
    r = np.linspace(p['r_ext'], p['r_max'], 3000)
    if chemin and os.path.exists(chemin):
        d = np.genfromtxt(chemin, delimiter=',', names=True)
        rf, ef = np.asarray(d['r_m']), np.asarray(d['E2_V2_par_m2'])
        ok = np.isfinite(ef) & (ef > 0)
        pp = np.polyfit(rf[ok] - p['r_ext'], np.log(ef[ok]), 1)
        longueur = -1.0 / pp[0]; amplitude = math.exp(pp[1])
        source = os.path.basename(chemin)
    else:
        amplitude = p['champ_amplitude']; longueur = p['champ_longueur']
        source = "parametres saisis"
    e2 = amplitude * np.exp(-(r - p['r_ext']) / longueur)
    grad = -amplitude / longueur * np.exp(-(r - p['r_ext']) / longueur)
    return dict(nom="mesure", r=r, e2=e2, gradient=grad, amplitude=amplitude,
                longueur=longueur, source=source,
                potentiel=math.sqrt(amplitude) * p['gap'])


def vitesse_profil(r, champ, pref, gamma):
    g = np.interp(r, champ['r'], champ['gradient'], left=0.0, right=0.0)
    return pref * g / gamma


# =============================================================================
# INTEGRATION
# =============================================================================

def integrer(r0, champ, pref, gamma, duree, dt, p):
    # Integration de dr/dt = v(r) par le schema de Crank-Nicolson.
    #
    # La vitesse moyenne entre le debut et la fin du pas est employee plutot que
    # celle du seul point de depart. Cela evite le saut brutal du premier pas,
    # la ou la vitesse varie le plus vite, et divise l'erreur par un ordre.
    n = int(duree / dt) + 1
    t = np.linspace(0.0, duree, n)
    r = np.zeros(n); r[0] = r0
    r_min, r_max = 1e-6, min(champ['r'].max(), p['r_max'])

    def v(x):
        return float(vitesse_profil(np.array([x]), champ, pref, gamma)[0])

    for i in range(n - 1):
        v1 = v(r[i])
        r_pred = min(max(r[i] + v1 * dt, r_min), r_max)
        v2 = v(r_pred)
        r_new = r[i] + 0.5 * (v1 + v2) * dt

        # En dielectrophorese positive, la particule est attiree vers l'anneau
        # et s'y arrete : elle ne traverse pas le metal.
        if pref > 0:
            if r[i] < p['r_int'] <= r_new:
                r[i + 1:] = p['r_int']; break
            if r[i] > p['r_ext'] >= r_new:
                r[i + 1:] = p['r_ext']; break
        r[i + 1] = min(max(r_new, r_min), r_max)
    return t, r


def simuler(champs, p, re_k, gamma, departs, bavard=True):
    pref = prefacteur_dep(p, re_k)
    res = {}
    for nom, ch in champs.items():
        res[nom] = [dict(r0=r0, **dict(zip(('t', 'r'),
                                           integrer(r0, ch, pref, gamma,
                                                    p['duree'], p['dt'], p))))
                    for r0 in departs]
    if bavard:
        print("\n" + "=" * 66)
        print("TRAJECTOIRES SIMULEES")
        print("=" * 66)
        entete = f"  {'depart':>9}"
        for nom in champs:
            entete += f" {'arrivee ' + nom:>18}"
        print(entete)
        print("  " + "-" * (9 + 19 * len(champs)))
        for k, r0 in enumerate(departs):
            ligne = f"  {r0*1e6:>8.0f}u"
            for nom in champs:
                ligne += f" {res[nom][k]['r'][-1]*1e6:>17.0f}u"
            print(ligne)
        print("=" * 66)
    return res


def abaque(champs, p, re_k, gamma, departs, cibles, bavard=True):
    # Temps mis pour atteindre chaque rayon cible, selon la position de depart.
    #
    # C'est la lecture la plus directe pour preparer une experience : elle dit
    # combien de temps appliquer le champ pour qu'une particule partie d'un
    # endroit donne atteigne un endroit voulu.
    pref = prefacteur_dep(p, re_k)
    tableaux = {}
    for nom, ch in champs.items():
        T = np.full((len(departs), len(cibles)), np.nan)
        for i, r0 in enumerate(departs):
            t, r = integrer(r0, ch, pref, gamma, p['duree'], p['dt'], p)
            for j, rc in enumerate(cibles):
                if (rc > r0 and np.max(r) >= rc):
                    T[i, j] = t[np.argmax(r >= rc)]
                elif (rc < r0 and np.min(r) <= rc):
                    T[i, j] = t[np.argmax(r <= rc)]
        tableaux[nom] = T
        if bavard:
            print(f"\n  abaque, champ {nom}, temps en secondes")
            e = f"  {'depart':>9}"
            for rc in cibles:
                e += f" {str(int(rc*1e6)) + 'u':>9}"
            print(e); print("  " + "-" * (9 + 10 * len(cibles)))
            for i, r0 in enumerate(departs):
                li = f"  {r0*1e6:>8.0f}u"
                for j in range(len(cibles)):
                    v = T[i, j]
                    li += f" {('-' if not np.isfinite(v) else f'{v:.1f}'):>9}"
                print(li)
    return dict(departs=np.array(departs), cibles=np.array(cibles),
                tableaux=tableaux)


# =============================================================================
# TRAJECTOIRES REELLES
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


def _lisser(t, r, duree=0.30, ordre=2):
    # Derivee lissee, employee uniquement pour reperer la mise en mouvement.
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
    from scipy.signal import savgol_filter
    return savgol_filter(r, f, ordre, deriv=1, delta=dt)


def regrouper_par_acquisition(detections, ecart_max=ECART_MAX_ACQUISITION):
    # Regroupe les billes par acquisition a partir de leurs seuls instants de
    # mise en mouvement.
    #
    # Les trajectoires proviennent souvent de plusieurs videos, chacune ayant
    # son propre instant de mise sous tension. Le regroupement ne repose pas
    # sur la numerotation des onglets, qui n'est qu'une convention, mais sur
    # les instants eux-memes : les detections sont triees, et une coupure est
    # placee partout ou l'ecart entre deux instants consecutifs depasse le
    # seuil. Au sein d'une meme video, deux billes ne peuvent differer que du
    # temps de reponse de la detection, soit une fraction de seconde.
    if not detections:
        return {}, {}
    cles = sorted(detections, key=lambda k: detections[k])
    groupes = {cles[0]: 0}
    numero = 0
    for prec, cour in zip(cles[:-1], cles[1:]):
        if detections[cour] - detections[prec] > ecart_max:
            numero += 1
        groupes[cour] = numero
    instants = {}
    for g in set(groupes.values()):
        instants[g] = float(np.median([detections[k] for k in groupes
                                       if groupes[k] == g]))
    return groupes, instants


def lire_suivi(chemin, seuil_um_s=5.0, mode=None):
    # Lit un classeur de suivi et recale chaque trajectoire sur l'instant ou le
    # champ lui a ete applique.
    #
    # Deux modes de recalage, comme dans le programme de reconstruction :
    #
    #   "individuel"  chaque bille est recalee sur SA propre mise en mouvement,
    #                 de sorte que toutes les courbes partent exactement de
    #                 zero. La latence de detection, qui depend de la vitesse
    #                 locale et donc de la position, ne vient plus decaler les
    #                 courbes les unes par rapport aux autres.
    #   "acquisition" toutes les billes d'une meme video partagent l'instant
    #                 median du groupe, ce qui est plus fidele a la physique
    #                 mais fait demarrer les billes lentes en retard.
    if mode is None:
        mode = MODE_RECALAGE
    cl = pd.ExcelFile(chemin)
    brutes, detections = [], {}

    for k, ong in enumerate(cl.sheet_names):
        df = cl.parse(ong)
        if df.empty or len(df) < 8:
            continue
        cr = _colonne(df.columns, COLONNES_RAYON)
        if cr is None:
            continue
        ct = _colonne(df.columns, COLONNES_TEMPS)
        r = pd.to_numeric(df[cr], errors='coerce').values * 1e-6
        t = (pd.to_numeric(df[ct], errors='coerce').values if ct is not None
             else np.arange(len(df)) / 20.0)
        ok = np.isfinite(r) & np.isfinite(t)
        r, t = r[ok], t[ok]
        if len(r) < 8:
            continue
        v = _lisser(t, r)
        dep = np.where(np.abs(v) > seuil_um_s * 1e-6)[0]
        # Le seuil doit etre franchi sur trois points consecutifs, faute de quoi
        # une fluctuation isolee suffirait a declencher la detection.
        indice = None
        for i in dep:
            if i + 2 < len(v) and np.all(np.abs(v[i:i + 3]) > seuil_um_s * 1e-6):
                indice = i
                break
        brutes.append(dict(nom=ong, t=t, r=r, cle=k))
        if indice is not None:
            detections[k] = float(t[indice])

    if not brutes:
        return []

    groupes, instants = regrouper_par_acquisition(detections)
    print(f"  {len(instants)} acquisition(s) detectee(s)")
    for g in sorted(instants):
        membres = [k for k in groupes if groupes[k] == g]
        print(f"    acquisition {g + 1} : activation a {instants[g]:.3f} s, "
              f"{len(membres)} billes")

    sorties = []
    for b in brutes:
        k = b['cle']
        if k in detections:
            t_ref = detections[k] if mode == "individuel" else instants[groupes[k]]
            acq = groupes[k]
        elif instants:
            # Bille sans detection propre : rattachee a l'acquisition la plus
            # proche en temps plutot que d'etre perdue.
            acq = min(instants, key=lambda g: abs(instants[g]
                                                  - float(np.median(b['t']))))
            t_ref = instants[acq]
        else:
            continue
        m = b['t'] >= t_ref
        if int(np.sum(m)) < 8:
            continue
        sorties.append(dict(nom=b['nom'], t=b['t'][m] - t_ref, r=b['r'][m],
                            r0=float(b['r'][m][0]), acquisition=acq,
                            activation=t_ref))

    if sorties:
        departs = [s_['t'][0] for s_ in sorties]
        print(f"  {len(sorties)} trajectoires recalees, premier point de "
              f"{min(departs):+.3f} a {max(departs):+.3f} s")
    return sorties


# =============================================================================
# TRACES SECONDAIRES
# =============================================================================

def _afficher(figure):
    # Ouvre une figure dans sa propre fenetre.
    #
    # La methode show d'une figure ne fonctionne pas de facon fiable quand une
    # autre figure est deja embarquee dans une interface Tk : le gestionnaire
    # de fenetres n'est alors pas celui que matplotlib attend. Passer par
    # plt.show sans blocage evite ce piege.
    try:
        figure.canvas.manager.show()
    except Exception:
        plt.show(block=False)
    plt.pause(0.001)
    return figure


def _coul(nom):
    return {'ideal': COUL_IDEAL, 'mesure': COUL_MESURE}.get(nom, COUL_GRIS)


def _style(nom):
    return {'ideal': '--', 'mesure': '-'}.get(nom, '-')


def _anneau_v(ax, p, etiquette=True):
    # L'etiquette n'est posee qu'une fois par axe, faute de quoi la legende
    # accumulerait autant d'entrees identiques que d'appels.
    deja = any(getattr(a, 'get_label', lambda: '')() == "anneau d'or"
               for a in ax.get_children())
    ax.axvspan(p['r_int'] * 1e6, p['r_ext'] * 1e6, alpha=0.22, color=COUL_OR,
               label=("anneau d'or" if etiquette and not deja else None))


def _anneau_h(ax, p):
    ax.axhspan(p['r_int'] * 1e6, p['r_ext'] * 1e6, alpha=0.22, color=COUL_OR)


def figure_abaque(res, ab, champs, p):
    # Abaque de deplacement : position radiale au cours du temps, une courbe
    # par position de depart.
    #
    # C'est la lecture la plus directe pour preparer une experience. On y suit
    # d'un coup d'oeil ou se trouve chaque particule a chaque instant, selon
    # l'endroit d'ou elle est partie. Le faisceau de courbes montre aussi si
    # les trajectoires convergent, ce qui signalerait une position d'equilibre,
    # ou si elles restent ordonnees.
    n = len(champs)
    fig = plt.figure(figsize=(9 * n, 11))
    gs = GridSpec(2, n, figure=fig, hspace=0.30, wspace=0.26)
    fig.suptitle(
        f"Abaque de deplacement, particule de {p['r_part']*1e6:.1f} um, "
        f"{p['duree']:.0f} s", fontsize=13, fontweight='bold')

    couleurs = plt.cm.viridis(np.linspace(0, 0.92, len(p['departs'])))

    for k, nom in enumerate(champs):
        # Ligne du haut : le faisceau de trajectoires
        ax = fig.add_subplot(gs[0, k])
        nommer = len(res[nom]) <= 8
        for s_, c in zip(res[nom], couleurs):
            ax.plot(s_['t'], s_['r'] * 1e6, '-', color=c, lw=2.2,
                    label=(f"depart {s_['r0']*1e6:.0f} um" if nommer else None))
            ax.plot([0], [s_['r0'] * 1e6], 'o', color=c, ms=7,
                    markeredgecolor='k', markeredgewidth=0.8, zorder=5)
            ax.plot([s_['t'][-1]], [s_['r'][-1] * 1e6], 's', color=c, ms=7,
                    markeredgecolor='k', markeredgewidth=0.8, zorder=5)
        _anneau_h(ax, p)
        ax.set_xlim(0, p['duree'])
        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("Position radiale (um)")
        ax.set_title(f"Champ {nom}\n"
                     "rond : depart, carre : arrivee", fontsize=11,
                     fontweight='bold')
        # Au dela d'une dizaine de trajectoires, une legende nominative
        # occuperait la moitie du cadre. Seules les extremes sont nommees, le
        # dégradé de couleur suffisant a ordonner les autres.
        if len(res[nom]) <= 8:
            ax.legend(fontsize=7, ncol=2, loc='best')
        else:
            ax.plot([], [], '-', color=couleurs[0], lw=2.2,
                    label=f"depart {res[nom][0]['r0']*1e6:.0f} um")
            ax.plot([], [], '-', color=couleurs[-1], lw=2.2,
                    label=f"depart {res[nom][-1]['r0']*1e6:.0f} um")
            ax.legend(fontsize=8, loc='lower right',
                      title=f"{len(res[nom])} departs")
        ax.grid(alpha=0.3)

        # Ligne du bas : deplacement accompli depuis le depart
        ax = fig.add_subplot(gs[1, k])
        nommer = len(res[nom]) <= 8
        for s_, c in zip(res[nom], couleurs):
            ax.plot(s_['t'], (s_['r'] - s_['r0']) * 1e6, '-', color=c, lw=2.2,
                    label=(f"depart {s_['r0']*1e6:.0f} um" if nommer else None))
        ax.axhline(0, color='k', lw=0.9)
        ax.set_xlim(0, p['duree'])
        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("Deplacement accompli (um)")
        ax.set_title(f"Deplacement depuis le depart, champ {nom}\n"
                     "l'ecart entre les courbes mesure la selectivite en "
                     "position", fontsize=11, fontweight='bold')
        if len(res[nom]) <= 8:
            ax.legend(fontsize=7, ncol=2)
        else:
            ax.plot([], [], '-', color=couleurs[0], lw=2.2,
                    label=f"depart {res[nom][0]['r0']*1e6:.0f} um")
            ax.plot([], [], '-', color=couleurs[-1], lw=2.2,
                    label=f"depart {res[nom][-1]['r0']*1e6:.0f} um")
            ax.legend(fontsize=8, loc='upper left',
                      title=f"{len(res[nom])} departs")
        ax.grid(alpha=0.3)

    return fig


def figure_temps_atteinte(ab, champs, p):
    # Temps mis pour atteindre chaque rayon cible, en courbes et en carte.
    #
    # Cette lecture complete la precedente : au lieu de suivre la position dans
    # le temps, elle repond directement a la question inverse, combien de temps
    # faut-il pour aller de tel endroit a tel autre.
    n = len(champs)
    fig = plt.figure(figsize=(8 * n, 10))
    gs = GridSpec(2, n, figure=fig, hspace=0.32, wspace=0.28)
    fig.suptitle("Temps necessaire pour atteindre un rayon donne", fontsize=12,
                 fontweight='bold')
    for k, nom in enumerate(champs):
        T = ab['tableaux'][nom]
        ax = fig.add_subplot(gs[0, k])
        for j, rc in enumerate(ab['cibles']):
            col = T[:, j]; ok = np.isfinite(col)
            if np.any(ok):
                ax.plot(ab['departs'][ok] * 1e6, col[ok], 'o-', ms=5, lw=1.8,
                        label=f"{rc*1e6:.0f} um")
        ax.set_xlabel("Position de depart (um)")
        ax.set_ylabel("Temps necessaire (s)")
        ax.set_title(f"Champ {nom}", fontsize=11, fontweight='bold')
        ax.legend(fontsize=7, ncol=2, title="atteindre")
        ax.grid(alpha=0.3)

        ax = fig.add_subplot(gs[1, k])
        im = ax.imshow(np.ma.masked_invalid(T), aspect='auto', origin='lower',
                       cmap='viridis',
                       extent=[ab['cibles'][0] * 1e6, ab['cibles'][-1] * 1e6,
                               ab['departs'][0] * 1e6, ab['departs'][-1] * 1e6])
        fig.colorbar(im, ax=ax, label="Temps (s)")
        ax.set_xlabel("Rayon a atteindre (um)")
        ax.set_ylabel("Position de depart (um)")
        ax.set_title(f"Carte des temps, champ {nom}\n"
                     "les cases vides sont hors de portee", fontsize=11,
                     fontweight='bold')
    return fig


def figure_champs(champs, p, re_k, gamma, diag):
    fig = plt.figure(figsize=(19, 6))
    gs = GridSpec(1, 3, figure=fig, wspace=0.30)
    fig.suptitle("Champs employes, vitesses et diagnostics", fontsize=12,
                 fontweight='bold')

    ax = fig.add_subplot(gs[0, 0])
    for nom, ch in champs.items():
        ax.plot(ch['r'] * 1e6, ch['e2'], _style(nom), color=_coul(nom), lw=2.4,
                label=f"champ {nom}")
    _anneau_v(ax, p)
    ax.set_xlim(p['r_ext'] * 1e6 * 0.9, R_AFFICHAGE_UM)
    hauts = []
    for ch in champs.values():
        m = ch['r'] >= p['r_ext'] * 1.12
        if not np.any(m):
            m = np.ones_like(ch['r'], dtype=bool)
        hauts.append(float(np.percentile(ch['e2'][m], 99)))
    haut = max(hauts) if hauts else 1.0
    ax.set_ylim(0, (haut if haut > 0 else 1.0) * 1.4)
    ax.set_xlabel("Position radiale (um)")
    ax.set_ylabel("Carre du champ (V2/m2)")
    ax.set_title("Carre du champ", fontsize=11, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    pref = prefacteur_dep(p, re_k)
    rr = np.linspace(p['r_ext'], R_AFFICHAGE_UM * 1e-6, 900)
    vals = []
    for nom, ch in champs.items():
        v = vitesse_profil(rr, ch, pref, gamma) * 1e6
        vals.append(v)
        ax.plot(rr * 1e6, v, _style(nom), color=_coul(nom), lw=2.4,
                label=f"champ {nom}")
    ax.axhline(0, color='k', lw=0.9)
    _anneau_v(ax, p)
    ax.set_xlim(p['r_ext'] * 1e6 * 0.9, R_AFFICHAGE_UM)
    # Le pic au bord de l'electrode depasse de plusieurs ordres de grandeur les
    # vitesses observees ailleurs. L'axe est donc borne sur la plage lisible et
    # le depassement annonce, plutot que de laisser toutes les courbes plaquees
    # contre l'axe.
    m_utile = rr > p['r_ext'] * 1.12
    if not np.any(m_utile):
        m_utile = np.ones_like(rr, dtype=bool)
    h = max(float(np.percentile(np.abs(v[m_utile]), 99)) for v in vals)
    h = h if h > 0 else 1.0
    ax.set_ylim(-1.35 * h if re_k > 0 else -0.15 * h,
                0.15 * h if re_k > 0 else 1.35 * h)
    pic = max(float(np.max(np.abs(v))) for v in vals)
    if pic > 1.35 * h:
        ax.text(0.97, 0.94, f"pic au bord : {pic:.0f} um/s", ha='right',
                va='top', transform=ax.transAxes, fontsize=8.5,
                color=COUL_GRIS,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    ax.set_xlabel("Position radiale (um)")
    ax.set_ylabel("Vitesse radiale (um/s)")
    ax.set_title(f"Vitesse d'une particule de {p['r_part']*1e6:.1f} um",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 2]); ax.axis('off')
    li = [["Re[K] calcule", f"{re_k:+.4f}"],
          ["Regime", "positive" if re_k > 0 else "negative"],
          ["Trainee corrigee", f"{gamma:.3e} N.s/m"],
          ["Amplitude imposee", f"{p['amplitude']:.2f} V"],
          ["Echauffement Joule", f"{diag['dT']:.2f} K"],
          ["Vitesse electrothermique", f"{diag['u_et']*1e6:.0f} um/s"],
          ["Amortissement electro-osmose", f"{diag['amortissement']:.1e}"],
          ["Sedimentation", f"{diag['v_sed']*1e6:.2f} um/s"],
          ["Traversee par sedimentation", f"{diag['t_sed']:.0f} s"]]
    if 'mesure' in champs:
        li += [["Longueur du champ mesure",
                f"{champs['mesure']['longueur']*1e6:.1f} um"],
               ["Longueur ideale, h / pi", f"{p['gap']/math.pi*1e6:.1f} um"],
               ["Potentiel implique",
                f"{champs['mesure']['potentiel']:.2f} V"]]
    t = ax.table(cellText=li, colLabels=["Grandeur", "Valeur"], loc='center',
                 cellLoc='left')
    t.auto_set_font_size(False); t.set_fontsize(9.5); t.scale(1, 1.7)
    ax.set_title("Recapitulatif", fontsize=11, fontweight='bold', pad=20)
    return fig


def figure_comparaison(champs, suivi, p, re_k, gamma, res):
    fig = plt.figure(figsize=(19, 6))
    gs = GridSpec(1, 3, figure=fig, wspace=0.28)
    fig.suptitle("Comparaison aux trajectoires reelles", fontsize=12,
                 fontweight='bold')
    pref = prefacteur_dep(p, re_k)

    ax = fig.add_subplot(gs[0, 0])
    for k, s in enumerate(suivi):
        ax.plot(s['t'], s['r'] * 1e6, '-', color=COUL_REEL, lw=1.6, alpha=0.75,
                label="mesure" if k == 0 else None)
    for nom, traj in res.items():
        for k, s in enumerate(traj):
            ax.plot(s['t'], s['r'] * 1e6, _style(nom), color=_coul(nom), lw=1.5,
                    alpha=0.7, label=f"champ {nom}" if k == 0 else None)
    _anneau_h(ax, p)
    ax.set_xlabel("Temps (s)"); ax.set_ylabel("Position radiale (um)")
    ax.set_title("Vue d'ensemble", fontsize=11, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Chaque particule reelle est reintegree depuis SA position de depart, ce
    # qui rend la comparaison individuelle et non plus statistique.
    ax = fig.add_subplot(gs[0, 1])
    coul = plt.cm.tab10(np.linspace(0, 1, max(len(suivi), 1)))
    for s, c in zip(suivi, coul):
        duree = float(s['t'].max())
        ax.plot(s['t'], s['r'] * 1e6, '-', color=c, lw=2, alpha=0.9)
        for nom, ch in champs.items():
            t, r = integrer(s['r0'], ch, pref, gamma, duree, p['dt'], p)
            ax.plot(t, r * 1e6, _style(nom), color=c, lw=1.3, alpha=0.6)
    ax.plot([], [], '-', color='k', lw=2, label="mesure")
    for nom in champs:
        ax.plot([], [], _style(nom), color='k', lw=1.3, label=f"simule, {nom}")
    _anneau_h(ax, p)
    ax.set_xlabel("Temps (s)"); ax.set_ylabel("Position radiale (um)")
    ax.set_title("Particule par particule, depuis le meme depart", fontsize=11,
                 fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    noms = [s['nom'] for s in suivi]
    x = np.arange(len(suivi))
    mes = np.array([s['r'][-1] for s in suivi]) * 1e6
    larg = 0.8 / (len(champs) + 1)
    ax.bar(x - larg, mes, larg, color=COUL_REEL, label="mesuree")
    for i, (nom, ch) in enumerate(champs.items()):
        arr = []
        for s in suivi:
            _, r = integrer(s['r0'], ch, pref, gamma, float(s['t'].max()),
                            p['dt'], p)
            arr.append(r[-1] * 1e6)
        ax.bar(x + i * larg, arr, larg, color=_coul(nom),
               label=f"simulee, {nom}")
    ax.set_xticks(x); ax.set_xticklabels(noms, rotation=25, ha='right')
    ax.set_ylabel("Position radiale finale (um)")
    ax.set_title("Arrivees", fontsize=11, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')
    return fig


# =============================================================================
# INTERFACE
# =============================================================================

class Application:
    # Interface interactive : panneau de controle a gauche, quatre vues
    # synchronisees par un curseur temporel a droite.

    def __init__(self, racine):
        self.racine = racine
        self.champs = {}
        self.res = None
        self.suivi = []
        self.p = None
        self.re_k = None
        self.gamma = None
        self.diag = None
        self.chemin_champ = None
        self.dossier = os.getcwd()
        self.curseur = None
        self.lignes_radiales = []

        racine.title("Simulation dielectrophoretique")
        racine.geometry("1500x820")

        self._construire_controles()
        self._construire_graphiques()
        self._rafraichir_rek()

    # ---------------------------------------------------------------- controles
    def _construire_controles(self):
        cadre = tk.Frame(self.racine, bg='#ECECEC', padx=12, pady=10)
        cadre.pack(side=tk.LEFT, fill=tk.Y)
        canevas = tk.Canvas(cadre, bg='#ECECEC', width=290,
                            highlightthickness=0)
        barre = ttk.Scrollbar(cadre, orient="vertical", command=canevas.yview)
        interieur = tk.Frame(canevas, bg='#ECECEC')
        interieur.bind("<Configure>", lambda e: canevas.configure(
            scrollregion=canevas.bbox("all")))
        canevas.create_window((0, 0), window=interieur, anchor="nw")
        canevas.configure(yscrollcommand=barre.set)
        canevas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        barre.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Label(interieur, text="PARAMETRES", font=('Arial', 12, 'bold'),
                 bg='#ECECEC').pack(pady=(0, 8))

        self.champs_saisie = {}

        def bloc(titre, entrees):
            ttk.Separator(interieur, orient='horizontal').pack(fill='x', pady=4)
            tk.Label(interieur, text=titre, font=('Arial', 9, 'bold'),
                     bg='#ECECEC').pack(anchor='w')
            for cle, lib, dft in entrees:
                f = tk.Frame(interieur, bg='#ECECEC'); f.pack(fill='x', pady=1)
                tk.Label(f, text=lib, font=('Arial', 8), bg='#ECECEC',
                         width=24, anchor='w').pack(side=tk.LEFT)
                e = tk.Entry(f, font=('Arial', 9), width=10)
                e.insert(0, str(dft)); e.pack(side=tk.LEFT)
                e.bind("<FocusOut>", lambda ev: self._rafraichir_rek())
                self.champs_saisie[cle] = e

        bloc("Particule", [
            ('r_part', "Rayon (um)", RAYON_PART_UM),
            ('eps_part', "Permittivite relative", EPS_PART_REL),
            ('sigma_part', "Conductivite (S/m)", SIGMA_PART),
            ('rho_part', "Masse volumique (kg/m3)", RHO_PART)])
        bloc("Milieu", [
            ('eps_milieu', "Permittivite relative", EPS_MILIEU_REL),
            ('sigma_milieu', "Conductivite (S/m)", SIGMA_MILIEU),
            ('viscosite', "Viscosite (mPa.s)", VISCOSITE_MPA_S),
            ('rho_milieu', "Masse volumique (kg/m3)", RHO_MILIEU)])
        bloc("Geometrie", [
            ('r_int', "Rayon interieur (um)", R_INT_UM),
            ('r_ext', "Rayon exterieur (um)", R_EXT_UM),
            ('gap', "Hauteur de cavite (um)", HAUTEUR_CAV_UM),
            ('z', "Hauteur de la particule (um)", Z_PARTICULE_UM)])
        bloc("Signal", [
            ('tension', "Tension affichee (V)", TENSION_AFFICHEE),
            ('frequence', "Frequence (kHz)", FREQUENCE_KHZ)])

        f = tk.Frame(interieur, bg='#ECECEC'); f.pack(fill='x', pady=1)
        tk.Label(f, text="Convention", font=('Arial', 8), bg='#ECECEC',
                 width=24, anchor='w').pack(side=tk.LEFT)
        self.var_cc = tk.StringVar(master=self.racine,
                                   value="crete a crete"
                                   if TENSION_EN_CRETE_A_CRETE else "amplitude")
        ttk.Combobox(f, textvariable=self.var_cc, width=8, state="readonly",
                     values=["amplitude", "crete a crete"]).pack(side=tk.LEFT)
        f = tk.Frame(interieur, bg='#ECECEC'); f.pack(fill='x', pady=1)
        tk.Label(f, text="Forme du signal", font=('Arial', 8), bg='#ECECEC',
                 width=24, anchor='w').pack(side=tk.LEFT)
        self.var_forme = tk.StringVar(master=self.racine, value=FORME_SIGNAL)
        ttk.Combobox(f, textvariable=self.var_forme, width=8, state="readonly",
                     values=["creneau", "sinus"]).pack(side=tk.LEFT)

        bloc("Champ mesure", [
            ('champ_amp', "Carre du champ au bord", CHAMP_AMPLITUDE),
            ('champ_lon', "Longueur (um)", CHAMP_LONGUEUR_UM)])
        bloc("Simulation", [
            ('duree', "Duree (s)", DUREE_S),
            ('dt', "Pas de temps (ms)", PAS_TEMPS_MS),
            ('departs', "Departs (um)", DEPARTS_UM),
            ('cibles', "Cibles (um)", CIBLES_UM)])

        ttk.Separator(interieur, orient='horizontal').pack(fill='x', pady=6)
        tk.Label(interieur, text="Champs a simuler", font=('Arial', 9, 'bold'),
                 bg='#ECECEC').pack(anchor='w')
        self.var_ideal = tk.BooleanVar(master=self.racine, value=True)
        self.var_mes = tk.BooleanVar(master=self.racine, value=True)
        tk.Checkbutton(interieur, text="ideal, Laplace", variable=self.var_ideal,
                       bg='#ECECEC', font=('Arial', 8)).pack(anchor='w')
        tk.Checkbutton(interieur, text="mesure, billes de reference",
                       variable=self.var_mes, bg='#ECECEC',
                       font=('Arial', 8)).pack(anchor='w')

        # Facteur de Clausius-Mossotti, recalcule et non saisi
        ttk.Separator(interieur, orient='horizontal').pack(fill='x', pady=6)
        self.label_rek = tk.Label(interieur, text="", font=('Courier', 9),
                                  bg='white', relief='sunken', width=34,
                                  height=3, justify='left', anchor='w')
        self.label_rek.pack(pady=4)

        cadre_diag = tk.LabelFrame(interieur, text="Diagnostics",
                                   font=('Arial', 9, 'bold'), bg='#ECECEC')
        cadre_diag.pack(fill='x', pady=6)
        self.texte_diag = tk.Text(cadre_diag, height=13, width=36,
                                  font=('Courier', 8), bg='white')
        self.texte_diag.pack(padx=4, pady=4)
        self.texte_diag.insert('1.0', "Cliquer sur CALCULER")
        self.texte_diag.config(state='disabled')

        for texte, cmd, coul in [
                ("CALCULER", self.calculer, '#2E7D32'),
                ("Abaque de deplacement", self.ouvrir_abaque, '#1565C0'),
                ("Champs et vitesses", self.ouvrir_champs, '#1565C0'),
                ("Charger un suivi", self.charger_suivi, '#6A1B9A'),
                ("Charger un champ mesure", self.charger_champ, '#6A1B9A'),
                ("Exporter", self.exporter, '#EF6C00')]:
            tk.Button(interieur, text=texte, command=cmd, bg=coul, fg='white',
                      font=('Arial', 9, 'bold'), width=26).pack(pady=2)

    # ------------------------------------------------------------- graphiques
    def _construire_graphiques(self):
        cadre = tk.Frame(self.racine)
        cadre.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.fig = plt.figure(figsize=(13, 8))
        gs = GridSpec(3, 2, figure=self.fig, height_ratios=[0.06, 1, 1],
                      hspace=0.35, wspace=0.24, left=0.08, right=0.96,
                      top=0.95, bottom=0.07)
        ax_s = self.fig.add_subplot(gs[0, :])
        self.slider = Slider(ax_s, "Temps (s)", 0, DUREE_S, valinit=0,
                             color='steelblue', alpha=0.8)
        self.slider.on_changed(self._maj_instant)

        self.ax_xy = self.fig.add_subplot(gs[1, 0])
        self.ax_rad = self.fig.add_subplot(gs[1, 1])
        self.ax_f = self.fig.add_subplot(gs[2, 0])
        self.ax_v = self.fig.add_subplot(gs[2, 1])
        self.canvas = FigureCanvasTkAgg(self.fig, master=cadre)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._preparer_axes()

    def _preparer_axes(self):
        for ax, titre, xl, yl in [
                (self.ax_xy, "Vue de dessus", "x (um)", "y (um)"),
                (self.ax_rad, "Position radiale", "Temps (s)", "r (um)"),
                (self.ax_f, "Profil de force", "r (um)", "Force (pN)"),
                (self.ax_v, "Profil de vitesse", "r (um)", "Vitesse (um/s)")]:
            ax.clear()
            ax.set_title(titre, fontsize=11, fontweight='bold')
            ax.set_xlabel(xl, fontsize=9); ax.set_ylabel(yl, fontsize=9)
            ax.grid(alpha=0.3, ls='--')
        self.ax_xy.set_aspect('equal')
        self.canvas.draw_idle()

    # ------------------------------------------------------------- parametres
    def _lire_parametres(self):
        c = self.champs_saisie
        t = float(c['tension'].get())
        amp = t / 2.0 if self.var_cc.get() == "crete a crete" else t
        dep = [float(x) * 1e-6 for x in c['departs'].get().split(',')]
        cib = [float(x) * 1e-6 for x in c['cibles'].get().split(',')]
        return dict(
            r_part=float(c['r_part'].get()) * 1e-6,
            eps_part=float(c['eps_part'].get()) * EPSILON_0,
            eps_part_rel=float(c['eps_part'].get()),
            sigma_part=float(c['sigma_part'].get()),
            rho_part=float(c['rho_part'].get()),
            eps_milieu=float(c['eps_milieu'].get()) * EPSILON_0,
            eps_milieu_rel=float(c['eps_milieu'].get()),
            sigma_milieu=float(c['sigma_milieu'].get()),
            viscosite=float(c['viscosite'].get()) * 1e-3,
            rho_milieu=float(c['rho_milieu'].get()),
            r_int=float(c['r_int'].get()) * 1e-6,
            r_ext=float(c['r_ext'].get()) * 1e-6,
            gap=float(c['gap'].get()) * 1e-6,
            z=float(c['z'].get()) * 1e-6,
            h_electrode=EPAISSEUR_OR_NM * 1e-9,
            tension_affichee=t, convention=self.var_cc.get(), amplitude=amp,
            frequence=float(c['frequence'].get()) * 1e3,
            forme=self.var_forme.get(),
            champ_amplitude=float(c['champ_amp'].get()),
            champ_longueur=float(c['champ_lon'].get()) * 1e-6,
            duree=float(c['duree'].get()), dt=float(c['dt'].get()) * 1e-3,
            departs=dep, cibles=cib, nr=NR, r_max=R_MAX_UM * 1e-6)

    def _rafraichir_rek(self, *args):
        # Le facteur de Clausius-Mossotti est recalcule a chaque modification,
        # de sorte qu'il ne puisse jamais etre incoherent avec les parametres.
        try:
            p = self._lire_parametres()
            k = facteur_clausius_mossotti(p)
            regime = "POSITIVE, vers l'anneau" if k > 0 else "NEGATIVE, repulsion"
            self.label_rek.config(
                text=f" Re[K] = {k:+.4f}\n dielectrophorese {regime}\n"
                     f" amplitude imposee : {p['amplitude']:.2f} V",
                fg='#B00000' if k > 0 else '#00529B')
        except Exception:
            self.label_rek.config(text=" parametres incomplets", fg='gray')

    def _ecrire_diag(self, texte):
        self.texte_diag.config(state='normal')
        self.texte_diag.delete('1.0', tk.END)
        self.texte_diag.insert('1.0', texte)
        self.texte_diag.config(state='disabled')

    # ----------------------------------------------------------------- calcul
    def calculer(self):
        try:
            p = self._lire_parametres()
        except Exception as e:
            messagebox.showerror("Erreur", f"Valeur invalide :\n\n{e}")
            return
        if not (self.var_ideal.get() or self.var_mes.get()):
            messagebox.showwarning("Champs", "Choisir au moins un champ.")
            return

        self.p = p
        self.re_k = facteur_clausius_mossotti(p)
        self.gamma, fconf, lam = coefficient_trainee(p['r_part'], p['gap'],
                                                     p['viscosite'])
        self.diag = diagnostic_ecoulements(p)

        self._ecrire_diag("Calcul en cours...")
        self.racine.update()

        self.champs = {}
        if self.var_ideal.get():
            self.champs['ideal'] = champ_ideal(p)
        if self.var_mes.get():
            self.champs['mesure'] = champ_mesure(p, self.chemin_champ)

        self.res = simuler(self.champs, p, self.re_k, self.gamma, p['departs'])

        pref = prefacteur_dep(p, self.re_k)
        ch0 = list(self.champs.values())[0]
        v0 = float(vitesse_profil(np.array([p['departs'][0]]), ch0, pref,
                                  self.gamma)[0])
        arr = self.res[list(self.champs)[0]][0]['r'][-1]

        self._ecrire_diag(
            f"Re[K]      : {self.re_k:+.4f}\n"
            f"gamma      : {self.gamma:.3e}\n"
            f"confinement: {fconf:.3f} (lam {lam:.2f})\n"
            f"amplitude  : {p['amplitude']:.2f} V\n"
            f"{'-'*30}\n"
            f"v initiale : {v0*1e6:+.1f} um/s\n"
            f"depart     : {p['departs'][0]*1e6:.0f} um\n"
            f"arrivee    : {arr*1e6:.0f} um\n"
            f"{'-'*30}\n"
            f"echauffement : {self.diag['dT']:.2f} K\n"
            f"u electroth. : {self.diag['u_et']*1e6:.0f} um/s\n"
            f"sedimentation: {self.diag['v_sed']*1e6:.2f} um/s\n"
            f"  traversee  : {self.diag['t_sed']:.0f} s\n"
            f"{'-'*30}\n"
            + ("champ mesure :\n"
               f"  longueur {self.champs['mesure']['longueur']*1e6:.1f} um\n"
               f"  ideale   {p['gap']/math.pi*1e6:.1f} um\n"
               if 'mesure' in self.champs else ""))

        self.slider.valmax = p['duree']
        self.slider.ax.set_xlim(0, p['duree'])
        self._tracer_profils()
        self.slider.set_val(0)

    def _tracer_profils(self):
        p = self.p
        pref = prefacteur_dep(p, self.re_k)
        self._preparer_axes()

        # Vue de dessus.
        #
        # Chaque position de depart occupe un angle propre, et sa trajectoire
        # est representee par un segment radial : un disque plein au depart, un
        # disque pale a l'arrivee, et un pointille de la meme teinte entre les
        # deux. La position initiale et la position finale sont donc lisibles
        # d'un seul coup d'oeil, sans avoir a manipuler le curseur.
        #
        # Le curseur ne sert plus qu'a promener un marqueur rouge le long de ce
        # segment, ce qui montre OU en est la particule a l'instant choisi sans
        # rien masquer du parcours complet.
        th = np.linspace(0, 2 * np.pi, 400)
        self.ax_xy.fill(p['r_ext'] * 1e6 * np.cos(th),
                        p['r_ext'] * 1e6 * np.sin(th), color=COUL_OR,
                        alpha=0.55, zorder=1)
        self.ax_xy.fill(p['r_int'] * 1e6 * np.cos(th),
                        p['r_int'] * 1e6 * np.sin(th), color='white', zorder=2)
        lim = R_AFFICHAGE_UM
        self.ax_xy.set_xlim(-lim, lim); self.ax_xy.set_ylim(-lim, lim)

        nom0 = list(self.champs)[0]
        traj0 = self.res[nom0]
        n = len(traj0)
        self.angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        couleurs = plt.cm.viridis(np.linspace(0, 0.92, n))

        for k, (s_, c) in enumerate(zip(traj0, couleurs)):
            a = self.angles[k]
            r0 = s_['r0'] * 1e6
            rf = s_['r'][-1] * 1e6
            # Segment radial entre depart et arrivee
            self.ax_xy.plot([r0 * np.cos(a), rf * np.cos(a)],
                            [r0 * np.sin(a), rf * np.sin(a)], ':', color=c,
                            lw=1.6, alpha=0.85, zorder=3)
            # Depart : disque plein
            self.ax_xy.plot([r0 * np.cos(a)], [r0 * np.sin(a)], 'o', color=c,
                            ms=8, markeredgecolor='black', markeredgewidth=1.0,
                            zorder=4)
            # Arrivee : meme teinte, pale et cerclee
            self.ax_xy.plot([rf * np.cos(a)], [rf * np.sin(a)], 'o', color=c,
                            ms=8, alpha=0.40, markeredgecolor='black',
                            markeredgewidth=1.0, zorder=4)

        # Marqueur mobile, commun a toutes les particules
        self.curseur, = self.ax_xy.plot(
            [s_['r0'] * 1e6 * np.cos(a) for s_, a in zip(traj0, self.angles)],
            [s_['r0'] * 1e6 * np.sin(a) for s_, a in zip(traj0, self.angles)],
            'o', color='red', ms=7, markeredgecolor='darkred',
            markeredgewidth=1.0, zorder=6)

        self.ax_xy.plot([], [], 'o', color='gray', ms=8,
                        markeredgecolor='black', label="depart")
        self.ax_xy.plot([], [], 'o', color='gray', ms=8, alpha=0.40,
                        markeredgecolor='black', label="arrivee")
        self.ax_xy.plot([], [], 'o', color='red', ms=7, label="instant choisi")
        self.ax_xy.legend(fontsize=7, loc='upper right')
        self.ax_xy.set_title(f"Vue de dessus, {n} positions de depart",
                             fontsize=11, fontweight='bold')

        # Position radiale au cours du temps
        self.lignes_radiales = []
        for nom, traj in self.res.items():
            for k, s in enumerate(traj):
                ligne, = self.ax_rad.plot(s['t'], s['r'] * 1e6, _style(nom),
                                          color=_coul(nom), lw=1.6, alpha=0.85,
                                          label=f"champ {nom}" if k == 0 else None)
                self.lignes_radiales.append(ligne)
        for s in self.suivi:
            self.ax_rad.plot(s['t'], s['r'] * 1e6, '-', color=COUL_REEL,
                             lw=1.2, alpha=0.5)
        if self.suivi:
            self.ax_rad.plot([], [], '-', color=COUL_REEL, lw=1.2,
                             label="mesure")
        _anneau_h(self.ax_rad, p)
        self.ax_rad.set_xlim(0, p['duree'])
        self.ax_rad.legend(fontsize=7)
        self.ligne_t = self.ax_rad.axvline(0, color='red', ls='--', lw=2,
                                           alpha=0.7)

        # Profils de force et de vitesse
        rr = np.linspace(p['r_ext'], R_AFFICHAGE_UM * 1e-6, 900)
        self.marqueurs = {}
        for ax, conv, lab in [(self.ax_f, 1e12, "Force (pN)"),
                              (self.ax_v, 1e6, "Vitesse (um/s)")]:
            vals = []
            for nom, ch in self.champs.items():
                g = np.interp(rr, ch['r'], ch['gradient'], left=0, right=0)
                y = (pref * g) * conv if conv == 1e12 else \
                    (pref * g / self.gamma) * conv
                vals.append(y)
                ax.plot(rr * 1e6, y, _style(nom), color=_coul(nom), lw=2,
                        label=f"champ {nom}")
            ax.axhline(0, color='k', lw=0.8)
            _anneau_v(ax, p)
            ax.set_xlim(p['r_ext'] * 1e6 * 0.9, R_AFFICHAGE_UM)
            m_utile = rr > p['r_ext'] * 1.12
            if not np.any(m_utile):
                m_utile = np.ones_like(rr, dtype=bool)
            h = max(float(np.percentile(np.abs(v[m_utile]), 99))
                    for v in vals)
            h = h if h > 0 else 1.0
            ax.set_ylim(-1.35 * h if self.re_k > 0 else -0.15 * h,
                        0.15 * h if self.re_k > 0 else 1.35 * h)
            ax.set_ylabel(lab, fontsize=9)
            ax.legend(fontsize=7)
            m, = ax.plot([], [], 'o', color='red', ms=5, alpha=0.85, zorder=6)
            self.marqueurs[ax] = m
        self.canvas.draw_idle()

    def _maj_instant(self, val):
        # Deplace le marqueur rouge le long de chaque segment radial, sans rien
        # effacer du parcours deja trace.
        if not self.res:
            return
        p = self.p
        nom0 = list(self.champs)[0]
        traj = self.res[nom0]
        i = min(int(val / p['dt']), len(traj[0]['t']) - 1)
        rr = np.array([s['r'][i] for s in traj])
        self.curseur.set_data(rr * np.cos(self.angles) * 1e6,
                              rr * np.sin(self.angles) * 1e6)
        self.ligne_t.set_xdata([val, val])

        # Les marqueurs des profils suivent TOUTES les particules, et non plus
        # leur seule moyenne : avec une vingtaine de positions de depart, la
        # moyenne ne representerait aucune particule reelle.
        pref = prefacteur_dep(p, self.re_k)
        ch = self.champs[nom0]
        g = np.interp(rr, ch['r'], ch['gradient'], left=0, right=0)
        self.marqueurs[self.ax_f].set_data(rr * 1e6, pref * g * 1e12)
        self.marqueurs[self.ax_v].set_data(rr * 1e6,
                                           pref * g / self.gamma * 1e6)
        self.canvas.draw_idle()

    # --------------------------------------------------------------- fenetres
    def ouvrir_abaque(self):
        if not self.res:
            messagebox.showinfo("Abaque", "Lancer d'abord le calcul.")
            return
        ab = abaque(self.champs, self.p, self.re_k, self.gamma,
                    self.p['departs'], self.p['cibles'])
        self.ab = ab
        _afficher(figure_abaque(self.res, ab, self.champs, self.p))
        _afficher(figure_temps_atteinte(ab, self.champs, self.p))

    def ouvrir_champs(self):
        if not self.res:
            messagebox.showinfo("Champs", "Lancer d'abord le calcul.")
            return
        _afficher(figure_champs(self.champs, self.p, self.re_k,
                                    self.gamma, self.diag))

    def charger_suivi(self):
        chemin = filedialog.askopenfilename(
            title="Trajectoires reelles",
            filetypes=[("Classeurs Excel", "*.xlsx *.xls")])
        if not chemin:
            return
        self.suivi = lire_suivi(chemin)
        messagebox.showinfo("Suivi", f"{len(self.suivi)} trajectoires lues.")
        if self.res:
            self._tracer_profils()
            self.slider.set_val(0)
            _afficher(figure_comparaison(self.champs, self.suivi, self.p,
                                         self.re_k, self.gamma, self.res))

    def charger_champ(self):
        chemin = filedialog.askopenfilename(
            title="Champ mesure (champ_reconstruit.csv)",
            filetypes=[("Fichiers CSV", "*.csv")])
        if chemin:
            self.chemin_champ = chemin
            messagebox.showinfo("Champ mesure",
                                f"{os.path.basename(chemin)} sera employe au "
                                f"prochain calcul.")

    def exporter(self):
        if not self.res:
            messagebox.showinfo("Export", "Lancer d'abord le calcul.")
            return
        dossier = filedialog.askdirectory(title="Dossier de sortie")
        if not dossier:
            return
        self.dossier = dossier
        p = self.p

        self.fig.savefig(os.path.join(dossier, "01-interface.svg"),
                         format='svg', bbox_inches='tight', facecolor='white')
        ab = abaque(self.champs, p, self.re_k, self.gamma, p['departs'],
                    p['cibles'], bavard=False)
        figure_abaque(self.res, ab, self.champs, p).savefig(
            os.path.join(dossier, "02-abaque-deplacement.svg"), format='svg',
            bbox_inches='tight', facecolor='white')
        figure_temps_atteinte(ab, self.champs, p).savefig(
            os.path.join(dossier, "03-temps-atteinte.svg"), format='svg',
            bbox_inches='tight', facecolor='white')
        figure_champs(self.champs, p, self.re_k, self.gamma, self.diag).savefig(
            os.path.join(dossier, "04-champs-et-vitesses.svg"), format='svg',
            bbox_inches='tight', facecolor='white')
        if self.suivi:
            figure_comparaison(self.champs, self.suivi, p, self.re_k,
                               self.gamma, self.res).savefig(
                os.path.join(dossier, "05-comparaison.svg"), format='svg',
                bbox_inches='tight', facecolor='white')

        lignes = []
        for nom, traj in self.res.items():
            for s in traj:
                for t, r in zip(s['t'], s['r']):
                    lignes.append(dict(champ=nom, depart_um=s['r0'] * 1e6,
                                       temps_s=t, r_um=r * 1e6))
        pd.DataFrame(lignes).to_csv(
            os.path.join(dossier, "trajectoires.csv"), index=False)

        meta = dict(particule=dict(rayon_um=p['r_part'] * 1e6, re_k=self.re_k,
                                   gamma=self.gamma),
                    signal=dict(tension_affichee=p['tension_affichee'],
                                convention=p['convention'],
                                amplitude_V=p['amplitude'],
                                frequence_Hz=p['frequence'], forme=p['forme']),
                    geometrie=dict(r_interieur_um=p['r_int'] * 1e6,
                                   r_exterieur_um=p['r_ext'] * 1e6,
                                   hauteur_cavite_um=p['gap'] * 1e6,
                                   z_um=p['z'] * 1e6),
                    mecanismes=dict(
                        echauffement_K=self.diag['dT'],
                        vitesse_electrothermique_um_s=self.diag['u_et'] * 1e6,
                        amortissement_aceo=self.diag['amortissement'],
                        sedimentation_um_s=self.diag['v_sed'] * 1e6),
                    champs=list(self.champs))
        if 'mesure' in self.champs:
            meta['champ_mesure'] = dict(
                longueur_um=self.champs['mesure']['longueur'] * 1e6,
                amplitude=self.champs['mesure']['amplitude'],
                potentiel_implique_V=self.champs['mesure']['potentiel'],
                longueur_ideale_um=p['gap'] / math.pi * 1e6,
                source=self.champs['mesure']['source'])
        with open(os.path.join(dossier, "simulation.json"), 'w',
                  encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        messagebox.showinfo("Export", f"Fichiers enregistres dans\n{dossier}")


def principal():
    if not TK_DISPO:
        print("[INFO] tkinter indisponible.")
        return
    racine = tk.Tk()
    app = Application(racine)

    def fermer():
        racine.quit(); racine.destroy(); plt.close('all')

    racine.protocol("WM_DELETE_WINDOW", fermer)
    print("Interface prete. Ajuster les parametres puis cliquer sur CALCULER.")
    racine.mainloop()


if __name__ == "__main__":
    principal()