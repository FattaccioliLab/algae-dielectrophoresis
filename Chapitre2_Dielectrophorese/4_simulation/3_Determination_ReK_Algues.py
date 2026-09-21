# -*- coding: utf-8 -*-
# =============================================================================
# CARACTERISATION DIELECTRIQUE DES ALGUES A LA CELLULE UNIQUE
# Ajustement de Re[K] sur le deplacement mesure, dans le champ calibre
# =============================================================================
#
# CE QUE FAIT LE PROGRAMME
#
#   Le champ ayant ete mesure une fois pour toutes sur des billes de
#   proprietes connues, il ne reste plus qu'une seule inconnue dans le bilan
#   des forces : la partie reelle du facteur de Clausius-Mossotti de la
#   cellule. Pour chaque algue suivie, cette valeur est ajustee de maniere a
#   ce que le DEPLACEMENT simule reproduise au mieux le deplacement mesure.
#
#     position mesuree r(t)
#             |
#             |  champ calibre, connu
#             |  rayon de la cellule, mesure
#             |  trainee corrigee du confinement
#             v
#     une seule inconnue : Re[K]
#             |
#             |  minimisation de l'ecart entre trajectoire simulee et mesuree
#             v
#     Re[K] de cette cellule
#
#   L'ajustement porte sur un unique parametre scalaire, borne entre -0,5 et 1
#   par les limites physiques du facteur pour une sphere. Ce caractere
#   unidimensionnel garantit l'unicite de la solution et permet de tracer la
#   courbe de l'ecart en fonction de la valeur d'essai, ce qui donne une
#   lecture directe de la qualite de l'ajustement.
#
#   L'ajustement se fait sur la POSITION et non sur la vitesse. Deriver une
#   trajectoire amplifie le bruit de pointage, alors que la position est la
#   grandeur effectivement mesuree ; integrer le modele pour le comparer aux
#   positions est donc plus robuste que deriver la mesure pour la comparer au
#   modele.
#
# -----------------------------------------------------------------------------
# CE QUE CE PROGRAMME APPORTE PAR RAPPORT A UN INDICE DE POPULATION
# -----------------------------------------------------------------------------
#   Il ne produit pas une valeur moyenne mais un ensemble de valeurs
#   individuelles, dont la dispersion est elle-meme une information. Une
#   distribution etroite signale une population homogene ; une distribution
#   large ou multimodale signale la coexistence de sous-populations aux
#   proprietes dielectriques distinctes, ce qu'une mesure de population ne
#   peut pas voir.
#
# -----------------------------------------------------------------------------
# ORGANISATION DE L'INTERFACE
# -----------------------------------------------------------------------------
#   A gauche  : parametres, chargement du champ et du suivi, liste des algues
#               a cocher, diagnostics, boutons.
#   A droite  : quatre vues.
#                 trajectoire mesuree et ajustee, pour les algues cochees
#                 profil du cout en fonction de la valeur d'essai de Re[K]
#                 distribution des valeurs obtenues sur la population
#                 recapitulatif chiffre
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

from scipy.optimize import minimize_scalar
from scipy.signal import savgol_filter
from scipy.stats import norm

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
R_INT_UM       = 50.0
R_EXT_UM       = 140.0
HAUTEUR_CAV_UM = 20.0

# --- Cellules ----------------------------------------------------------------
# Le rayon intervient au cube dans la force et lineairement dans la trainee :
# la vitesse varie donc comme son carre. Une erreur de dix pour cent sur le
# rayon se traduit par vingt pour cent sur Re[K]. Il vaut la peine de le
# mesurer sur la distribution des tailles detectees plutot que de le supposer.
RAYON_ALGUE_UM = 5.0

# Source du rayon employe dans le calcul.
#
#   "fichier"  le rayon propre de chaque cellule, mediane des valeurs
#              detectees le long de sa trajectoire. C'est le choix a retenir
#              quand la population est reellement heterogene en taille, ce que
#              le programme verifie et annonce.
#   "impose"   la valeur saisie, la meme pour toutes. A retenir quand la
#              detection de taille est trop bruitee pour etre fiable, ou pour
#              comparer un jeu a un autre a taille fixee.
#
# Le choix n'est pas anodin : la force varie comme le cube du rayon et la
# trainee comme sa premiere puissance, donc la vitesse comme son CARRE. A
# vitesse mesuree egale, Re[K] varie donc comme l'inverse du carre du rayon :
# passer de 5 a 7 um divise la valeur deduite par deux.
SOURCE_RAYON = "fichier"

# --- Milieu ------------------------------------------------------------------
EPS_MILIEU_REL  = 78.0
SIGMA_MILIEU    = 0.055
VISCOSITE_MPA_S = 1.0
RHO_MILIEU      = 1000.0

# --- Champ calibre -----------------------------------------------------------
# Decroissance exponentielle depuis le bord de l'electrode, telle que la
# reconstruction sur billes l'a etablie. Un fichier CSV peut aussi etre charge.
CHAMP_AMPLITUDE = 1.381e10      # carre du champ au bord, en V2/m2
CHAMP_LONGUEUR_UM = 19.4        # longueur de decroissance
CHAMP_VALIDITE_MIN_UM = 160.0   # plage sur laquelle la reconstruction repose
CHAMP_VALIDITE_MAX_UM = 220.0

# --- Signal ------------------------------------------------------------------
FREQUENCE_KHZ = 100.0

# --- Ajustement --------------------------------------------------------------
# Bornes explorees par l'ajustement.
#
# Les bornes PHYSIQUES d'une sphere homogene sont -0,5 et 1. La borne basse
# est cependant volontairement elargie a -1 : si l'optimum se trouve en deca de
# -0,5, cela signale que le modele ne rend pas compte du mouvement observe, et
# il vaut mieux le voir que de le masquer en butant contre la borne. Une valeur
# ainsi obtenue n'est evidemment pas un facteur de Clausius-Mossotti : c'est un
# diagnostic.
RE_K_MIN = -10.0
RE_K_MAX = 10.0
RE_K_PHYSIQUE_MIN = -0.5
RE_K_PHYSIQUE_MAX = 1.0

# Nombre de points du balayage initial. Il doit suivre l'etendue des bornes :
# le pas de grille vaut leur difference divisee par ce nombre, et un pas trop
# grossier ferait manquer le minimum au balayage, que le raffinement local ne
# rattraperait pas. Le calcul etant analytique, augmenter ce nombre ne coute
# presque rien.
RE_K_POINTS = 2001

# Pas du curseur de valeur imposee. Sur une plage de vingt unites, un pas trop
# fin rendrait le curseur inutilisable.
RE_K_PAS_CURSEUR = 0.02
PAS_TEMPS_MS = 5.0

# --- Fenetre temporelle retenue pour l'ajustement ----------------------------
# Bornes, en secondes depuis l'activation, entre lesquelles l'ecart au modele
# est evalue. Les points situes hors de cette fenetre restent traces et
# simules, mais n'entrent pas dans le calcul du cout.
#
# Cette exclusion n'est pas une commodite. En fin de trajectoire, la cellule
# s'est eloignee au point que le gradient de champ y est negligeable : son
# mouvement residuel n'est alors plus dielectrophoretique, et l'inclure
# reviendrait a demander au modele d'expliquer ce qu'il ne decrit pas. Le
# debut peut de meme etre entache du transitoire d'etablissement du champ.
#
# Laisser une borne a None revient a ne pas la contraindre.
T_FIT_MIN_S = 0.0
T_FIT_MAX_S = None

# --- Recalage temporel -------------------------------------------------------
SEUIL_ACTIVATION_UM_S = 5.0
POINTS_CONSECUTIFS = 3
ECART_MAX_ACQUISITION = 1.5
# Nombre minimal de cellules pour qu'un groupe soit tenu pour une acquisition
# a part entiere plutot que pour une detection tardive isolee.
EFFECTIF_MIN_ACQUISITION = 3
MODE_RECALAGE = "individuel"
LISSAGE_DETECTION_S = 0.30

# --- Criteres de rejet -------------------------------------------------------
POINTS_MIN = 15
DEPLACEMENT_MIN_UM = 2.0

# --- Affichage ---------------------------------------------------------------
R_AFFICHAGE_UM = 300.0

EPSILON_0 = 8.854187817e-12
GRAVITE   = 9.81

matplotlib.rcParams['figure.facecolor']  = 'white'
matplotlib.rcParams['savefig.facecolor'] = 'white'
matplotlib.rcParams['axes.facecolor']    = 'white'

COUL_MES  = 'magenta'
COUL_FIT  = '#1f77b4'
COUL_DIST = '#2ca02c'
COUL_OR   = 'gold'
COUL_GRIS = '#7f7f7f'
COUL_ALER = '#d62728'

COLONNES_TEMPS = ['temps_s', 'time_s', 'temps', 'time', 't_s', 't']
COLONNES_RAYON = ['r_um', 'rayon_um', 'radius_um', 'r', 'rayon', 'distance_um']
COLONNES_TAILLE = ['r_part_um', 'rayon_particule_um', 'radius_particle_um',
                   'a_um', 'taille_um', 'size_um', 'r_particule_um']
COLONNES_IDENT = ['particle_id', 'cellule', 'particule', 'particle', 'id',
                  'track_id', 'track', 'nom', 'name', 'label', 'objet']

# Colonne indiquant si la particule a effectivement ete suivie sur l'image.
# Les points ou le suivi a echoue portent souvent la derniere position connue,
# qui n'est plus une mesure : ils sont donc ecartes.
COLONNES_SUIVI = ['tracked', 'suivi', 'valide', 'valid', 'ok']


# =============================================================================
# PHYSIQUE
# =============================================================================

def coefficient_trainee(rayon, hauteur, viscosite):
    # Trainee corrigee du confinement entre deux parois planes, modele de
    # Happel et Brenner. Pour une cellule de 5 um de rayon dans une cavite de
    # 20 um, le rapport de confinement vaut 0,5 et le facteur environ 1,8 : a
    # force egale, la vitesse est reduite de pres de moitie.
    g0 = 6.0 * math.pi * viscosite * rayon
    lam = rayon / (hauteur / 2.0)
    if lam < 0.95:
        den = (1.0 - 1.004 * lam + 0.418 * lam ** 3
               + 0.21 * lam ** 4 - 0.169 * lam ** 5)
        f = 1.0 / den
    else:
        f = 1.0 + (9.0 / 8.0) * lam + lam ** 3
    return g0 * f, f, lam


def prefacteur_unitaire(p, rayon):
    # 2 pi a^3 eps_m, c'est-a-dire le prefacteur de la force SANS Re[K], qui
    # est justement l'inconnue a ajuster. Le rayon eleve au cube est celui de
    # la CELLULE, non sa position radiale.
    return 2.0 * math.pi * rayon ** 3 * p['eps_milieu']


def gradient_champ(r, champ):
    # Gradient du carre du champ, evalue analytiquement.
    #
    # Le champ calibre etant decrit par une exponentielle, sa derivee s'obtient
    # exactement : cela evite d'introduire du bruit de derivation numerique
    # dans l'integration des trajectoires.
    a, L, rb = champ['amplitude'], champ['longueur'], champ['r_bord']
    r = np.asarray(r, dtype=float)
    return np.where(r > rb, -a / L * np.exp(-(r - rb) / L), 0.0)


def carre_champ(r, champ):
    a, L, rb = champ['amplitude'], champ['longueur'], champ['r_bord']
    r = np.asarray(r, dtype=float)
    return a * np.exp(-np.maximum(r - rb, 0.0) / L)


def charger_champ(chemin=None, p=None):
    # Champ calibre, lu dans un fichier ou decrit par ses deux parametres.
    if chemin and os.path.exists(chemin):
        d = np.genfromtxt(chemin, delimiter=',', names=True)
        r, e2 = np.asarray(d['r_m']), np.asarray(d['E2_V2_par_m2'])
        ok = np.isfinite(e2) & (e2 > 0)
        rb = p['r_ext']
        pp = np.polyfit(r[ok] - rb, np.log(e2[ok]), 1)
        champ = dict(amplitude=float(math.exp(pp[1])),
                     longueur=float(-1.0 / pp[0]), r_bord=rb,
                     source=os.path.basename(chemin),
                     r_min=float(r[ok].min()), r_max=float(r[ok].max()))
    else:
        champ = dict(amplitude=p['champ_amplitude'],
                     longueur=p['champ_longueur'], r_bord=p['r_ext'],
                     source="parametres saisis",
                     r_min=p['validite_min'], r_max=p['validite_max'])
    champ['potentiel'] = math.sqrt(champ['amplitude']) * p['gap']
    return champ


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


def regrouper_par_acquisition(detections, ecart_max=ECART_MAX_ACQUISITION,
                              effectif_min=EFFECTIF_MIN_ACQUISITION):
    # Regroupe les cellules par acquisition a partir de leurs seuls instants de
    # mise en mouvement, sans recourir a la numerotation des onglets.
    #
    # Un garde-fou est necessaire : une cellule qui se met en mouvement bien
    # apres les autres, parce qu'elle est lente ou que sa detection a echoue,
    # formerait a elle seule un groupe. On la prendrait alors pour une
    # acquisition distincte, et elle serait recalee sur son propre retard.
    #
    # Les groupes trop peu nombreux sont donc rattaches au groupe le plus
    # proche en temps, ce qui est signale.
    if not detections:
        return {}, {}
    cles = sorted(detections, key=lambda k: detections[k])
    groupes = {cles[0]: 0}
    num = 0
    for prec, cour in zip(cles[:-1], cles[1:]):
        if detections[cour] - detections[prec] > ecart_max:
            num += 1
        groupes[cour] = num

    def mediane(g):
        return float(np.median([detections[k] for k in groupes
                                if groupes[k] == g]))

    # Rattachement des groupes minoritaires
    seuil = max(effectif_min, int(0.05 * len(detections)))
    while True:
        effectifs = {}
        for g in groupes.values():
            effectifs[g] = effectifs.get(g, 0) + 1
        if len(effectifs) < 2:
            break
        petits = [g for g, n in effectifs.items() if n < seuil]
        if not petits:
            break
        g_petit = min(petits, key=lambda g: effectifs[g])
        autres = [g for g in effectifs if g != g_petit]
        t_petit = mediane(g_petit)
        g_proche = min(autres, key=lambda g: abs(mediane(g) - t_petit))
        membres = [k for k in groupes if groupes[k] == g_petit]
        print(f"  [INFO] groupe de {len(membres)} cellule(s) detecte a "
              f"{t_petit:.2f} s : trop peu nombreux pour etre une")
        print(f"         acquisition distincte, rattache au groupe de "
              f"{mediane(g_proche):.2f} s")
        for k in membres:
            groupes[k] = g_proche

    # Renumerotation contigue
    ordre = sorted(set(groupes.values()), key=mediane)
    corresp = {g: i for i, g in enumerate(ordre)}
    groupes = {k: corresp[g] for k, g in groupes.items()}
    instants = {i: float(np.median([detections[k] for k in groupes
                                    if groupes[k] == i]))
                for i in set(groupes.values())}
    return groupes, instants


def _tableaux_par_cellule(chemin, cadence=20.0):
    # Renvoie une liste de couples (nom, tableau), un par cellule, quel que soit
    # le format du fichier.
    #
    # Trois dispositions sont reconnues, sans que l'utilisateur ait a le
    # preciser.
    #
    #   Classeur Excel a plusieurs onglets : un onglet par cellule.
    #
    #   Tableau au format LONG : une colonne identifie la cellule, les autres
    #   portent le temps et le rayon. Chaque cellule occupe donc plusieurs
    #   lignes. C'est la disposition la plus courante en sortie de suivi.
    #
    #   Tableau au format LARGE : une colonne de temps, puis une colonne de
    #   rayon par cellule. Le nom de chaque colonne devient celui de la cellule.
    if chemin.lower().endswith(('.xlsx', '.xls')):
        cl = pd.ExcelFile(chemin)
        if len(cl.sheet_names) > 1:
            return [(o, cl.parse(o)) for o in cl.sheet_names], "onglets"
        df = cl.parse(cl.sheet_names[0])
    else:
        # Le separateur et le decimal sont devines, les exports francais
        # employant souvent le point-virgule et la virgule decimale.
        try:
            df = pd.read_csv(chemin, sep=None, engine='python')
        except Exception:
            df = pd.read_csv(chemin, sep=';', decimal=',')
        if df.shape[1] == 1:
            df = pd.read_csv(chemin, sep=';', decimal=',')

    ci = _colonne(df.columns, COLONNES_IDENT)
    if ci is not None and df[ci].nunique() > 1:
        return [(str(v), df[df[ci] == v].copy())
                for v in df[ci].dropna().unique()], "long"

    # Format large : une colonne de temps, plusieurs colonnes de rayon
    ct = _colonne(df.columns, COLONNES_TEMPS)
    colonnes_r = [c for c in df.columns
                  if c != ct and pd.api.types.is_numeric_dtype(df[c])
                  and (c != ci)]
    if ct is not None and len(colonnes_r) > 1:
        sorties = []
        for c in colonnes_r:
            sous = pd.DataFrame({'temps_s': df[ct], 'r_um': df[c]})
            sorties.append((str(c), sous))
        return sorties, "large"

    # Une seule cellule
    return [(os.path.splitext(os.path.basename(chemin))[0], df)], "unique"


def lire_algues(chemin, p):
    # Lit un fichier de suivi et recale chaque trajectoire sur l'instant ou le
    # champ lui a ete applique. Le format est reconnu automatiquement.
    tableaux, disposition = _tableaux_par_cellule(chemin)
    print(f"  format reconnu : {disposition}, {len(tableaux)} cellules")
    brutes, det = [], {}

    for k, (nom, df) in enumerate(tableaux):
        if df.empty or len(df) < POINTS_MIN:
            continue
        cr = _colonne(df.columns, COLONNES_RAYON)
        if cr is None:
            # En format large la colonne a ete renommee, sinon on prend la
            # premiere colonne numerique qui n'est ni le temps ni l'identifiant.
            cr = 'r_um' if 'r_um' in df.columns else None
        if cr is None:
            continue
        ct = _colonne(df.columns, COLONNES_TEMPS)
        ca = _colonne(df.columns, COLONNES_TAILLE)
        cs = _colonne(df.columns, COLONNES_SUIVI)
        r = pd.to_numeric(df[cr], errors='coerce').values * 1e-6
        t = (pd.to_numeric(df[ct], errors='coerce').values if ct is not None
             else np.arange(len(df)) / 20.0)
        ok = np.isfinite(r) & np.isfinite(t)
        if cs is not None:
            # La colonne peut porter des booleens, des entiers ou du texte
            suivi = df[cs].astype(str).str.strip().str.lower()
            ok &= suivi.isin(['true', '1', 'vrai', 'oui', 'yes',
                              'ok']).values
        r, t = r[ok], t[ok]
        if len(r) < POINTS_MIN:
            continue
        ordre = np.argsort(t)
        r, t = r[ordre], t[ordre]
        # Dispersion temporelle du rayon detecte, qui mesure la fiabilite de
        # la detection de taille : un rayon qui varie de moitie au fil des
        # images ne peut pas etre pris pour une mesure.
        rayon_dispersion = None

        # Rayon propre a la cellule s'il figure dans le fichier.
        #
        # C'est un point important : la vitesse varie comme le CARRE du rayon,
        # le cube intervenant dans la force et la puissance un dans la trainee.
        # Employer un rayon commun alors que le fichier donne le rayon de
        # chaque cellule fausserait donc Re[K] d'autant.
        rayon = None
        if ca is not None:
            vals = pd.to_numeric(df[ca], errors='coerce').values[ok]
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if len(vals):
                rayon = float(np.median(vals)) * 1e-6
                if len(vals) > 3 and np.median(vals) > 0:
                    rayon_dispersion = float(np.std(vals) / np.median(vals))

        v = _derivee_lissee(t, r)
        seuil = p['seuil']
        indice = None
        for i in range(len(v) - POINTS_CONSECUTIFS + 1):
            if np.all(np.abs(v[i:i + POINTS_CONSECUTIFS]) > seuil):
                indice = i
                break
        brutes.append(dict(nom=nom, t=t, r=r, cle=k, rayon=rayon,
                           dispersion=rayon_dispersion))
        if indice is not None:
            det[k] = float(t[indice])

    if not brutes:
        return []

    # Si presque aucune cellule ne franchit le seuil, c'est le seuil qui est
    # inadapte, pas les donnees. Le programme le dit et propose une valeur,
    # plutot que de renvoyer une liste vide sans explication.
    if len(det) < max(2, 0.2 * len(brutes)):
        vitesses = []
        for b in brutes:
            v = _derivee_lissee(b['t'], b['r'])
            vitesses.append(float(np.max(np.abs(v))))
        vitesses = np.array(vitesses)
        suggere = float(np.percentile(vitesses, 25)) * 0.6
        print(f"\n  [ALERTE] seulement {len(det)} cellule(s) sur "
              f"{len(brutes)} franchissent le seuil de")
        print(f"           {p['seuil']*1e6:.1f} um/s. les vitesses maximales "
              f"observees vont de")
        print(f"           {vitesses.min()*1e6:.2f} a "
              f"{vitesses.max()*1e6:.2f} um/s.")
        print(f"\n           ces cellules se deplacent bien plus lentement que")
        print(f"           celles pour lesquelles le seuil a ete regle. abaisser")
        print(f"           le seuil a environ {suggere*1e6:.1f} um/s le rendrait")
        print(f"           adapte a ce jeu.")
        print(f"\n           a defaut, l'origine des temps est prise au premier")
        print(f"           point de chaque trajectoire, ce qui suppose que le")
        print(f"           champ etait deja applique au debut de la video.")
        # Repli : origine au premier point, plutot que de tout perdre
        for b in brutes:
            det.setdefault(b['cle'], float(b['t'][0]))

    groupes, instants = regrouper_par_acquisition(det)
    print(f"\n  {len(brutes)} cellules lues, {len(instants)} acquisition(s)")
    for g in sorted(instants):
        n = sum(1 for kk in groupes if groupes[kk] == g)
        print(f"    acquisition {g + 1} : activation a {instants[g]:.3f} s, "
              f"{n} cellules")

    mode = p.get('mode_recalage', MODE_RECALAGE)
    sorties = []
    for b in brutes:
        k = b['cle']
        if k in det:
            t_ref = det[k] if mode == "individuel" else instants[groupes[k]]
            acq = groupes[k]
        elif instants:
            acq = min(instants,
                      key=lambda g: abs(instants[g] - float(np.median(b['t']))))
            t_ref = instants[acq]
        else:
            continue
        m = b['t'] >= t_ref
        if int(np.sum(m)) < POINTS_MIN:
            continue
        t_r, r_r = b['t'][m] - t_ref, b['r'][m]
        deplacement = abs(r_r[-1] - r_r[0])
        sorties.append(dict(
            nom=b['nom'], t=t_r, r=r_r, r0=float(r_r[0]),
            rayon=b['rayon'], rayon_dispersion=b.get('dispersion'),
            acquisition=acq, activation=t_ref,
            deplacement=deplacement,
            retenue=deplacement >= p['deplacement_min']))

    if not sorties:
        print(f"\n  [ERREUR] aucune cellule exploitable.")
        print(f"           verifier le seuil d'activation et le deplacement")
        print(f"           minimal, qui sont les deux criteres de rejet.")
        return []

    n_rejet = sum(1 for s in sorties if not s['retenue'])
    if n_rejet:
        print(f"  {n_rejet} cellules a deplacement insuffisant, signalees mais "
              f"conservees")

    # Sens du mouvement, qui annonce le signe attendu de Re[K]
    vers_exterieur = sum(1 for s in sorties
                         if s['r'][-1] - s['r'][0] > 1e-6)
    vers_interieur = sum(1 for s in sorties
                         if s['r'][-1] - s['r'][0] < -1e-6)
    print(f"\n  SENS DU MOUVEMENT")
    print(f"    {vers_exterieur} cellules s'eloignent de l'anneau, "
          f"{vers_interieur} s'en rapprochent")
    if vers_interieur > vers_exterieur:
        print(f"    la majorite se rapproche : dielectrophorese POSITIVE,")
        print(f"    Re[K] attendu positif.")
    elif vers_exterieur > vers_interieur:
        print(f"    la majorite s'eloigne : dielectrophorese NEGATIVE,")
        print(f"    Re[K] attendu negatif.")
    else:
        print(f"    les deux sens coexistent : population heterogene, ou")
        print(f"    mouvement domine par autre chose que la dielectrophorese.")

    # Bilan sur les rayons, qui conditionne le choix de leur source
    rayons = [s['rayon'] for s in sorties if s['rayon']]
    if rayons:
        rayons = np.array(rayons) * 1e6
        disp = [s['rayon_dispersion'] for s in sorties
                if s.get('rayon_dispersion')]
        print(f"\n  RAYONS LUS DANS LE FICHIER")
        print(f"    de {rayons.min():.2f} a {rayons.max():.2f} um, "
              f"mediane {np.median(rayons):.2f} um")
        print(f"    dispersion ENTRE cellules : "
              f"{100*rayons.std()/rayons.mean():.1f} %")
        if disp:
            d_med = float(np.median(disp)) * 100
            print(f"    dispersion TEMPORELLE mediane : {d_med:.1f} %")
            if d_med > 25:
                print(f"    [ALERTE] la taille detectee varie fortement au fil")
                print(f"             des images : elle est probablement peu")
                print(f"             fiable. envisager un rayon impose.")
            elif rayons.std() / rayons.mean() > 2 * float(np.median(disp)):
                print(f"    la dispersion entre cellules depasse nettement la")
                print(f"    dispersion temporelle : les tailles different")
                print(f"    reellement, le rayon du fichier est preferable.")
        print(f"    source retenue : {p.get('source_rayon', SOURCE_RAYON)}")
    return sorties


# =============================================================================
# AJUSTEMENT DE Re[K]
# =============================================================================

def simuler_trajectoire(r0, re_k, champ, pref_unit, gamma, t, dt=None,
                        r_min=None, r_max=480e-6, deplacement_max=0.5e-6):
    # Trajectoire radiale, obtenue par SOLUTION ANALYTIQUE.
    #
    # Le champ calibre etant une exponentielle, l'equation du mouvement
    #
    #     dr/dt = -C exp( -(r - rb) / L )     avec C = pref Re[K] A / (L gamma)
    #
    # est a variables separables et s'integre exactement :
    #
    #     exp( (r - rb) / L ) dr = -C dt
    #     r(t) = rb + L ln( exp( (r0 - rb) / L ) - C t / L )
    #
    # Cette forme close remplace l'integration pas a pas, qu'elle reproduit au
    # nanometre pres tout en etant environ quatre cents fois plus rapide. Le
    # gain est decisif ici : l'ajustement evalue la trajectoire une centaine de
    # fois par cellule, et il y a autant de cellules que de particules suivies.
    A = champ['amplitude']
    L = champ['longueur']
    rb = champ['r_bord']
    if r_min is None:
        r_min = rb

    t = np.asarray(t, dtype=float)
    coef = pref_unit * re_k * A / (L * gamma)

    # Le mouvement ne commence qu'a partir du bord : en deca, le modele de
    # champ ne s'applique pas.
    if r0 <= rb:
        return np.full_like(t, rb)

    arg = math.exp((r0 - rb) / L) - coef * t / L

    # En dielectrophorese positive, la particule se rapproche du bord et
    # l'argument decroit. Il s'annule au moment ou elle l'atteint : elle s'y
    # arrete, car elle ne traverse pas le metal.
    atteint = arg <= 1e-12
    arg = np.maximum(arg, 1e-12)
    r = rb + L * np.log(arg)
    r = np.clip(r, r_min, r_max)
    if np.any(atteint):
        # Une fois le bord atteint, la position n'evolue plus
        premier = int(np.argmax(atteint))
        r[premier:] = r_min
    return r


def fenetre_ajustement(t, p):
    # Masque des instants retenus pour evaluer l'ecart au modele.
    t_min = p.get('t_fit_min', T_FIT_MIN_S)
    t_max = p.get('t_fit_max', T_FIT_MAX_S)
    m = np.ones(len(t), dtype=bool)
    if t_min is not None:
        m &= t >= t_min
    if t_max is not None:
        m &= t <= t_max
    # Si la fenetre est trop etroite, elle est ignoree plutot que de rendre
    # l'ajustement impossible.
    if int(np.sum(m)) < 8:
        return np.ones(len(t), dtype=bool), False
    return m, True


def ajuster_re_k(cellule, champ, p, rayon=None):
    # Ajuste Re[K] par minimisation de l'ecart entre trajectoire simulee et
    # trajectoire mesuree.
    #
    # Le probleme etant unidimensionnel et borne, la solution est unique et la
    # minimisation immediate. Le profil du cout est conserve : un minimum net
    # signale un ajustement bien contraint, un fond plat signale au contraire
    # que la mesure ne contraint pas la valeur.
    t, r_mes = cellule['t'], cellule['r']
    # Le rayon suit le choix declare, et non une regle implicite : c'est ce
    # qui permet de voir l'effet du parametre saisi lorsqu'on le modifie.
    if rayon is not None:
        a = rayon
    elif p.get('source_rayon', SOURCE_RAYON) == "fichier" and cellule['rayon']:
        a = cellule['rayon']
    else:
        a = p['r_algue']
    gamma, fconf, lam = coefficient_trainee(a, p['gap'], p['viscosite'])
    pref = prefacteur_unitaire(p, a)

    # La trajectoire entiere est simulee, mais l'ecart n'est evalue que sur la
    # fenetre retenue.
    masque, fenetre_ok = fenetre_ajustement(t, p)

    # POINT ESSENTIEL. Quand la fenetre ne commence pas au premier point, la
    # simulation doit partir de la position occupee AU DEBUT DE LA FENETRE, et
    # non de la position initiale.
    #
    # Integrer depuis le premier point tout en n'evaluant l'ecart qu'a partir
    # d'un instant ulterieur revient en effet a demander au modele de
    # reproduire une portion de trajectoire sans lui donner la position d'ou
    # elle part : l'optimisation compense alors l'erreur accumulee avant la
    # fenetre en faussant Re[K], parfois jusqu'a en inverser le signe.
    i0 = int(np.argmax(masque))
    t_ancre = float(t[i0])
    r_ancre = float(r_mes[i0])
    t_relatif = t - t_ancre

    def cout(re_k):
        r_sim = simuler_trajectoire(r_ancre, re_k, champ, pref, gamma,
                                    t_relatif)
        return float(np.mean((r_sim[masque] - r_mes[masque]) ** 2))

    # Recherche du minimum en deux temps.
    #
    # Une minimisation directe ne convient pas : en dielectrophorese positive
    # la particule atteint le bord de l'electrode et s'y arrete, de sorte que
    # le cout devient CONSTANT au-dela d'une certaine valeur. La fonction n'est
    # donc pas unimodale, et un algorithme qui le suppose peut se figer sur ce
    # plateau en ignorant le vrai minimum.
    #
    # Un balayage complet est fait d'abord, ce que la solution analytique rend
    # instantane, puis le minimum est raffine localement autour du meilleur
    # point de grille.
    essais = np.linspace(RE_K_MIN, RE_K_MAX, RE_K_POINTS)
    couts = np.array([cout(x) for x in essais])
    i_min = int(np.argmin(couts))
    pas = essais[1] - essais[0]
    borne_bas = max(RE_K_MIN, essais[i_min] - 2 * pas)
    borne_haut = min(RE_K_MAX, essais[i_min] + 2 * pas)

    if borne_haut - borne_bas > 1e-9:
        res = minimize_scalar(cout, bounds=(borne_bas, borne_haut),
                              method='bounded', options={'xatol': 1e-6})
        re_k = float(res.x) if res.fun <= couts[i_min] else float(essais[i_min])
        cout_min = min(float(res.fun), float(couts[i_min]))
    else:
        re_k = float(essais[i_min])
        cout_min = float(couts[i_min])

    r_fit = simuler_trajectoire(r_ancre, re_k, champ, pref, gamma, t_relatif)

    # Les indicateurs de qualite portent sur la fenetre d'ajustement, seule
    # portion que le modele est cense reproduire.
    rm, rf = r_mes[masque], r_fit[masque]
    ss_res = float(np.sum((rm - rf) ** 2))
    ss_tot = float(np.sum((rm - np.mean(rm)) ** 2))
    rmse = math.sqrt(ss_res / len(rm))

    # Ecart sur la portion exclue, pour mesurer ce que le modele n'explique pas
    rmse_hors = float('nan')
    if int(np.sum(~masque)) > 3:
        rmse_hors = math.sqrt(float(np.mean(
            (r_mes[~masque] - r_fit[~masque]) ** 2)))

    # Le coefficient de determination compare l'ecart residuel a la variance
    # des positions. Pour une cellule quasi immobile, cette variance se reduit
    # au bruit de pointage : le coefficient devient alors ininterpretable, et
    # peut meme etre negatif alors que l'ajustement est parfait. Dans ce cas
    # c'est l'ecart en micrometres qui fait foi.
    deplacement = float(np.max(rm) - np.min(rm))
    r2_valable = deplacement > 5.0 * rmse
    r2 = (1.0 - ss_res / ss_tot if ss_tot > 0 and r2_valable
          else float('nan'))


    # Incertitude approchee par la courbure du cout au voisinage du minimum
    incert = float('nan')
    i_min = int(np.argmin(couts))
    if 0 < i_min < len(essais) - 1:
        h = essais[1] - essais[0]
        courbure = (couts[i_min - 1] - 2 * couts[i_min]
                    + couts[i_min + 1]) / h ** 2
        if courbure > 0:
            incert = math.sqrt(2.0 * couts[i_min] / courbure)

    # Le minimum est-il identifiable, ou repose-t-il sur un plateau ?
    #
    # En dielectrophorese positive suffisamment forte, la cellule atteint le
    # bord de l'electrode des les premieres images et s'y arrete. Toutes les
    # trajectoires deviennent alors identiques, et la mesure ne distingue plus
    # une valeur d'une autre : le cout presente un plateau, et Re[K] n'a plus
    # qu'une borne inferieure. Le cas est detecte et signale, faute de quoi on
    # prendrait pour une mesure la premiere valeur du plateau.
    tolerance = 1e-6 * max(float(np.max(couts)), 1e-30)
    proches = np.abs(couts - couts[i_min]) <= tolerance
    largeur_plateau = float(essais[proches].max() - essais[proches].min())
    sur_plateau = largeur_plateau > 5 * (essais[1] - essais[0])

    # La cellule reste-t-elle dans le domaine ou le champ a ete calibre ?
    hors_bas = float(np.sum(r_mes < champ['r_min'])) / len(r_mes)
    hors_haut = float(np.sum(r_mes > champ['r_max'])) / len(r_mes)

    # Les bornes physiques sont-elles atteintes ?
    marge = 0.02
    borne = None
    if re_k < RE_K_MIN + marge:
        borne = "basse"
    elif re_k > RE_K_MAX - marge:
        borne = "haute"
    # Sortie du domaine physique d'une sphere homogene, signalee a part
    hors_physique = (re_k < RE_K_PHYSIQUE_MIN or re_k > RE_K_PHYSIQUE_MAX)

    return dict(nom=cellule['nom'], re_k=re_k, incertitude=incert, r2=r2,
                r2_valable=r2_valable,
                rmse=rmse, cout_min=cout_min, rayon=a,
                gamma=gamma, confinement=fconf, lam=lam,
                t=t, r_mes=r_mes, r_fit=r_fit, r0=cellule['r0'],
                masque_fit=masque, rmse_hors=rmse_hors,
                confinement_limite=(lam > 0.7),
                hors_physique=hors_physique,
                sur_plateau=sur_plateau, largeur_plateau=largeur_plateau,
                rayon_source=("fichier"
                              if (p.get('source_rayon', SOURCE_RAYON)
                                  == "fichier" and cellule['rayon'])
                              else "impose"),
                n_fit=int(np.sum(masque)), n_hors=int(np.sum(~masque)),
                t_ancre=t_ancre, r_ancre=r_ancre,
                essais=essais, couts=couts, borne=borne,
                hors_domaine=hors_bas + hors_haut,
                acquisition=cellule.get('acquisition', 0),
                retenue=cellule.get('retenue', True),
                deplacement=cellule.get('deplacement', 0.0))


def ajuster_toutes(cellules, champ, p, bavard=True):
    resultats = []
    for c in cellules:
        try:
            resultats.append(ajuster_re_k(c, champ, p))
        except Exception as e:
            print(f"  [ALERTE] {c['nom']} : ajustement impossible, {e}")
    if bavard and resultats:
        print("\n" + "=" * 78)
        print("AJUSTEMENT DE Re[K], CELLULE PAR CELLULE")
        print("=" * 78)
        print(f"{'cellule':>14} {'r0':>8} {'Re[K]':>9} {'incert.':>9} "
              f"{'R2':>8} {'ecart':>9} {'remarque':>18}")
        print("-" * 78)
        for r in resultats:
            rem = []
            if not r['retenue']:
                rem.append("peu mobile")
            if r['borne']:
                rem.append(f"borne {r['borne']}")
            if r['hors_domaine'] > 0.2:
                rem.append("hors domaine")
            if r.get('confinement_limite'):
                rem.append("confinement fort")
            if r.get('hors_physique'):
                rem.append("hors domaine physique")
            if r.get('sur_plateau'):
                rem.append(f"non identifiable, plateau "
                           f"{r['largeur_plateau']:.2f}")
            if not r.get('r2_valable', True):
                rem.append("R2 non defini")
            r2_txt = f"{r['r2']:8.4f}" if np.isfinite(r['r2']) else "       -"
            if np.isfinite(r.get('rmse_hors', np.nan)) and \
                    r['rmse_hors'] > 3 * r['rmse']:
                rem.append("derive hors fenetre")
            print(f"{r['nom']:>14} {r['r0']*1e6:>7.0f}u {r['re_k']:>+9.4f} "
                  f"{r['incertitude']:>9.4f} {r2_txt} "
                  f"{r['rmse']*1e6:>8.2f}u {', '.join(rem):>18}")
        vals = np.array([r['re_k'] for r in resultats if r['retenue']])
        if len(vals):
            print("-" * 78)
            print(f"{'population':>14} {'':>8} {np.mean(vals):>+9.4f} "
                  f"{'':>9} {'':>8} {'':>9}  ecart type "
                  f"{np.std(vals, ddof=1) if len(vals) > 1 else 0:.4f}")
        print("=" * 78)
    return resultats


# =============================================================================
# TRACES
# =============================================================================

def _anneau_h(ax, p):
    ax.axhspan(p['r_int'] * 1e6, p['r_ext'] * 1e6, alpha=0.22, color=COUL_OR)


def t_trajectoires(ax, choisis, p, champ, re_k_manuel=None):
    # Trajectoire mesuree et trajectoire ajustee, pour les cellules cochees.
    if not choisis:
        ax.axis('off')
        ax.text(0.5, 0.5, "cocher au moins une cellule", ha='center',
                va='center', color=COUL_GRIS, fontsize=11,
                transform=ax.transAxes)
        return
    coul = plt.cm.tab10(np.linspace(0, 1, max(len(choisis), 1)))
    for r, c in zip(choisis, coul):
        m = r.get('masque_fit')
        if m is not None and not np.all(m):
            # Portion exclue de l'ajustement, tracee en pale : elle est simulee
            # mais n'a pas contribue a determiner Re[K].
            ax.plot(r['t'][~m], r['r_mes'][~m] * 1e6, '-', color=c, lw=1.2,
                    alpha=0.30)
            ax.plot(r['t'][m], r['r_mes'][m] * 1e6, '-', color=c, lw=1.6,
                    alpha=0.95,
                    label=f"{r['nom']}, a = {r['rayon']*1e6:.1f} um, "
                          f"Re[K] = {r['re_k']:+.3f}")
        else:
            ax.plot(r['t'], r['r_mes'] * 1e6, '-', color=c, lw=1.4, alpha=0.95,
                    label=f"{r['nom']}, a = {r['rayon']*1e6:.1f} um, "
                          f"Re[K] = {r['re_k']:+.3f}")
        ax.plot(r['t'], r['r_fit'] * 1e6, '--', color=c, lw=2.2, alpha=0.85,
                dashes=(5, 3))

        # Trajectoire qu'on obtiendrait avec la valeur imposee, calculee depuis
        # le meme point d'ancrage que l'ajustement pour que la comparaison
        # porte sur la seule valeur de Re[K].
        if re_k_manuel is not None:
            gamma, _, _ = coefficient_trainee(r['rayon'], p['gap'],
                                              p['viscosite'])
            pref = prefacteur_unitaire(p, r['rayon'])
            r_man = simuler_trajectoire(
                r.get('r_ancre', r['r0']), re_k_manuel, champ, pref, gamma,
                r['t'] - r.get('t_ancre', 0.0))
            ax.plot(r['t'], r_man * 1e6, ':', color=c, lw=2.4, alpha=0.9)
            m = r.get('masque_fit')
            if m is None:
                m = np.ones(len(r['t']), dtype=bool)
            ecart = math.sqrt(float(np.mean(
                (r_man[m] - r['r_mes'][m]) ** 2)))
            ax.plot([], [], ':', color=c, lw=2.4,
                    label=f"impose {re_k_manuel:+.3f}, ecart "
                          f"{ecart*1e6:.1f} um")

        ax.plot([0], [r['r0'] * 1e6], 'o', color=c, ms=7,
                markeredgecolor='k', markeredgewidth=0.9, zorder=5)

    # Bornes de la fenetre d'ajustement
    t_min = p.get('t_fit_min')
    t_max = p.get('t_fit_max')
    if choisis:
        t_fin = max(float(r['t'][-1]) for r in choisis)
        if t_min is not None and t_min > 0:
            ax.axvspan(ax.get_xlim()[0], t_min, color='gray', alpha=0.12, lw=0)
            ax.axvline(t_min, color=COUL_GRIS, ls='--', lw=1.5)
        if t_max is not None and t_max < t_fin:
            ax.axvspan(t_max, t_fin * 1.02, color='gray', alpha=0.12, lw=0,
                       label="exclu de l'ajustement")
            ax.axvline(t_max, color=COUL_GRIS, ls='--', lw=1.5)
    ax.plot([], [], '-', color='k', lw=1.4, label="mesure")
    ax.plot([], [], '--', color='k', lw=2.2, dashes=(5, 3), label="ajustement")
    if re_k_manuel is not None:
        ax.plot([], [], ':', color='k', lw=2.4, label="valeur imposee")
    # Domaine ou le champ a ete calibre
    ax.axhspan(champ['r_min'] * 1e6, champ['r_max'] * 1e6, color=COUL_DIST,
               alpha=0.08, lw=0, label="domaine calibre")
    _anneau_h(ax, p)
    ax.set_xlabel("Temps depuis l'activation (s)")
    ax.set_ylabel("Position radiale (um)")
    ax.set_title("Trajectoires mesurees et ajustees", fontsize=11,
                 fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)


def t_cout(ax, choisis, re_k_manuel=None):
    # Profil du cout en fonction de la valeur d'essai de Re[K].
    #
    # Un minimum net signale un ajustement bien contraint. Un fond plat, ou un
    # minimum contre une borne, signale que la mesure ne determine pas la
    # valeur : c'est le controle a regarder avant d'accorder du credit au
    # chiffre obtenu.
    if not choisis:
        ax.axis('off')
        return
    coul = plt.cm.tab10(np.linspace(0, 1, max(len(choisis), 1)))
    for r, c in zip(choisis, coul):
        norm_c = r['couts'] / np.max(r['couts']) if np.max(r['couts']) > 0 \
            else r['couts']
        ax.plot(r['essais'], norm_c, '-', color=c, lw=1.8, alpha=0.9)
        marque = 'x' if r.get('sur_plateau') else 'v'
        taille = 11 if r.get('sur_plateau') else 9
        ax.plot([r['re_k']], [np.min(norm_c)], marque, color=c, ms=taille,
                markeredgecolor='k', markeredgewidth=0.9)
    ax.axvline(0, color='k', ls=':', lw=1.2, alpha=0.7)
    # La zone en deca de -0,5 est exploree mais grisee : aucune sphere
    # homogene ne peut s'y trouver.
    if RE_K_MIN < RE_K_PHYSIQUE_MIN:
        ax.axvspan(RE_K_MIN, RE_K_PHYSIQUE_MIN, color=COUL_ALER, alpha=0.10,
                   lw=0, label="hors domaine physique")
    ax.axvline(RE_K_PHYSIQUE_MIN, color=COUL_GRIS, ls='--', lw=1.6,
               label="bornes physiques")
    ax.axvline(RE_K_PHYSIQUE_MAX, color=COUL_GRIS, ls='--', lw=1.6)
    if re_k_manuel is not None:
        # La valeur imposee est reportee sur le profil : on lit ainsi de combien
        # elle degrade l'ecart par rapport a l'optimum.
        ax.axvline(re_k_manuel, color=COUL_ALER, ls='-', lw=2.2, alpha=0.85,
                   label=f"valeur imposee, {re_k_manuel:+.3f}")
    ax.set_xlabel("Valeur d'essai de Re[K]")
    ax.set_ylabel("Ecart quadratique, normalise")
    # Le cout sature loin du minimum : l'axe est borne sur la zone qui porte
    # l'information, faute de quoi les minima se confondraient avec l'axe.
    ax.set_ylim(-0.03, 1.05)

    # Sur une plage large, les minima se tassent au centre et deviennent
    # illisibles. L'axe horizontal est donc resserre autour des valeurs
    # trouvees, en gardant les bornes physiques en vue quand elles sont
    # proches.
    vals = [r['re_k'] for r in choisis]
    if re_k_manuel is not None:
        vals.append(re_k_manuel)
    if vals:
        centre_bas, centre_haut = min(vals), max(vals)
        marge = max(0.3, 2.5 * (centre_haut - centre_bas))
        bas = max(RE_K_MIN, min(centre_bas - marge, RE_K_PHYSIQUE_MIN - 0.1))
        haut = min(RE_K_MAX, max(centre_haut + marge,
                                 RE_K_PHYSIQUE_MAX + 0.1))
        ax.set_xlim(bas, haut)
        if bas > RE_K_MIN or haut < RE_K_MAX:
            ax.text(0.02, 0.04,
                    f"plage exploree : {RE_K_MIN:.0f} a {RE_K_MAX:.0f}",
                    transform=ax.transAxes, fontsize=7.5, color=COUL_GRIS)
    titre = ("Profil du cout\n"
             "un minimum net signale un ajustement bien contraint")
    if any(r.get('sur_plateau') for r in choisis):
        titre = ("Profil du cout\n"
                 "croix : minimum sur un plateau, valeur non identifiable")
    ax.set_title(titre, fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def t_distribution(ax, resultats, choisis=None):
    # Distribution des valeurs individuelles sur la population.
    retenus = [r for r in resultats if r['retenue']]
    if len(retenus) < 2:
        ax.axis('off')
        ax.text(0.5, 0.5, "trop peu de cellules pour une distribution",
                ha='center', va='center', color=COUL_GRIS,
                transform=ax.transAxes)
        return
    vals = np.array([r['re_k'] for r in retenus])
    n_classes = max(5, min(20, int(math.sqrt(len(vals)) * 2)))
    ax.hist(vals, bins=n_classes, density=True, alpha=0.60, color=COUL_DIST,
            edgecolor='k', linewidth=0.6, label=f"{len(vals)} cellules")
    if len(vals) > 2:
        mu, sigma = float(np.mean(vals)), float(np.std(vals, ddof=1))
        if sigma > 0:
            x = np.linspace(min(vals.min() - 3 * sigma, RE_K_MIN),
                            max(vals.max() + 3 * sigma, 0.2), 400)
            ax.plot(x, norm.pdf(x, mu, sigma), '-', color=COUL_FIT, lw=2.4,
                    label=f"moyenne {mu:+.3f}, ecart type {sigma:.3f}")
    ymax = ax.get_ylim()[1]
    ax.plot(vals, np.full_like(vals, -0.04 * ymax), '|', ms=13,
            color=COUL_MES, alpha=0.8)
    if choisis:
        sel = np.array([r['re_k'] for r in choisis])
        ax.plot(sel, np.full_like(sel, -0.04 * ymax), '|', ms=16, color='red',
                markeredgewidth=2.2, label="cellules cochees")
    ax.axvline(0, color='k', ls='--', lw=1.2)
    ax.set_xlabel("Re[K]")
    ax.set_ylabel("Densite de probabilite")
    ax.set_title("Distribution sur la population", fontsize=11,
                 fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def t_recapitulatif(ax, resultats, choisis, p, champ):
    ax.axis('off')
    retenus = [r for r in resultats if r['retenue']]
    vals = np.array([r['re_k'] for r in retenus]) if retenus else np.array([])
    li = [["Cellules lues", f"{len(resultats)}"],
          ["Retenues", f"{len(retenus)}"],
          ["Cochees", f"{len(choisis)}"]]
    if len(vals):
        li += [["Re[K] moyen", f"{np.mean(vals):+.4f}"],
               ["Re[K] median", f"{np.median(vals):+.4f}"],
               ["Ecart type",
                f"{np.std(vals, ddof=1):.4f}" if len(vals) > 1 else "0"],
               ["Etendue", f"{vals.min():+.3f} a {vals.max():+.3f}"]]
        if len(vals) > 1:
            li.append(["Erreur standard",
                       f"{np.std(vals, ddof=1)/math.sqrt(len(vals)):.4f}"])
    if choisis:
        li.append(["", ""])
        for r in choisis[:4]:
            q = (f"R2 = {r['r2']:.3f}" if np.isfinite(r['r2'])
                 else f"ecart {r['rmse']*1e6:.2f} um")
            li.append([r['nom'], f"Re[K] = {r['re_k']:+.4f}, {q}"])
    li += [["", ""],
           ["Rayon, source", p.get('source_rayon', SOURCE_RAYON)],
           ["Rayon employe",
            (f"{p['r_algue']*1e6:.2f} um"
             if p.get('source_rayon') == "impose" or not resultats
             else f"{np.median([r['rayon'] for r in resultats])*1e6:.2f} um "
                  f"median")],
           ["Conductivite du milieu", f"{p['sigma_milieu']:.4f} S/m"],
           ["Champ, longueur", f"{champ['longueur']*1e6:.1f} um"],
           ["Champ, potentiel", f"{champ['potentiel']:.2f} V"],
           ["Domaine calibre",
            f"{champ['r_min']*1e6:.0f} a {champ['r_max']*1e6:.0f} um"]]
    t = ax.table(cellText=li, colLabels=["Grandeur", "Valeur"], loc='center',
                 cellLoc='left')
    t.auto_set_font_size(False)
    t.set_fontsize(8.5)
    t.scale(1, 1.28)
    ax.set_title("Recapitulatif", fontsize=11, fontweight='bold', y=1.12)


def figure_population(resultats, p, champ):
    # Vue d'ensemble de la population, dans une fenetre separee.
    retenus = [r for r in resultats if r['retenue']]
    fig = plt.figure(figsize=(17, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.26)
    fig.suptitle(
        f"Caracterisation dielectrique de la population, "
        f"{len(retenus)} cellules retenues sur {len(resultats)}",
        fontsize=13, fontweight='bold')

    ax = fig.add_subplot(gs[0, 0])
    t_distribution(ax, resultats)

    # Re[K] en fonction de la position de depart : un lien systematique
    # signalerait un biais, le champ n'etant pas connu de la meme facon partout.
    ax = fig.add_subplot(gs[0, 1])
    if retenus:
        r0 = np.array([r['r0'] for r in retenus]) * 1e6
        vals = np.array([r['re_k'] for r in retenus])
        ax.plot(r0, vals, 'o', color=COUL_DIST, ms=8, markeredgecolor='k',
                markeredgewidth=0.8)
        if len(r0) > 2:
            c = float(np.corrcoef(r0, vals)[0, 1])
            pente = np.polyfit(r0, vals, 1)
            xx = np.linspace(r0.min(), r0.max(), 50)
            ax.plot(xx, np.polyval(pente, xx), '--', color=COUL_ALER, lw=1.8,
                    label=f"correlation {c:+.2f}")
            ax.legend(fontsize=8)
        ax.axvspan(champ['r_min'] * 1e6, champ['r_max'] * 1e6, color=COUL_DIST,
                   alpha=0.10, lw=0)
    ax.axhline(0, color='k', ls=':', lw=1.2)
    ax.set_xlabel("Position de depart (um)")
    ax.set_ylabel("Re[K]")
    ax.set_title("Re[K] en fonction de la position de depart\n"
                 "une correlation signalerait un biais", fontsize=11,
                 fontweight='bold')
    ax.grid(alpha=0.3)

    # Qualite des ajustements
    ax = fig.add_subplot(gs[1, 0])
    if retenus:
        noms = [r['nom'] for r in retenus]
        r2 = [r['r2'] if np.isfinite(r['r2']) else 0.0 for r in retenus]
        x = np.arange(len(retenus))
        coul = [COUL_DIST if v > 0.9 else (COUL_OR if v > 0.7 else COUL_ALER)
                for v in r2]
        ax.bar(x, r2, color=coul, edgecolor='k', linewidth=0.6)
        ax.axhline(0.9, color='k', ls='--', lw=1.2, label="seuil de 0,90")
        ax.set_xticks(x)
        ax.set_xticklabels(noms, rotation=60, ha='right', fontsize=7)
        ax.set_ylabel("Coefficient de determination")
        ax.set_title("Qualite de chaque ajustement", fontsize=11,
                     fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis='y')

    ax = fig.add_subplot(gs[1, 1])
    t_recapitulatif(ax, resultats, [], p, champ)
    return fig


# =============================================================================
# INTERFACE
# =============================================================================

class Application:

    def __init__(self, racine):
        self.racine = racine
        self.cellules = []
        self.resultats = []
        self.champ = None
        self.p = None
        self.chemin_champ = None
        self.vars_coche = {}
        self.lignes_liste = []
        self.canevas_liste = None
        # Rendu du defilement au panneau principal, defini a la construction de
        # celui-ci et appele quand le pointeur quitte la liste des cellules.
        self._molette_panneau = lambda: None

        racine.title("Caracterisation dielectrique des algues")
        racine.geometry("1560x850")
        self._construire_controles()
        self._construire_graphiques()

    # --------------------------------------------------------------- controles
    def _construire_controles(self):
        cadre = tk.Frame(self.racine, bg='#ECECEC', padx=10, pady=8)
        cadre.pack(side=tk.LEFT, fill=tk.Y)
        canevas = tk.Canvas(cadre, bg='#ECECEC', width=300,
                            highlightthickness=0)
        barre = ttk.Scrollbar(cadre, orient="vertical", command=canevas.yview)
        interieur = tk.Frame(canevas, bg='#ECECEC')
        interieur.bind("<Configure>", lambda e: canevas.configure(
            scrollregion=canevas.bbox("all")))
        canevas.create_window((0, 0), window=interieur, anchor="nw")
        canevas.configure(yscrollcommand=barre.set)
        canevas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        barre.pack(side=tk.RIGHT, fill=tk.Y)

        def molette_panneau(evenement=None):
            if evenement is None:
                # Retablissement apres passage sur la liste des cellules
                for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                    canevas.bind_all(sequence, molette_panneau)
                return
            if getattr(evenement, 'num', None) == 4:
                pas = -1
            elif getattr(evenement, 'num', None) == 5:
                pas = 1
            else:
                pas = int(-evenement.delta / 120) or (
                    -1 if evenement.delta > 0 else 1)
            canevas.yview_scroll(pas, "units")

        self._molette_panneau = molette_panneau
        molette_panneau()

        tk.Label(interieur, text="PARAMETRES", font=('Arial', 12, 'bold'),
                 bg='#ECECEC').pack(pady=(0, 6))
        self.saisie = {}

        def bloc(titre, entrees):
            ttk.Separator(interieur, orient='horizontal').pack(fill='x', pady=3)
            tk.Label(interieur, text=titre, font=('Arial', 9, 'bold'),
                     bg='#ECECEC').pack(anchor='w')
            for cle, lib, dft in entrees:
                f = tk.Frame(interieur, bg='#ECECEC')
                f.pack(fill='x', pady=1)
                tk.Label(f, text=lib, font=('Arial', 8), bg='#ECECEC',
                         width=25, anchor='w').pack(side=tk.LEFT)
                e = tk.Entry(f, font=('Arial', 9), width=9)
                e.insert(0, str(dft))
                e.pack(side=tk.LEFT)
                self.saisie[cle] = e

        bloc("Cellules", [('r_algue', "Rayon impose (um)", RAYON_ALGUE_UM)])
        f = tk.Frame(interieur, bg='#ECECEC')
        f.pack(fill='x', pady=1)
        tk.Label(f, text="Source du rayon", font=('Arial', 8), bg='#ECECEC',
                 width=25, anchor='w').pack(side=tk.LEFT)
        self.var_source_rayon = tk.StringVar(master=self.racine,
                                             value=SOURCE_RAYON)
        ttk.Combobox(f, textvariable=self.var_source_rayon, width=7,
                     state="readonly",
                     values=["fichier", "impose"]).pack(side=tk.LEFT)
        bloc("Milieu", [
            ('eps_milieu', "Permittivite relative", EPS_MILIEU_REL),
            ('sigma_milieu', "Conductivite (S/m)", SIGMA_MILIEU),
            ('viscosite', "Viscosite (mPa.s)", VISCOSITE_MPA_S)])
        bloc("Geometrie", [
            ('r_int', "Rayon interieur (um)", R_INT_UM),
            ('r_ext', "Rayon exterieur (um)", R_EXT_UM),
            ('gap', "Hauteur de cavite (um)", HAUTEUR_CAV_UM)])
        bloc("Champ calibre", [
            ('champ_amp', "Carre du champ au bord", CHAMP_AMPLITUDE),
            ('champ_lon', "Longueur (um)", CHAMP_LONGUEUR_UM),
            ('val_min', "Domaine valide, min (um)", CHAMP_VALIDITE_MIN_UM),
            ('val_max', "Domaine valide, max (um)", CHAMP_VALIDITE_MAX_UM)])
        bloc("Traitement", [
            ('seuil', "Seuil d'activation (um/s)", SEUIL_ACTIVATION_UM_S),
            ('depl_min', "Deplacement minimal (um)", DEPLACEMENT_MIN_UM),
            ('frequence', "Frequence (kHz)", FREQUENCE_KHZ)])
        # --- Fenetre d'ajustement, reglee par curseurs ---------------------
        #
        # Deux curseurs plutot que deux champs de saisie : l'effet du choix se
        # lit alors directement sur la courbe, ce qui est le seul moyen de
        # juger si la portion exclue l'est a bon droit.
        ttk.Separator(interieur, orient='horizontal').pack(fill='x', pady=4)
        tk.Label(interieur, text="FENETRE D'AJUSTEMENT",
                 font=('Arial', 9, 'bold'), bg='#ECECEC').pack(anchor='w')
        self.var_tmin = tk.DoubleVar(master=self.racine, value=0.0)
        self.var_tmax = tk.DoubleVar(master=self.racine, value=20.0)
        self.curseur_tmin = tk.Scale(
            interieur, from_=0.0, to=20.0, resolution=0.1,
            orient=tk.HORIZONTAL, variable=self.var_tmin, length=270,
            label="debut (s)", font=('Arial', 8), bg='#ECECEC',
            command=lambda v: self._marquer_a_refaire())
        self.curseur_tmin.pack(fill='x')
        self.curseur_tmax = tk.Scale(
            interieur, from_=0.0, to=20.0, resolution=0.1,
            orient=tk.HORIZONTAL, variable=self.var_tmax, length=270,
            label="fin (s)", font=('Arial', 8), bg='#ECECEC',
            command=lambda v: self._marquer_a_refaire())
        self.curseur_tmax.pack(fill='x')
        self.bouton_refit = tk.Button(
            interieur, text="reajuster sur cette fenetre",
            command=self._reajuster, bg='#2E7D32', fg='white',
            font=('Arial', 8, 'bold'), width=28)
        self.bouton_refit.pack(pady=2)

        # --- Valeur imposee manuellement -----------------------------------
        #
        # Permet de comparer la trajectoire qu'on obtiendrait avec une valeur
        # choisie a celle que l'ajustement retient. C'est utile pour se faire
        # une idee de la sensibilite : si deux valeurs eloignees donnent des
        # trajectoires indiscernables, c'est que la mesure ne les distingue
        # pas, quel que soit le minimum trouve.
        ttk.Separator(interieur, orient='horizontal').pack(fill='x', pady=4)
        tk.Label(interieur, text="Re[K] IMPOSE", font=('Arial', 9, 'bold'),
                 bg='#ECECEC').pack(anchor='w')
        self.var_manuel = tk.BooleanVar(master=self.racine, value=False)
        tk.Checkbutton(interieur, text="afficher une valeur imposee",
                       variable=self.var_manuel, bg='#ECECEC',
                       font=('Arial', 8),
                       command=self._rafraichir).pack(anchor='w')
        self.var_rek = tk.DoubleVar(master=self.racine, value=-0.20)
        tk.Scale(interieur, from_=RE_K_MIN, to=RE_K_MAX,
                 resolution=RE_K_PAS_CURSEUR,
                 orient=tk.HORIZONTAL, variable=self.var_rek, length=270,
                 label="valeur de Re[K]", font=('Arial', 8), bg='#ECECEC',
                 command=lambda v: self._rafraichir()).pack(fill='x')

        ttk.Separator(interieur, orient='horizontal').pack(fill='x', pady=5)
        for texte, cmd, coul in [
                ("1. Charger le champ calibre", self.charger_champ, '#6A1B9A'),
                ("2. Charger le suivi des algues", self.charger_suivi,
                 '#6A1B9A'),
                ("3. AJUSTER Re[K]", self.ajuster, '#2E7D32'),
                ("Vue de la population", self.ouvrir_population, '#1565C0'),
                ("Exporter", self.exporter, '#EF6C00')]:
            tk.Button(interieur, text=texte, command=cmd, bg=coul, fg='white',
                      font=('Arial', 9, 'bold'), width=28).pack(pady=2)

        # Liste des cellules a cocher
        ttk.Separator(interieur, orient='horizontal').pack(fill='x', pady=5)
        tk.Label(interieur, text="CELLULES A AFFICHER",
                 font=('Arial', 9, 'bold'), bg='#ECECEC').pack(anchor='w')
        f = tk.Frame(interieur, bg='#ECECEC')
        f.pack(fill='x', pady=2)
        for texte, cmd in [("tout", lambda: self._cocher(True)),
                           ("rien", lambda: self._cocher(False)),
                           ("la 1re", lambda: self._cocher_une()),
                           ("10 par 10", lambda: self._tranche_suivante())]:
            tk.Button(f, text=texte, command=cmd, font=('Arial', 8),
                      width=7).pack(side=tk.LEFT, padx=1)

        self.cadre_nav = tk.Frame(interieur, bg='#ECECEC')
        self.cadre_nav.pack(fill='x', pady=1)
        self.cadre_liste = tk.Frame(interieur, bg='white', relief='sunken',
                                    bd=1)
        self.cadre_liste.pack(fill='x', pady=3)
        tk.Label(self.cadre_liste, text="charger un suivi",
                 font=('Arial', 8), bg='white', fg='gray').pack(pady=8)

        cadre_diag = tk.LabelFrame(interieur, text="Diagnostics",
                                   font=('Arial', 9, 'bold'), bg='#ECECEC')
        cadre_diag.pack(fill='x', pady=5)
        self.texte_diag = tk.Text(cadre_diag, height=11, width=36,
                                  font=('Courier', 8), bg='white')
        self.texte_diag.pack(padx=3, pady=3)
        self._diag("Charger le champ, puis le suivi,\npuis lancer "
                   "l'ajustement.")

    def _construire_graphiques(self):
        cadre = tk.Frame(self.racine)
        cadre.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.fig = plt.figure(figsize=(13, 8.4))
        gs = GridSpec(2, 2, figure=self.fig, hspace=0.32, wspace=0.24,
                      left=0.07, right=0.97, top=0.94, bottom=0.07)
        self.ax_traj = self.fig.add_subplot(gs[0, 0])
        self.ax_cout = self.fig.add_subplot(gs[0, 1])
        self.ax_dist = self.fig.add_subplot(gs[1, 0])
        self.ax_reca = self.fig.add_subplot(gs[1, 1])
        self.canvas = FigureCanvasTkAgg(self.fig, master=cadre)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        for ax in (self.ax_traj, self.ax_cout, self.ax_dist, self.ax_reca):
            ax.axis('off')
        self.canvas.draw_idle()

    def _diag(self, texte):
        self.texte_diag.config(state='normal')
        self.texte_diag.delete('1.0', tk.END)
        self.texte_diag.insert('1.0', texte)
        self.texte_diag.config(state='disabled')

    # -------------------------------------------------------------- parametres
    def _lire_parametres(self):
        c = self.saisie
        return dict(
            r_algue=float(c['r_algue'].get()) * 1e-6,
            source_rayon=self.var_source_rayon.get(),
            eps_milieu=float(c['eps_milieu'].get()) * EPSILON_0,
            eps_milieu_rel=float(c['eps_milieu'].get()),
            sigma_milieu=float(c['sigma_milieu'].get()),
            viscosite=float(c['viscosite'].get()) * 1e-3,
            r_int=float(c['r_int'].get()) * 1e-6,
            r_ext=float(c['r_ext'].get()) * 1e-6,
            gap=float(c['gap'].get()) * 1e-6,
            champ_amplitude=float(c['champ_amp'].get()),
            champ_longueur=float(c['champ_lon'].get()) * 1e-6,
            validite_min=float(c['val_min'].get()) * 1e-6,
            validite_max=float(c['val_max'].get()) * 1e-6,
            seuil=float(c['seuil'].get()) * 1e-6,
            deplacement_min=float(c['depl_min'].get()) * 1e-6,
            frequence=float(c['frequence'].get()) * 1e3,
            t_fit_min=float(self.var_tmin.get()),
            t_fit_max=float(self.var_tmax.get()),
            mode_recalage=MODE_RECALAGE)

    # ----------------------------------------------------------- chargements
    def charger_champ(self):
        chemin = filedialog.askopenfilename(
            title="Champ calibre (champ_reconstruit.csv)",
            filetypes=[("Fichiers CSV", "*.csv")])
        if chemin:
            self.chemin_champ = chemin
            messagebox.showinfo(
                "Champ", f"{os.path.basename(chemin)} sera employe.\n\n"
                         f"Sans fichier, les parametres saisis font foi.")

    def charger_suivi(self):
        chemin = filedialog.askopenfilename(
            title="Suivi des algues",
            filetypes=[("Fichiers de suivi", "*.csv *.xlsx *.xls"),
                       ("Fichiers CSV", "*.csv"),
                       ("Classeurs Excel", "*.xlsx *.xls"),
                       ("Tous les fichiers", "*.*")])
        if not chemin:
            return
        try:
            p = self._lire_parametres()
        except Exception as e:
            messagebox.showerror("Erreur", f"Parametre invalide :\n\n{e}")
            return
        self.p = p
        self.cellules = lire_algues(chemin, p)
        if not self.cellules:
            messagebox.showwarning(
                "Suivi",
                "Aucune cellule exploitable.\n\n"
                "La cause la plus frequente est un seuil d'activation trop "
                "eleve pour ce jeu : les cellules s'y deplacent plus lentement "
                "que le seuil ne le suppose.\n\n"
                "La console indique les vitesses observees et propose une "
                "valeur adaptee.")
            return
        self._construire_liste()
        self._bornes_curseurs()
        self._diag(f"{len(self.cellules)} cellules lues.\n"
                   f"{sum(1 for c in self.cellules if c['retenue'])} retenues.\n"
                   f"\nLancer l'ajustement.")

    def _construire_liste(self):
        # Liste des cellules, avec une case a cocher par cellule.
        #
        # La zone est defilante et possede son PROPRE traitement de la molette :
        # sans cela, la molette ferait defiler le panneau de parametres qui
        # l'englobe, et les cellules de numero eleve resteraient inaccessibles.
        # Le traitement est active a l'entree du pointeur dans la zone et rendu
        # a sa sortie.
        for w in self.cadre_liste.winfo_children():
            w.destroy()
        self.vars_coche = {}

        canevas = tk.Canvas(self.cadre_liste, bg='white', height=210,
                            highlightthickness=0)
        barre = ttk.Scrollbar(self.cadre_liste, orient="vertical",
                              command=canevas.yview)
        inter = tk.Frame(canevas, bg='white')
        inter.bind("<Configure>", lambda e: canevas.configure(
            scrollregion=canevas.bbox("all")))
        canevas.create_window((0, 0), window=inter, anchor="nw")
        canevas.configure(yscrollcommand=barre.set)
        canevas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        barre.pack(side=tk.RIGHT, fill=tk.Y)

        def molette(evenement):
            # Windows et macOS renvoient delta, X11 renvoie les boutons 4 et 5
            if getattr(evenement, 'num', None) == 4:
                pas = -1
            elif getattr(evenement, 'num', None) == 5:
                pas = 1
            else:
                pas = int(-evenement.delta / 120) or (
                    -1 if evenement.delta > 0 else 1)
            canevas.yview_scroll(pas, "units")
            return "break"

        def entrer(_):
            for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                canevas.bind_all(sequence, molette)

        def sortir(_):
            for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                canevas.unbind_all(sequence)
            # Le panneau englobant retrouve son propre defilement
            if hasattr(self, '_molette_panneau'):
                self._molette_panneau()

        canevas.bind("<Enter>", entrer)
        canevas.bind("<Leave>", sortir)

        self.lignes_liste = []
        for i, c in enumerate(self.cellules):
            v = tk.BooleanVar(master=self.racine, value=(i == 0))
            v.trace_add('write', lambda *a: self._rafraichir())
            self.vars_coche[c['nom']] = v
            texte = f"{i+1:>3}. {c['nom']}  r0 {c['r0']*1e6:.0f} um"
            if c.get('rayon'):
                texte += f"  a {c['rayon']*1e6:.1f} um"
            if not c['retenue']:
                texte += "  peu mobile"
            cb = tk.Checkbutton(inter, text=texte, variable=v, bg='white',
                                font=('Arial', 8), anchor='w',
                                fg=('black' if c['retenue'] else COUL_GRIS))
            cb.pack(fill='x', anchor='w')
            self.lignes_liste.append(cb)

        # Navigation rapide par dizaines, pour les jeux nombreux
        for w in self.cadre_nav.winfo_children():
            w.destroy()
        n = len(self.cellules)
        if n > 12:
            tk.Label(self.cadre_nav, text="aller a :", font=('Arial', 8),
                     bg='#ECECEC').pack(side=tk.LEFT)
            for debut in range(0, n, 10):
                fin = min(debut + 10, n)
                tk.Button(self.cadre_nav, text=f"{debut+1}", width=3,
                          font=('Arial', 7),
                          command=lambda d=debut: self._aller_a(d, canevas)
                          ).pack(side=tk.LEFT, padx=1)

        self.canevas_liste = canevas

    def _aller_a(self, indice, canevas):
        # Fait defiler la liste jusqu'a la cellule demandee.
        canevas.update_idletasks()
        total = max(len(self.cellules), 1)
        canevas.yview_moveto(min(indice / total, 1.0))

    def _cocher_plage(self, debut, fin):
        # Coche une tranche de cellules et decoche les autres.
        for i, (nom, v) in enumerate(self.vars_coche.items()):
            v.set(debut <= i < fin)

    def _cocher(self, etat):
        for v in self.vars_coche.values():
            v.set(etat)

    def _tranche_suivante(self):
        # Fait defiler les cellules par groupes de dix, ce qui evite de cocher
        # cinquante cases une a une pour parcourir un jeu nombreux.
        n = len(self.vars_coche)
        if not n:
            return
        coches = [i for i, v in enumerate(self.vars_coche.values()) if v.get()]
        debut = (min(coches) + 10) if coches else 0
        if debut >= n:
            debut = 0
        self._cocher_plage(debut, min(debut + 10, n))
        if self.canevas_liste is not None:
            self._aller_a(debut, self.canevas_liste)

    def _cocher_une(self):
        for i, (nom, v) in enumerate(self.vars_coche.items()):
            v.set(i == 0)

    # ---------------------------------------------------------------- calcul
    def ajuster(self):
        if not self.cellules:
            messagebox.showinfo("Ajustement", "Charger d'abord un suivi.")
            return
        try:
            self.p = self._lire_parametres()
        except Exception as e:
            messagebox.showerror("Erreur", f"Parametre invalide :\n\n{e}")
            return

        self.champ = charger_champ(self.chemin_champ, self.p)
        self._diag("Ajustement en cours...")
        self.racine.update()

        self.resultats = ajuster_toutes(self.cellules, self.champ, self.p)
        retenus = [r for r in self.resultats if r['retenue']]
        vals = np.array([r['re_k'] for r in retenus]) if retenus else np.array([])

        # La trainee affichee correspond au rayon median effectivement employe
        r_typique = (np.median([r['rayon'] for r in self.resultats])
                     if self.resultats else self.p['r_algue'])
        gamma, fconf, lam = coefficient_trainee(r_typique, self.p['gap'],
                                                self.p['viscosite'])
        alertes = []
        n_borne = sum(1 for r in self.resultats if r['borne'])
        if n_borne:
            alertes.append(f"{n_borne} contre une borne")
        n_hors = sum(1 for r in self.resultats if r['hors_domaine'] > 0.2)
        if n_hors:
            alertes.append(f"{n_hors} hors domaine")
        n_mauvais = sum(1 for r in retenus
                        if np.isfinite(r['r2']) and r['r2'] < 0.9)
        if n_mauvais:
            alertes.append(f"{n_mauvais} R2 sous 0,90")
        n_conf = sum(1 for r in self.resultats if r.get('confinement_limite'))
        if n_conf:
            alertes.append(f"{n_conf} confinement > 0,7")
        n_phys = sum(1 for r in self.resultats if r.get('hors_physique'))
        if n_phys:
            alertes.append(f"{n_phys} hors domaine physique")
        n_plat = sum(1 for r in self.resultats if r.get('sur_plateau'))
        if n_plat:
            alertes.append(f"{n_plat} sur un plateau, non identifiables")

        self._diag(
            f"{len(self.resultats)} cellules ajustees\n"
            f"{len(retenus)} retenues\n"
            f"{'-'*30}\n"
            + (f"Re[K] moyen : {np.mean(vals):+.4f}\n"
               f"ecart type  : "
               f"{np.std(vals, ddof=1) if len(vals) > 1 else 0:.4f}\n"
               f"etendue     : {vals.min():+.3f} a {vals.max():+.3f}\n"
               if len(vals) else "")
            + f"{'-'*30}\n"
            + (f"rayon impose : {self.p['r_algue']*1e6:.2f} um\n"
               if self.p['source_rayon'] == "impose" else
               f"rayons du fichier :\n"
               f"  de {min(r['rayon'] for r in self.resultats)*1e6:.2f} a "
               f"{max(r['rayon'] for r in self.resultats)*1e6:.2f} um\n"
               f"  median "
               f"{np.median([r['rayon'] for r in self.resultats])*1e6:.2f} um\n")
            + f"trainee  : {gamma:.3e}\n"
            f"confin.  : {fconf:.3f}\n"
            f"{'-'*30}\n"
            f"champ, longueur : {self.champ['longueur']*1e6:.1f} um\n"
            f"champ, potentiel: {self.champ['potentiel']:.2f} V\n"
            f"domaine : {self.champ['r_min']*1e6:.0f} a "
            f"{self.champ['r_max']*1e6:.0f} um\n"
            f"{'-'*30}\n"
            f"fenetre de fit : "
            f"{self.p['t_fit_min'] if self.p['t_fit_min'] is not None else 0:.1f}"
            f" a "
            + (f"{self.p['t_fit_max']:.1f} s\n"
               if self.p['t_fit_max'] is not None else "la fin\n")
            + (f"points retenus : {sum(r['n_fit'] for r in self.resultats)}\n"
               f"points exclus  : {sum(r['n_hors'] for r in self.resultats)}\n"
               if self.resultats else "")
            + (f"{'-'*30}\nALERTES\n  " + "\n  ".join(alertes)
               if alertes else ""))
        self._rafraichir()

    def _marquer_a_refaire(self, *args):
        # Signale que la fenetre a change sans relancer le calcul : deplacer un
        # curseur ne doit pas declencher cinquante ajustements a chaque pixel.
        if hasattr(self, 'bouton_refit') and self.resultats:
            self.bouton_refit.config(bg='#EF6C00',
                                     text="reajuster, fenetre modifiee")
        self._rafraichir()

    def _reajuster(self):
        # Relance l'ajustement sur la fenetre courante.
        if not self.cellules:
            messagebox.showinfo("Reajustement", "Charger d'abord un suivi.")
            return
        if self.var_tmax.get() <= self.var_tmin.get():
            messagebox.showwarning("Fenetre",
                                   "La fin doit etre posterieure au debut.")
            return
        self.ajuster()
        self.bouton_refit.config(bg='#2E7D32',
                                 text="reajuster sur cette fenetre")

    def _bornes_curseurs(self):
        # Cale la course des curseurs sur la duree reellement disponible.
        if not self.cellules:
            return
        t_fin = max(float(c['t'][-1]) for c in self.cellules)
        for cur in (self.curseur_tmin, self.curseur_tmax):
            cur.config(to=round(t_fin, 1))
        if self.var_tmax.get() > t_fin or self.var_tmax.get() <= 0:
            self.var_tmax.set(round(t_fin, 1))

    def _rafraichir(self, *args):
        if not self.resultats:
            return
        choisis = [r for r in self.resultats
                   if self.vars_coche.get(r['nom'])
                   and self.vars_coche[r['nom']].get()]
        for ax in (self.ax_traj, self.ax_cout, self.ax_dist, self.ax_reca):
            ax.clear()
            ax.axis('on')
        manuel = (float(self.var_rek.get()) if self.var_manuel.get()
                  else None)
        t_trajectoires(self.ax_traj, choisis, self.p, self.champ, manuel)
        t_cout(self.ax_cout, choisis, manuel)
        t_distribution(self.ax_dist, self.resultats, choisis)
        t_recapitulatif(self.ax_reca, self.resultats, choisis, self.p,
                        self.champ)
        self.canvas.draw_idle()

    # -------------------------------------------------------------- fenetres
    def ouvrir_population(self):
        if not self.resultats:
            messagebox.showinfo("Population", "Lancer d'abord l'ajustement.")
            return
        fig = figure_population(self.resultats, self.p, self.champ)
        try:
            fig.canvas.manager.show()
        except Exception:
            plt.show(block=False)
        plt.pause(0.001)

    def exporter(self):
        if not self.resultats:
            messagebox.showinfo("Export", "Lancer d'abord l'ajustement.")
            return
        dossier = filedialog.askdirectory(title="Dossier de sortie")
        if not dossier:
            return

        self.fig.savefig(os.path.join(dossier, "01-interface.svg"),
                         format='svg', bbox_inches='tight', facecolor='white')
        figure_population(self.resultats, self.p, self.champ).savefig(
            os.path.join(dossier, "02-population.svg"), format='svg',
            bbox_inches='tight', facecolor='white')

        lignes = [dict(cellule=r['nom'], acquisition=r['acquisition'] + 1,
                       r0_um=r['r0'] * 1e6, rayon_um=r['rayon'] * 1e6,
                       re_k=r['re_k'], incertitude=r['incertitude'],
                       r2=r['r2'], rmse_um=r['rmse'] * 1e6,
                       deplacement_um=r['deplacement'] * 1e6,
                       retenue=r['retenue'],
                       borne_atteinte=r['borne'] or "",
                       part_hors_domaine=r['hors_domaine'],
                       n_points_fit=r['n_fit'], n_points_exclus=r['n_hors'],
                       rmse_hors_fenetre_um=(r['rmse_hors'] * 1e6
                                             if np.isfinite(r['rmse_hors'])
                                             else None))
                  for r in self.resultats]
        pd.DataFrame(lignes).to_csv(
            os.path.join(dossier, "resultats_ReK.csv"), index=False)

        traj = []
        for r in self.resultats:
            for t, rm, rf in zip(r['t'], r['r_mes'], r['r_fit']):
                traj.append(dict(cellule=r['nom'], temps_s=t,
                                 r_mesure_um=rm * 1e6, r_ajuste_um=rf * 1e6))
        pd.DataFrame(traj).to_csv(
            os.path.join(dossier, "trajectoires.csv"), index=False)

        retenus = [r for r in self.resultats if r['retenue']]
        vals = [r['re_k'] for r in retenus]
        meta = dict(
            origine="Caracterisation_ReK_Algues_Interface.py",
            methode=("ajustement de Re[K] sur le deplacement mesure, dans le "
                     "champ calibre par billes de reference"),
            cellules=dict(rayon_impose_um=self.p['r_algue'] * 1e6,
                          source_rayon=self.p['source_rayon'],
                          rayon_median_employe_um=float(np.median(
                              [r['rayon'] for r in self.resultats])) * 1e6,
                          n_lues=len(self.resultats), n_retenues=len(retenus)),
            milieu=dict(eps_relative=self.p['eps_milieu_rel'],
                        sigma_S_par_m=self.p['sigma_milieu'],
                        viscosite_Pa_s=self.p['viscosite']),
            geometrie=dict(r_interieur_um=self.p['r_int'] * 1e6,
                           r_exterieur_um=self.p['r_ext'] * 1e6,
                           hauteur_cavite_um=self.p['gap'] * 1e6),
            fenetre_ajustement=dict(
                debut_s=self.p['t_fit_min'], fin_s=self.p['t_fit_max']),
            champ=dict(amplitude_V2_par_m2=self.champ['amplitude'],
                       longueur_um=self.champ['longueur'] * 1e6,
                       potentiel_V=self.champ['potentiel'],
                       domaine_um=[self.champ['r_min'] * 1e6,
                                   self.champ['r_max'] * 1e6],
                       source=self.champ['source']),
            resultats=dict(
                re_k_moyen=float(np.mean(vals)) if vals else None,
                re_k_median=float(np.median(vals)) if vals else None,
                ecart_type=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0,
                minimum=float(np.min(vals)) if vals else None,
                maximum=float(np.max(vals)) if vals else None),
            avertissement=("Re[K] est une propriete de la cellule RELATIVEMENT "
                           "a son milieu : les valeurs sont indissociables de "
                           "la conductivite employee."))
        with open(os.path.join(dossier, "resultats.json"), 'w',
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
        racine.quit()
        racine.destroy()
        plt.close('all')

    racine.protocol("WM_DELETE_WINDOW", fermer)
    print("Interface prete.")
    print("  1. charger le champ calibre")
    print("  2. charger le suivi des algues")
    print("  3. lancer l'ajustement, puis cocher les cellules a afficher")
    racine.mainloop()


if __name__ == "__main__":
    principal()