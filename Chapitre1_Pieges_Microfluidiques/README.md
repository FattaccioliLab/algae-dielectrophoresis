# Chapitre 1 : dispositif microfluidique de piégeage et de libération

Ce dossier rassemble les outils associés au premier chapitre de la thèse, consacré au développement d'une plateforme capable de piéger un grand nombre de particules puis de les libérer à la demande.

Le principe retenu, baptisé *PULL*, repose sur une membrane en PDMS qui porte l'ensemble d'une matrice de pièges en U. Une dépression appliquée dans une cavité pneumatique située au-dessus soulève cette membrane, et avec elle tous les pièges, ce qui libère les particules d'un seul geste, sans valve individuelle ni inversion du flux. Le chapitre présente une preuve de concept sur des particules d'environ 80 µm, puis la miniaturisation du dispositif vers l'échelle cellulaire, autour de 10 µm.

---

## Contenu

| Dossier | Fichier | Rôle |
|---|---|---|
| `conception/` | *à compléter* | Masques de photolithographie et moules imprimés |
| `analyse_membrane/` | `Analyse_Franges_Fizeau.py` | Déformation de la membrane mesurée par interférométrie de Fizeau |
| `analyse_pieges/` | `Analyse_Pieges.py` | Détection des particules et suivi du remplissage de la matrice |
| `analyse_pieges/` | `Comparaison_Cycles.py` | Comparaison de cycles successifs de capture et de libération |
| `automatisation/` | *à compléter* | Pilotage de la plateforme (imagerie, platine, fluidique) |

---

## `conception/`

Fichiers de fabrication des puces :

- les designs AutoCAD des masques, pour les puces de 80 µm (masque souple) et de 10 µm (masque en chrome), exportés au format `.dxf` ;
- le fichier `.stl` du moule de la cavité pneumatique, imprimé en résine (Mars 2 Pro, Elegoo).

Les paramètres de lithographie (épaisseurs de résine, vitesses de centrifugation, doses d'insolation) sont donnés dans le tableau de fabrication du chapitre 1 du manuscrit.

---

## `analyse_membrane/Analyse_Franges_Fizeau.py`

**Ce qu'il fait.** Il reconstruit la déformation de la membrane des puces de 10 µm à partir des franges de Fizeau observées au profilomètre optique. Le long d'un segment tracé par l'utilisateur, les maxima d'intensité sont détectés, l'interfrange local donne la pente de la membrane, et son intégration donne le profil de déflexion. L'analyse est conduite sur le grand axe puis sur le petit axe de la cavité elliptique, et les deux profils sont combinés en une surface tridimensionnelle. La flèche maximale est ensuite confrontée au modèle de membrane épaisse (Schomburg), avec la largeur géométrique puis avec une largeur effective ajustée.

**Entrées.** Deux dossiers d'images acquises à différents paliers de dépression, l'un pour le grand axe, l'autre pour le petit axe. Le nom de chaque image doit contenir la dépression appliquée, en mbar : le script lit le premier nombre du nom de fichier (par exemple `50mbar.png` ou `P_120.tif`).

**Sorties.**
- `analyse_biaxiale.xlsx` : profils par pression et paramètres d'ajustement ;
- `synthese_wmax_vs_dP.png` : flèche maximale en fonction de la dépression, pour les deux axes ;
- `synthese_profiles.png` : profils de déformation superposés ;
- `surface_3D.png` : surface reconstruite.

**Paramètres à vérifier**, en tête du fichier :

| Paramètre | Valeur | Signification |
|---|---|---|
| `LAMBDA_NM` | 661 | Longueur d'onde du laser du profilomètre |
| `SCALE_UM_PX` | 2,75 | Échelle des images |
| `A_UM`, `B_UM` | 3750, 2800 | Demi-axes de la cavité elliptique |
| `E0_UM` | 12 | Gap initial entre le verre et la membrane |
| `E_PA`, `NU` | 1,3 MPa ; 0,5 | Module d'Young et coefficient de Poisson du PDMS |
| `PEAK_DIST_PX` | 2 | Espacement minimal entre deux pics détectés |

**Points critiques**, déjà discutés dans le manuscrit : l'espacement minimal de détection doit rester inférieur ou égal à 2 pixels, faute de quoi des franges sont ignorées sur les flancs les plus pentus ; le segment doit couvrir un seul flanc, du bord encastré au centre ; seule la longueur d'onde, et non l'échelle spatiale, conditionne l'amplitude reconstruite.

---

## `analyse_pieges/Analyse_Pieges.py`

**Ce qu'il fait.** Il analyse une vidéo de la matrice de pièges et détermine, image par image, l'état de chaque piège. Le traitement reprend une routine de type ImageJ :

1. soustraction facultative d'une image de référence, prise sur la matrice vide, pour corriger un éclairage non uniforme ;
2. conversion en 8 bits et remise à l'échelle des niveaux de gris ;
3. seuillage sur une plage de niveaux ;
4. détection des objets, filtrés par taille et par circularité.

Deux modes de comptage sont proposés : **binaire** (piège vide ou occupé), plus robuste, et **comptage** (nombre de particules par piège), plus informatif. Tous les réglages sont ajustés dans l'interface avec un aperçu en direct, puis peuvent être exportés dans un fichier de configuration et rechargés.

**Entrées.** Une vidéo (`.avi`, `.mp4`, `.mov`, `.mkv`).

**Sorties**, dans le dossier de la vidéo :
- `<nom>_analyse.xlsx` : état de chaque piège au cours du temps ;
- `<nom>_timeline.csv` : nombre de pièges occupés en fonction du temps ;
- `<nom>_capture_timeline.svg` : courbe de remplissage ;
- `<nom>_config.json` : réglages de l'analyse, réutilisables sur d'autres vidéos.

**Utilisé pour** les billes de polystyrène des chapitres 1 (80 µm et 6 µm) et les microalgues du chapitre 3.

---

## `analyse_pieges/Comparaison_Cycles.py`

**Ce qu'il fait.** À partir du fichier produit par `Analyse_Pieges.py`, il découpe l'enregistrement en cycles de capture et de libération, les aligne sur un même instant initial et les compare.

**Entrées.** Le fichier `<nom>_analyse.xlsx`. L'utilisateur indique le début et la fin de chaque cycle dans l'interface ; l'instant de libération est pris au maximum d'occupation de chaque cycle.

**Sorties**, dans un dossier `comparaison_cycles/` :
- chronologie complète annotée ;
- cycles superposés, alignés sur leur instant initial ;
- courbe moyenne avec écart type ;
- histogramme des particules capturées et libérées par cycle ;
- vitesses de remplissage et boîtes à moustaches ;
- classeurs Excel récapitulatifs.

**Paramètres à vérifier.** `FPS` (20 images par seconde par défaut) et `N_CYCLES`, le nombre de cycles à comparer.

---

## `automatisation/`

Interface Python de pilotage de la plateforme, fondée sur µManager, qui centralise l'imagerie, la platine motorisée et la gestion fluidique. Le manuscrit en décrit quatre niveaux d'automatisation, dont les deux premiers sont opérationnels : cycles temporisés avec photographie globale, puis analyse piège par piège.

---

## Enchaînement type

```
vidéo de la matrice
      │
      ▼
Analyse_Pieges.py  ──►  <nom>_analyse.xlsx  ──►  Comparaison_Cycles.py  ──►  figures de cycles
```

L'analyse de la membrane est indépendante :

```
images de franges (grand axe, petit axe)  ──►  Analyse_Franges_Fizeau.py  ──►  déformation et module effectif
```

---

## Dépendances

`numpy`, `scipy`, `matplotlib`, `opencv-python`, `Pillow`, `openpyxl`, et `tkinter` pour les interfaces. Voir le fichier `requirements.txt` à la racine.
