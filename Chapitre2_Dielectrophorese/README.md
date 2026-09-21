# Chapitre 2 : caractérisation diélectrophorétique

Ce dossier rassemble les outils du deuxième chapitre, consacré à la caractérisation de microalgues par diélectrophorèse (DEP).

Le dispositif repose sur une électrode en plot : un anneau d'or de 50 à 140 µm de rayon, déposé sur verre, face à une contre-électrode d'ITO, dans une cavité de 20 µm de hauteur. Un générateur de signaux applique un signal alternatif entre 1 et 50 MHz, et les cellules sont filmées pendant qu'elles se redistribuent sous l'effet du champ.

Deux échelles d'analyse sont proposées :

- **la population**, par l'analyse d'images : comptage des cellules et mesure d'intensité lumineuse dans trois régions de l'électrode, résumées en un indice diélectrophorétique normalisé $I_{DEP}$ ;
- **la cellule unique**, par le suivi individuel des trajectoires, confronté à un modèle de champ calibré sur des billes de polystyrène, afin d'estimer la partie réelle du facteur de Clausius-Mossotti de chaque cellule.

---

## Contenu

| Dossier | Fichier | Rôle |
|---|---|---|
| `1_acquisition/` | `Controle_Generateur_TTi_Linux.py` | Pilotage du générateur de signaux |
| `2_analyse_population/` | `DEP_Analysis_Pipeline.py` | Analyse d'une vidéo : comptage, contrastes, signal de redistribution |
| `2_analyse_population/` | `DEP_Bilan_Analysis.py` | Indice $I_{DEP}$ sur un ensemble de vidéos, par fréquence et par réplicat |
| `3_suivi_individuel/` | `Suivi_Particules.py` | Suivi semi-manuel de particules autour de l'électrode |
| `4_simulation/` | `1_Reconstruction_Champ_Billes.py` | Champ calculé, champ reconstruit sur billes, et leur comparaison |
| `4_simulation/` | `2_Confrontation_Laplace_Suivi.py` | Confrontation du champ calculé aux trajectoires des algues |
| `4_simulation/` | `3_Determination_ReK_Algues.py` | Ajustement de Re[K] cellule par cellule dans le champ calibré |
| `4_simulation/` | `Simulateur_Trajectoires.py` | Outil interactif de simulation et abaques |

---

## Vue d'ensemble

```
                         Controle_Generateur_TTi_Linux.py
                                        │
                                        ▼
                                vidéos de DEP
                   ┌────────────────────┴────────────────────┐
                   ▼                                          ▼
      ANALYSE DE POPULATION                          ANALYSE INDIVIDUELLE
                   │                                          │
      DEP_Analysis_Pipeline.py                      Suivi_Particules.py
      (une vidéo à la fois)                     (billes puis algues)
                   │                                          │
      ##CONDITION-Sx-PRE-fMHz##.txt                 tracking_results.xlsx
                   │                                          │
      DEP_Bilan_Analysis.py              1_Reconstruction_Champ_Billes.py
                   │                     ├─ champ_simule.csv (Laplace)
          spectres de I_DEP              └─ champ_reconstruit.csv (billes)
                                                 │                │
                                  2_Confrontation_       3_Determination_
                                  Laplace_Suivi.py       ReK_Algues.py
                                                 │                │
                                   échec du champ idéal   Re[K] par cellule
```

---

## `1_acquisition/Controle_Generateur_TTi_Linux.py`

Interface de pilotage du générateur **TTi TGF4162** (série TGF4000) par son port USB virtuel. Elle permet de choisir la voie, la forme d'onde, la fréquence et l'amplitude, d'activer ou de couper la sortie, et propose des boutons de fréquences rapides à 1, 10, 20, 30, 40 et 50 MHz, qui correspondent au balayage employé dans la thèse.

Version Linux : le port est typiquement `/dev/ttyUSB0` ou `/dev/ttyACM0`, et l'utilisateur doit appartenir au groupe `dialout`.

**Attention à la convention d'amplitude** : le générateur reçoit une amplitude crête à crête. Les 10 V crête à crête employés dans la thèse correspondent à une amplitude de 5 V sur l'électrode, valeur à utiliser dans les simulations.

---

## `2_analyse_population/`

### `DEP_Analysis_Pipeline.py`

**Ce qu'il fait.** Pour une vidéo donnée, il compare l'état de référence et l'état sous champ, chacun moyenné sur quelques images :

1. l'utilisateur désigne le centre de l'électrode ;
2. les cellules sont comptées par transformée de Hough, et leur répartition radiale est tracée ;
3. l'intensité moyenne est mesurée dans trois régions d'intérêt, un carré central et deux couronnes centrées sur les bords de l'anneau, après correction de la dérive d'éclairage mesurée dans une région de fond ;
4. le signal de redistribution, différence entre le contraste du centre et celui des couronnes, est calculé avec son rapport signal sur bruit.

**Entrées.** Une vidéo, un dossier de résultats, et quelques métadonnées saisies dans une fenêtre : condition (`TEMOIN` ou `TEST`), réplicat, exposition (`PRE` ou `POST`), fréquence.

**Sorties.** Un fichier texte `##<CONDITION>-S<réplicat>-<PRE|POST>-<f>MHz##.txt` par vidéo, contenant les contrastes, le signal de redistribution et le contraste de référence, ainsi que les figures de comptage et de contraste.

**Paramètres à vérifier.** `SCALE`, l'échelle en pixels par micromètre (3,06 dans la thèse), et les rayons intérieur et extérieur de l'anneau (50 et 140 µm).

### `DEP_Bilan_Analysis.py`

**Ce qu'il fait.** Il lit l'ensemble des fichiers texte d'un dossier, regroupe les mesures par condition, réplicat et fréquence, et calcule l'indice diélectrophorétique :

$$I_{DEP}(f) = \tanh\left[\alpha \, \frac{S(f)}{S_{ref}}\right], \qquad \alpha = \mathrm{arctanh}(0{,}9)$$

où $S_{ref}$ est la plus grande amplitude du signal observée sur les six fréquences d'un même réplicat. L'indice est borné entre $-1$ et $1$, positif en DEP positive et négatif en DEP négative.

**Entrées.** Le dossier contenant les fichiers `##...##.txt`.

**Sorties**, dans un sous-dossier `BILAN/` : spectres de l'indice par réplicat et moyennés, comparaison entre condition test et condition témoin, et un rapport texte récapitulatif.

---

## `3_suivi_individuel/Suivi_Particules.py`

**Ce qu'il fait.** Il suit des particules individuelles, billes ou algues, autour de l'électrode, par corrélation croisée normalisée entre une imagette de référence et une fenêtre de recherche. Les positions sont exprimées en coordonnées polaires centrées sur l'électrode.

Deux modes sont proposés :

- **classique** : l'utilisateur désigne plusieurs particules sur la première image, suivies ensemble ;
- **caractérisation** : une particule à la fois, que l'on choisit de sauvegarder ou non, ce qui produit un onglet par particule.

**Commandes** : clic gauche pour placer le centre puis les particules, `P` et `M` pour ajuster le cercle guide, Entrée ou Espace pour valider, Retour arrière pour annuler le dernier clic, `Q` pour quitter.

**Sorties.** `tracking_results.xlsx`, ainsi que les figures `trajectories_rt.png` (rayon et angle en fonction du temps) et `trajectories_polar.png` (carte polaire).

**Paramètres à vérifier.** `SCALE_PX_PER_UM`, l'échelle des images, et `NCC_THRESHOLD`, le score minimal de corrélation en deçà duquel le suivi est interrompu.

---

## `4_simulation/`

Les trois scripts numérotés forment une chaîne. Ils s'exécutent dans cet ordre, car les deux derniers lisent le champ produit par le premier.

### `1_Reconstruction_Champ_Billes.py`

Le programme comporte trois sections indépendantes :

- **A, chemin ascendant, des billes vers le champ.** À partir des trajectoires de billes de polystyrène de propriétés connues : vitesse mesurée, puis force, gradient du carré du champ, carré du champ par intégration, et enfin potentiel effectif. Aucune résolution de Laplace n'intervient : le champ obtenu est celui que subissent réellement les billes, écoulements induits compris.
- **B, chemin descendant, du potentiel vers les vitesses.** Résolution de l'équation de Laplace en géométrie axisymétrique pour la tension et la géométrie déclarées, puis prédiction des vitesses. Aucun paramètre n'est ajusté.
- **C, confrontation des deux.** Leur écart mesure ce que le modèle idéal ne décrit pas, et sa dépendance radiale en indique la nature.

**Entrées.** Le classeur de suivi des billes (`tracking_results.xlsx`) et un dossier de sortie.

**Sorties.** `champ_reconstruit.csv` (champ calibré, utilisé par `3_Determination_ReK_Algues.py`), `champ_simule.csv` (champ de Laplace, utilisé par `2_Confrontation_Laplace_Suivi.py`), `comparaison.csv`, `parametres.json`, et les figures de la chaîne de reconstruction, des cartes du potentiel et du champ, et de la comparaison des longueurs de décroissance et des vitesses.

### `2_Confrontation_Laplace_Suivi.py`

Il répond à une seule question : le champ calculé par Laplace rend-il compte du mouvement observé ? Il n'ajuste rien. Le facteur de Clausius-Mossotti est imposé, et la prédiction est encadrée par une bande de sûreté entre deux valeurs extrêmes plausibles (−0,5 et −0,1 par défaut).

**Entrées.** `champ_simule.csv` et le classeur de suivi des algues.

**Sorties.** Figures de vitesse en fonction de la position et de position en fonction du temps, `trajectoires.csv`, `confrontation.csv` et `bilan.json`.

### `3_Determination_ReK_Algues.py`

Le champ ayant été calibré une fois pour toutes sur les billes, il ne reste qu'une inconnue dans le bilan des forces : la partie réelle du facteur de Clausius-Mossotti. Pour chaque cellule suivie, cette valeur est ajustée pour que le déplacement simulé reproduise le déplacement mesuré. L'ajustement porte sur la position plutôt que sur la vitesse, pour ne pas amplifier le bruit de pointage, et la courbe de l'écart en fonction de la valeur d'essai est tracée pour juger de la qualité de chaque ajustement.

Les bornes du balayage sont volontairement larges (de −10 à 10), bien au-delà du domaine physique d'une sphère (de −0,5 à 1). Une valeur qui sortirait de ce domaine signale une incomplétude du bilan des forces plutôt qu'un résultat à masquer.

**Entrées.** Le champ calibré (`champ_reconstruit.csv`, ou à défaut son amplitude et sa longueur de décroissance saisies dans l'interface) et le classeur de suivi des algues.

**Sorties.** `resultats_ReK.csv` (une valeur par cellule), `trajectoires.csv`, `resultats.json`, et les figures de l'interface et de la distribution sur la population.

### `Simulateur_Trajectoires.py`

Outil interactif, indépendant de la chaîne ci-dessus. Il intègre le mouvement radial de particules dans le champ idéal de Laplace ou dans le champ mesuré, pour des paramètres choisis librement, et produit des abaques : déplacement atteint, temps d'atteinte d'une position, comparaison des deux champs. Il permet d'anticiper le comportement d'une particule avant une expérience, ou de dimensionner un nouveau dispositif.

---

## Paramètres physiques de la thèse

| Grandeur | Valeur |
|---|---|
| Rayons de l'anneau | 50 et 140 µm |
| Hauteur de cavité | 20 µm |
| Épaisseur d'or | 1 µm |
| Tension | 10 V crête à crête, soit 5 V d'amplitude, signal créneau |
| Fréquence de calibration sur billes | 100 kHz |
| Conductivité du TAP frais (billes) | 0,3 S/m |
| Conductivité du TAP au plateau de croissance (algues) | 0,12 S/m |
| Permittivité relative du milieu | 78 |
| Billes de référence | polystyrène, 6 µm de diamètre |

---

## Limites connues

Les limites de la méthode sont discutées en détail dans le manuscrit. En résumé : la calibration du champ est systématique et a été établie dans un milieu plus conducteur que celui des algues ; la hauteur des cellules dans la cavité n'est pas mesurée ; le champ reconstruit n'est valide qu'à distance du bord de l'électrode ; enfin le suivi reste semi-manuel. Les valeurs de Re[K] obtenues sur les algues sont très inférieures aux prédictions du modèle à coquille, ce qui indique que le bilan à deux forces (diélectrophorèse et traînée) est incomplet : une retenue au substrat, et peut-être des effets collectifs, restent à intégrer.

---

## Dépendances

`numpy`, `scipy`, `matplotlib`, `pandas`, `openpyxl`, `opencv-python`, `shapely`, `pyserial`, et `tkinter` pour les interfaces. Voir le fichier `requirements.txt` à la racine.
