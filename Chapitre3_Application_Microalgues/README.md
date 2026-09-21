# Chapitre 3 : application aux microalgues

Ce dossier rassemble les outils du troisième chapitre, qui applique les développements des deux premiers à la microalgue *Chlamydomonas reinhardtii*.

Le chapitre comporte deux parties. La première porte le piégeage et la libération sur des algues vivantes, et prépare l'étude de l'accumulation de polyphosphates. La seconde met en évidence un résultat inattendu : la réponse diélectrophorétique des algues bascule de négative à positive sous éclairement, de façon réversible, avec une cinétique de quelques minutes. Pour l'étudier, nous avons construit un banc d'exposition lumineuse compatible avec la cavité de diélectrophorèse.

---

## Contenu

| Dossier | Fichier | Rôle |
|---|---|---|
| `banc_lumiere/` | `Controle_Banc_Lumiere.py` | Pilotage de la LED, mesure d'éclairement et de spectre |
| `banc_lumiere/firmware_arduino/` | *à compléter* | Programme de la carte Arduino |
| `banc_lumiere/impression_3D/` | `SupportDEP.stl`, `SupportLight.stl` | Pièces imprimées du banc |
| `modelisation/` | *à compléter* | Modèle à coquille et bilan de carbone dans la cavité |

Ce chapitre réutilise aussi les outils des chapitres précédents, sans les dupliquer :

| Besoin | Outil | Emplacement |
|---|---|---|
| Piégeage et libération des algues | `Analyse_Pieges.py`, `Comparaison_Cycles.py` | `Chapitre1_Pieges_Microfluidiques/analyse_pieges/` |
| Pilotage du générateur | `Controle_Generateur_TTi_Linux.py` | `Chapitre2_Dielectrophorese/1_acquisition/` |
| Réponse DEP avant et après exposition | `DEP_Analysis_Pipeline.py`, `DEP_Bilan_Analysis.py` | `Chapitre2_Dielectrophorese/2_analyse_population/` |

Pour ce dernier usage, les métadonnées `PRE` et `POST` saisies dans `DEP_Analysis_Pipeline.py` correspondent aux mesures avant et après exposition, et les conditions `TEST` et `TEMOIN` à la série éclairée et à la série maintenue à l'obscurité. `DEP_Bilan_Analysis.py` produit alors directement la comparaison entre les deux séries.

---

## `banc_lumiere/`

### Matériel

| Élément | Référence |
|---|---|
| Source lumineuse | LED blanche Meodex Rebel |
| Mesure d'éclairement | capteur VEML7700, interface Qwiic |
| Mesure de spectre | capteur spectral à 18 canaux, de 410 à 940 nm [RÉFÉRENCE À PRÉCISER] |
| Microcontrôleur | Arduino Uno |
| Alimentation | Eventek KPS305D |
| Filtre vert | BP 545/25 |
| Supports | impression 3D, filament 3D-Volumic-SC2 |

### `Controle_Banc_Lumiere.py`

**Ce qu'il fait.** Interface de pilotage du banc par port série :

- allumage et extinction de la LED ;
- mesure ponctuelle de l'éclairement, en lux ;
- réglage du temps d'intégration du capteur spectral, par pas de 2,78 ms ;
- acquisition du spectre sur 18 canaux, affiché en histogramme coloré, avec une estimation de la densité de flux de photons photosynthétiques (PPFD).

**Protocole série** (115 200 bauds), que le firmware Arduino doit respecter :

| Commande envoyée | Action attendue | Réponse attendue |
|---|---|---|
| `1` | allumer la LED | aucune |
| `0` | éteindre la LED | aucune |
| `3` | mesurer l'éclairement | `LUX:<valeur>` |
| `4` | mesurer le spectre | `SPECTRE:410:<v>\|435:<v>\|...\|940:<v>` |
| `I<n>` | temps d'intégration de `n` cycles | aucune |

**Estimation de la PPFD.** Elle est obtenue en sommant les canaux compris dans le domaine photosynthétique (410 à 680 nm) et en multipliant par un facteur de conversion `CONV_FACTOR = 0.0185`. Ce facteur est empirique et propre à notre capteur et à notre source : il doit être recalibré, par exemple contre un quantamètre, avant toute mesure absolue sur un autre montage. [PRÉCISER L'ORIGINE DE LA VALEUR 0,0185.]

### `firmware_arduino/`

Programme `.ino` de la carte Arduino, qui lit les deux capteurs et commande la LED selon le protocole ci-dessus. Sans lui, l'interface Python ne peut pas fonctionner.

### `impression_3D/`

- `SupportDEP.stl` : [DÉCRIRE : maintien de la cavité de diélectrophorèse sous le microscope]
- `SupportLight.stl` : [DÉCRIRE : maintien de la LED et du porte-filtre au-dessus de la cavité]

Paramètres d'impression : [IMPRIMANTE, BUSE, HAUTEUR DE COUCHE, REMPLISSAGE].

---

## `modelisation/`

Scripts de modélisation cités dans le manuscrit :

- `CM_Parameter_Search.py` : partie réelle du facteur de Clausius-Mossotti calculée avec un modèle à coquille unique, en balayant la conductivité cytoplasmique, et recherche des paramètres compatibles avec le basculement observé ;
- `CO2_Limitation_Cavite.py` : bilan du carbone inorganique et de l'oxygène dans la cavité éclairée, et carte de l'appauvrissement en carbone du bord vers le centre ;
- les ajustements des cinétiques d'installation et de relaxation de l'inversion.

---

## Protocole d'exposition de référence

| Paramètre | Valeur |
|---|---|
| Adaptation préalable à l'obscurité | 3 h |
| Éclairement | 12 000 lux, lumière blanche |
| Durée d'exposition | 10 min |
| Fréquence de suivi de la cinétique | 20 MHz |
| Conductivité du milieu (TAP au plateau de croissance) | 0,12 S/m |

---

## Dépendances

`pyserial` et `tkinter` pour le banc ; les outils réutilisés des chapitres 1 et 2 ont leurs propres dépendances, listées dans le fichier `requirements.txt` à la racine.
