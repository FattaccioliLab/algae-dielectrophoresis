# Piéger, libérer, caractériser : outils microfluidiques et diélectrophorétiques pour l'étude des microalgues

**Codes, fichiers de conception et outils d'analyse de la thèse de doctorat d'Alexandre Chargueraud**
ENS-PSL, Institut Pierre-Gilles de Gennes (IPGG), sous la direction de Jacques Fattaccioli
Projet ANR Phosphalgue · Soutenance le 12 novembre 2026

\---

## Pourquoi ce dépôt

Une thèse expérimentale ne tient pas seulement dans son manuscrit. Derrière chaque figure se trouvent des vidéos, des routines d'analyse, des montages imprimés en trois dimensions et des interfaces de pilotage, développés au fil de trois années de travail. Ce dépôt rassemble ces outils, organisés dans l'ordre des chapitres du manuscrit, afin que chacun puisse :

* **reproduire** les analyses et les figures présentées dans la thèse ;
* **réutiliser** les outils sur ses propres données, qu'il s'agisse de pièges microfluidiques, de vidéos de diélectrophorèse ou de suivi de particules ;
* **prolonger** le travail là où il s'est arrêté, les limites et les pistes ouvertes étant signalées dans chaque dossier.

Le fil conducteur de la thèse est simple à énoncer : piéger en grand nombre des microalgues *Chlamydomonas reinhardtii*, les libérer à la demande, et caractériser leurs propriétés électriques par diélectrophorèse. Les trois chapitres suivent cette logique, du dispositif à l'application biologique.

\---

## Organisation

```
.
├── Chapitre1\_Pieges\_Microfluidiques/     Dispositif de piégeage et de libération
│   ├── conception/                        Masques et moules de fabrication
│   ├── analyse\_membrane/                  Déformation de la membrane par franges de Fizeau
│   ├── analyse\_pieges/                    Remplissage des pièges et comparaison des cycles
│   └── automatisation/                    Pilotage de la plateforme
│
├── Chapitre2\_Dielectrophorese/           Caractérisation diélectrophorétique
│   ├── 1\_acquisition/                     Pilotage du générateur de signaux
│   ├── 2\_analyse\_population/              Comptage, intensité et indice DEP
│   ├── 3\_suivi\_individuel/                Suivi de particules autour de l'électrode
│   └── 4\_simulation/                      Champ, calibration et facteur de Clausius-Mossotti
│
├── Chapitre3\_Application\_Microalgues/    Réponse des algues à la lumière
│   ├── banc\_lumiere/                      Banc d'exposition, firmware et pièces imprimées
│   └── modelisation/                      Modèle à coquille et bilan de carbone
│
├── requirements.txt                       Dépendances Python



```

Chaque chapitre possède son propre `README.md`, qui décrit le rôle de chaque script, les figures du manuscrit qu'il produit, ses entrées et ses sorties, et l'ordre dans lequel les lancer.

|Chapitre|Contenu principal|Point d'entrée|
|-|-|-|
|1. Dispositif microfluidique|Membrane en PDMS portant une matrice de pièges, libérée par dépression pneumatique|[README du chapitre 1](Chapitre1_Pieges_Microfluidiques/README.md)|
|2. Diélectrophorèse|Électrode en plot, indice diélectrophorétique de population, simulation et mesure à la cellule unique|[README du chapitre 2](Chapitre2_Dielectrophorese/README.md)|
|3. Application aux microalgues|Influence de la lumière sur la réponse diélectrophorétique|[README du chapitre 3](Chapitre3_Application_Microalgues/README.md)|

Certains outils servent dans plusieurs chapitres : l'analyse des pièges développée au chapitre 1 est réutilisée pour les algues au chapitre 3, et toute la chaîne d'analyse diélectrophorétique du chapitre 2 sert de base au chapitre 3. Les README l'indiquent à chaque fois, plutôt que de dupliquer les fichiers.

\---

## Installation

Les scripts sont écrits en Python 3 (version 3.9 ou ultérieure) et ont été développés sous l'environnement [Pyzo](https://pyzo.org). Ils fonctionnent aussi en ligne de commande.

```bash
git clone https://github.com/\[COMPTE]/these-microalgues-microfluidique-DEP.git
cd these-microalgues-microfluidique-DEP
pip install -r requirements.txt
```

Quelques précisions :

* Les interfaces graphiques reposent sur `tkinter`, fourni avec Python sous Windows et macOS. Sous Linux, il s'installe séparément : `sudo apt install python3-tk`.
* Les scripts de pilotage d'instruments (générateur de signaux, banc de lumière) communiquent par port série. Sous Linux, l'utilisateur doit appartenir au groupe `dialout` : `sudo usermod -aG dialout $USER`, puis se déconnecter et se reconnecter.
* Chaque script s'exécute seul. Sous Pyzo, ouvrir le fichier et lancer l'exécution (touche F5) ; en ligne de commande, `python nom\_du\_script.py`. Les fichiers d'entrée sont demandés par des fenêtres de sélection.
* Les paramètres physiques et expérimentaux (géométrie, échelle en pixels par micromètre, conductivité du milieu, tension) sont regroupés en tête de chaque script. **Ils sont à adapter à votre montage avant toute utilisation sur vos propres données.**

\---

## Données

Les vidéos brutes ne sont pas incluses, leur volume dépassant ce que GitHub permet d'héberger.

\---

## Contact

Alexandre Chargueraud · alexandrechargueraud@gmail.com · https://www.linkedin.com/in/alexandre-chargueraud-1018721a5/

Les questions, signalements d'erreur et propositions d'amélioration sont bienvenus via l'onglet *Issues* du dépôt.

\---

## English summary

This repository gathers the code, design files and analysis tools developed during the PhD thesis of Alexandre Chargueraud (ENS-PSL, IPGG, 2026). The thesis develops a microfluidic platform able to trap and release large numbers of cells through the pneumatic deformation of a PDMS membrane (Chapter 1), a dielectrophoresis characterisation workflow combining population-level image analysis, single-cell tracking and field simulation (Chapter 2), and applies both to the microalga *Chlamydomonas reinhardtii*, revealing a light-dependent dielectrophoretic response (Chapter 3). Each chapter folder contains a README describing the scripts, their inputs and outputs, and the thesis figures they produce. Comments and documentation are written in French.

