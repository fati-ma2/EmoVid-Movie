# Reconnaissance des émotions à partir du son — Partie audio (sons généraux)

Projet Données Multimédia — M2 SID 2026
Auteur : Ghislain Nezerwe

## Contexte

Ce dépôt contient le code de la **partie audio** du projet, consacrée aux **sons généraux** : musique, bruitages et ambiance sonore. L'objectif est de reconnaître l'émotion d'un extrait de film à partir de son **son uniquement**.

**Corpus :** EmoVid (Qiu et al., 2025), section « films » : 3 088 extraits issus de 350 films, annotés selon 8 émotions (amusement, awe, contentment, excitement, anger, disgust, fear, sadness).

**Partition du groupe** (aucun film partagé entre deux ensembles) :

| Ensemble | Films | Extraits | Rôle |
|---|---|---|---|
| train | 250 | 2 207 | apprentissage |
| dev | 50 | 442 | choix des réglages |
| test | 50 | 439 | score final |

## Méthode

1. Chaque extrait est transformé en **vecteur de nombres** par un **modèle pré-entraîné et gelé**, que l'on ne réentraîne pas.
2. Un **classifieur commun** (standardisation + régression logistique pondérée par classe) apprend à reconnaître les 8 émotions.
3. La régularisation `C` est choisie sur le **dev** : on prend le plus petit `C` dont le score est à moins de 0,01 du meilleur, pour limiter le surapprentissage.
4. Le score final est mesuré **une seule fois sur le test**, avec le **F1 macro** (même poids pour les 8 émotions). Il est aussi calculé séparément pour les extraits **avec** et **sans parole**.

| Modèle | Référence Hugging Face | Paramètres | Ce qu'il a appris |
|---|---|---|---|
| **AST** | `MIT/ast-finetuned-audioset-10-10-0.4593` | 87 M | 527 types de sons (AudioSet) |
| **CLAP** | `laion/clap-htsat-unfused` | 153 M | sons associés à des descriptions en mots |
| **Whisper** (encodeur) | `openai/whisper-base` | 21 M | 680 000 h d'audio, surtout de la parole |

## Organisation du dépôt

```
├── README.md
├── requirements.txt
├── utils.py                      # code commun à tous les notebooks
├── notebooks/
│   ├── b0_nettoyage_audio.ipynb       # transcription, langue, extraits avec / sans parole
│   ├── b1_ast.ipynb                   # AST : vecteurs, classifieur, sons détectés par émotion
│   ├── b2_clap.ipynb                  # CLAP : zero-shot, classifieur, fusion avec AST
│   ├── b3_whisper_encoder.ipynb       # encodeur Whisper : choix de la couche, fusion avec AST
│   └── b4_surapprentissage.ipynb      # contrôle du surapprentissage des 3 modèles
├── splits/                       # liste des films de chaque ensemble
├── resultats/                    # resultats.csv et probabilités dev / test (preds/)
└── figures/                      # matrices de confusion, courbes
```

## Exécution

Les notebooks sont prévus pour **Google Colab**. Ils sont à exécuter **dans l'ordre** :

1. Copier `utils.py` et les notebooks dans un dossier Google Drive, puis adapter le chemin `CODE` dans la première cellule de chaque notebook.
2. Déposer l'archive du corpus (`PROJET_SID_2026.zip`) sur le Drive : `utils.setup()` la copie et la décompresse automatiquement dans Colab.
3. Exécuter **b0**, puis **b1 → b2 → b3** (un GPU T4 est conseillé pour l'extraction des vecteurs), puis **b4** (pas besoin de GPU).

Les vecteurs sont enregistrés au fur et à mesure : un calcul interrompu reprend là où il s'était arrêté.

Les probabilités dev et test d'AST (`resultats/preds/`) sont celles utilisées dans la partie fusion du projet.

> Les données (vidéos, audios) et les vecteurs extraits ne sont pas inclus dans ce dépôt : ils sont trop volumineux, et le corpus EmoVid est réservé à un usage de recherche non commercial.

## Résultats

### Les trois modèles audio (F1 macro, hasard ≈ 0,13 au test)

| Modèle | Dev | Test | Validation croisée (train + dev, 5 découpages par films) |
|---|---|---|---|
| **AST** | 0,236 | **0,248** | 0,237 ± 0,023 |
| CLAP | 0,257 | 0,221 | 0,234 ± 0,034 |
| Whisper | 0,310 | 0,254 | 0,247 ± 0,035 |
| Fusion tardive AST + Whisper | 0,319 | 0,267 | 0,250 ± 0,028 |

- Les trois modèles font environ **deux fois mieux que le hasard**, mais leurs écarts sont **plus petits que la marge d'erreur**.
- **AST est retenu** : c'est le seul modèle stable entre le dev et le test, avec l'écart-type le plus faible en validation croisée et peu de sensibilité au réglage de `C`.
- **Pas de surapprentissage marqué** : les scores sur le train restent bas (0,30 à 0,48) grâce à la régularisation.
- **Pas de fuite d'information** : avec des étiquettes mélangées au hasard, les scores tombent au niveau du hasard (0,09 à 0,11).

### Ce que le son permet de reconnaître

- Les émotions les mieux reconnues sont **la peur** (F1 0,46 pour AST au test) et **l'amusement**, qui ont une musique typique : « Scary » ×2,8 pour la peur, « Funny » ×3,8 pour l'amusement.
- **Awe** est la moins bien reconnue : elle n'a pas de musique propre et est souvent prise pour de la peur.
- Dans les 986 extraits **sans parole**, le son dominant détecté par AST est la **musique** (68,5 %).

## Conclusion

Le son contient une information réelle sur l'émotion, surtout à travers la **musique**, mais limitée (F1 macro autour de 0,25). **AST** est le modèle de sons le plus fiable. Les courbes d'apprentissage suggèrent qu'avec davantage de données, Whisper et CLAP pourraient encore progresser, alors qu'AST semble avoir atteint un plateau.

## Références

- Qiu, Z., Wang, B., Chen, X., He, Y., & Wang, Z. (2025). *EmoVid: A Multimodal Emotion Video Dataset for Emotion-Centric Video Understanding and Generation*. arXiv:2511.11002.
- Gong, Y., Chung, Y.-A., & Glass, J. (2021). *AST: Audio Spectrogram Transformer*. Interspeech.
- Gemmeke, J. F., et al. (2017). *Audio Set: An ontology and human-labeled dataset for audio events*. ICASSP.
