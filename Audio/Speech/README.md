# Audio avec parole – Reconnaissance des émotions (EmoVid)

## 1. Présentation

Cette partie du projet EmoVid est consacrée à la reconnaissance automatique des émotions à partir des **signaux audio contenant de la parole**.

L'objectif est d'exploiter les informations acoustiques et vocales présentes dans les extraits audio afin de prédire l'émotion exprimée parmi les huit catégories du dataset.

Les expérimentations reposent sur des modèles pré-entraînés de représentation de la parole et sur un classifieur de régression logistique.

## 2. Objectifs

- Prétraiter les fichiers audio contenant de la parole.
- Extraire des représentations acoustiques à l'aide de modèles pré-entraînés.
- Comparer différentes architectures et leurs couches internes.
- Étudier différentes combinaisons de couches pour améliorer les représentations.
- Optimiser les hyperparamètres du classifieur.
- Évaluer les performances de classification des huit émotions.

## 3. Prétraitement des données

Les fichiers audio ont été préparés afin d'obtenir un format homogène pour les modèles de représentation.

Les principales opérations sont :

- Conversion des fichiers MP3 en WAV.
- Conversion en mono.
- Rééchantillonnage à 16 kHz.
- Vérification de la disponibilité des fichiers audio et de leur correspondance avec les annotations.

Les données sont réparties en trois ensembles :

| Partition | Nombre d'extraits |
|---|---:|
| Train | 2 207 |
| Dev | 442 |
| Test | 439 |
| **Total** | **3 088** |

## 4. Extraction des caractéristiques audio

Trois modèles pré-entraînés ont été étudiés :

### HuBERT Base
Modèle auto-supervisé permettant d'extraire des représentations contextualisées de la parole.

Modèle utilisé : `facebook/hubert-base-ls960`

### wav2vec 2.0 Base
Modèle pré-entraîné permettant d'obtenir des représentations acoustiques à partir du signal audio.

Modèle utilisé : `facebook/wav2vec2-base`

### wav2vec2-SUPERB-ER
Modèle spécialisé dans la reconnaissance des émotions à partir de la parole.

Modèle utilisé : `superb/wav2vec2-base-superb-er`

Les représentations extraites comportent 13 couches cachées de 768 dimensions chacune.

Différentes stratégies ont été étudiées :
- Utilisation individuelle de chaque couche.
- Combinaison de plusieurs couches.
- Moyenne des représentations issues des 13 couches.

## 5. Classification des émotions

Les représentations audio extraites sont utilisées comme entrées d'un classifieur de **régression logistique**.

La méthodologie comprend :

1. Standardisation des caractéristiques, avec ajustement du scaler sur Train uniquement.
2. Entraînement du classifieur sur Train.
3. Sélection des hyperparamètres sur Dev.
4. Évaluation finale sur Test.

Le paramètre de régularisation \(C\) est optimisé à partir du macro-F1.

## 6. Résultats expérimentaux

### Comparaison des modèles

| Modèle | Macro-F1 Dev | Macro-F1 Test |
|---|---:|---:|
| HuBERT Base | 0,274 | 0,214 |
| wav2vec 2.0 Base | 0,235 | 0,196 |
| wav2vec2-SUPERB-ER | 0,251 | **0,262** |

Le modèle wav2vec2-SUPERB-ER obtient le meilleur résultat sur l'ensemble de test, avec un macro-F1 de 0,262.

HuBERT Base obtient néanmoins le meilleur résultat sur l'ensemble de validation, avec un macro-F1 de 0,274.

### Analyse des performances

Les performances sont étudiées à l'aide des indicateurs suivants :

- **Accuracy** : proportion de prédictions correctes.
- **Macro-F1** : moyenne des scores F1 des huit catégories émotionnelles.
- **Matrice de confusion** : analyse des émotions correctement reconnues et des principales confusions.

## 7. Technologies utilisées

- Python
- PyTorch
- Hugging Face Transformers
- Librosa
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## 8. Organisation de cette partie

```text
audio_avec_parole/
│
├── notebooks/
│   └── audioETUD1.ipynb
│
├── features/
│   ├── hubert_base.npy
│   ├── wav2vec2_base.npy
│   └── wav2vec2_emotion.npy
│
├── preds/
│   ├── audio_hubert.csv
│   ├── audio_wav2vec2.csv
│   └── audio_superber.csv
│
├── results/
│   ├── figures/
│   └── matrices_confusion/
│
├── requirements.txt
└── README.md
```

*Cette arborescence est indicative : les chemins doivent être adaptés aux fichiers effectivement présents dans le dépôt.*

## 9. Conclusion

Cette étude a permis de comparer trois modèles pré-entraînés pour la reconnaissance des émotions à partir de la parole.

Les résultats montrent que wav2vec2-SUPERB-ER obtient les meilleures performances sur l'ensemble de test parmi les modèles audio étudiés, avec un macro-F1 de 0,262.

Ce modèle constitue ainsi la meilleure référence audio individuelle sur l'ensemble de test pour les expérimentations suivantes du projet EmoVid.
