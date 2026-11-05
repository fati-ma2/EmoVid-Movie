# Fusion multimodale Texte + Audio + Vidéo – EmoVid

## 1. Présentation du projet

Cette partie du projet EmoVid est consacrée à la reconnaissance multimodale des émotions à partir de trois sources d'information : le texte, l'audio et la vidéo.

L'objectif est de combiner les informations issues de ces trois modalités afin d'améliorer la classification automatique des huit catégories émotionnelles du dataset EmoVid.

L'approche retenue repose sur une **fusion précoce (Early Fusion)**, consistant à concaténer les représentations numériques extraites de chaque modalité avant de les transmettre à un classifieur de régression logistique.

## 2. Objectifs

- Exploiter conjointement les informations textuelles, acoustiques et visuelles.
- Combiner les représentations numériques des trois modalités.
- Construire une représentation multimodale unique.
- Optimiser les hyperparamètres de la régression logistique.
- Évaluer les performances de la fusion sur les ensembles de validation et de test.
- Comparer les résultats obtenus avec ceux des modèles unimodaux.

## 3. Description des modalités

Trois types de représentations sont utilisés :

| Modalité | Représentation | Dimensions |
|---|---|---:|
| Texte | Embeddings textuels | 384 |
| Audio | Embeddings HuBERT | 768 |
| Vidéo | Embeddings visuels | 512 |
| **Total** | **Représentation fusionnée** | **1 664** |

Les représentations sont concaténées pour obtenir un vecteur de 1 664 caractéristiques par extrait.

## 4. Méthode de fusion

### Fusion précoce (Early Fusion)

La fusion précoce consiste à combiner directement les embeddings des trois modalités avant la classification.

La représentation multimodale est obtenue par concaténation :

\[
X_{\text{fusion}} =
[X_{\text{texte}} \; || \; X_{\text{audio}} \; || \; X_{\text{vidéo}}]
\]

Avec :

- Texte : 384 dimensions.
- Audio : 768 dimensions.
- Vidéo : 512 dimensions.

La représentation finale comporte donc :

\[
384 + 768 + 512 = 1664
\]

caractéristiques.

## 5. Classification et optimisation

Un modèle de **régression logistique** est utilisé pour prédire les huit catégories émotionnelles à partir des représentations fusionnées.

La démarche expérimentale comprend :

1. Concaténation des embeddings des trois modalités.
2. Standardisation des caractéristiques.
3. Entraînement du classifieur sur l'ensemble Train.
4. Recherche des meilleurs hyperparamètres sur Dev.
5. Évaluation finale sur Test.

### Sélection des hyperparamètres

Plusieurs valeurs du paramètre de régularisation \(C\) ont été testées :

\[
C \in \{0.001,\;0.01,\;0.1,\;1,\;10,\;100\}
\]

Deux stratégies de pondération des classes ont également été comparées :

- `class_weight=None`
- `class_weight="balanced"`

Le macro-F1 sur Dev est utilisé comme critère de sélection.

La meilleure configuration obtenue est :

| Hyperparamètre | Valeur retenue |
|---|---|
| Classifieur | Régression logistique |
| C | 0,01 |
| class_weight | None |
| Critère de sélection | Macro-F1 sur Dev |

## 6. Résultats expérimentaux

### Performances sur Dev et Test

| Indicateur | Résultat |
|---|---:|
| Macro-F1 Dev | **0,3905** |
| Accuracy Test | **0,3690** |
| Macro-F1 Test | **0,3327** |

La configuration retenue obtient un macro-F1 de 0,3905 sur l'ensemble de validation et de 0,3327 sur l'ensemble de test.

### Interprétation

Ces résultats montrent que la combinaison des informations textuelles, acoustiques et visuelles permet d'exploiter des informations complémentaires pour la reconnaissance des émotions.

Sur l'ensemble de test, la fusion multimodale obtient un macro-F1 de 0,3327, contre 0,262 pour le meilleur modèle audio individuel étudié (wav2vec2-SUPERB-ER).

Cette comparaison suggère un apport intéressant de la combinaison des trois modalités.

## 7. Métriques d'évaluation

Les performances sont mesurées à l'aide de deux indicateurs principaux :

- **Accuracy** : proportion de prédictions correctement classifiées.
- **Macro-F1** : moyenne des scores F1 calculés pour chacune des huit émotions, avec une importance identique accordée à chaque classe.

## 8. Technologies utilisées

- Python
- NumPy
- Pandas
- Scikit-learn
- PyTorch
- Hugging Face Transformers
- Jupyter Notebook
- Matplotlib
- Seaborn

## 9. Conclusion

Cette expérimentation a permis de mettre en œuvre une fusion précoce des modalités Texte, Audio et Vidéo pour la reconnaissance automatique des émotions.

La concaténation des représentations produit un vecteur de 1 664 dimensions, utilisé par un classifieur de régression logistique.

Après optimisation des hyperparamètres sur Dev, la configuration retenue atteint un macro-F1 de 0,3327 sur Test.

Les résultats obtenus montrent l'intérêt de combiner plusieurs modalités pour exploiter différentes sources d'information dans une tâche de reconnaissance multimodale des émotions.
