# Fusion audio + vidéo

Combinaison des probabilités de trois modèles pour reconnaître l'émotion d'un extrait de film :
**HuBERT** (voix), **AST** (sons généraux) et le **modèle vidéo**.

## Données d'entrée (probabilités dev et test, produites par les autres parties)

| Source | Fichiers | Partie |
|---|---|---|
| HuBERT | `audio_hubert_corrige.csv` | audio, voix (recalculé dans la 1re cellule du notebook) |
| AST | `ast_dev.csv`, `ast_test.csv` | audio, sons généraux |
| Vidéo | `probas_video_dev.csv`, `probas_video_test.csv` | vidéo |

Les extraits sont alignés par `video_id` et les émotions par leur nom, selon la partition officielle du groupe.

## Méthode

- Entrées : les 8 probabilités de chaque modèle, soit 24 valeurs par extrait (logarithme puis standardisation).
- Réseau de neurones : une couche cachée (16 ou 32 neurones, ReLU, dropout), perte pondérée par classe, optimiseur AdamW.
- Apprentissage sur le **dev** uniquement (442 extraits). Les réglages sont choisis par validation croisée en 5 plis par film, et on fait la moyenne de 5 graines.
- Évaluation une seule fois sur le **test** (439 extraits), avec le F1 macro.
- Références de comparaison : chaque modèle seul, la moyenne pondérée (poids choisis sur le dev) et la régression logistique.

## Résultats (F1 macro)

| Méthode | Dev | Test |
|---|---|---|
| HuBERT seul | 0,274 | 0,214 |
| AST seul | 0,236 | 0,248 |
| Vidéo seule | 0,406 | **0,351** |
| Moyenne pondérée (vidéo 0,8 / HuBERT 0,2 / AST 0) | 0,414 | **0,351** |
| Régression logistique (C = 0,03) | 0,331* | 0,320 |
| Réseau de neurones (5 graines) | 0,273* | 0,335 |

\* Score de validation croisée sur le dev.

## Conclusion

La vidéo est la source la plus informative. Aucune fusion ne fait mieux que la **vidéo** seule : l'audio est faible, et le dev (442 extraits) est trop petit pour apprendre à bien combiner les sources. Les écarts restent dans la marge d'erreur (environ ±0,05).
