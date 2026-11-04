# Fusion: Text + Audio

This folder holds the **late fusion of the text and audio models**: each model gives 8 emotion probabilities per clip, and the fusion combines them into one prediction.

**Classes (fixed order):** amusement, awe, contentment, excitement, anger, disgust, fear, sadness

## 1. Inputs

Both models were trained on the professor's **train** films only (250 films, 2,207 clips) and predict the dev (50 films, 442 clips) and test (50 films, 439 clips) sets.

| Modality | Model | Files |
|---|---|---|
| Text | `both_char_C1.0`: character n-gram TF-IDF of the caption and transcription, logistic regression (see `../../Text/`) | `text_both_char_C1p0_{dev,test}.csv` |
| Audio | `ast_hand`: pretrained AST (AudioSet) embeddings plus handcrafted MFCC and spectral features, logistic regression | `audio_ast_hand_{dev,test}.csv` |
| Labels | The professor's split | `splits/manifest.csv` (`clip_id, film_id, emotion, split`) |

Every prediction file has the columns `clip_id, p_amusement ... p_sadness, available`, with probabilities summing to 1. A clip with `available = 0` is left out of the average for that modality. On this split no input was missing.

## 2. Fusion rules

| Rule | How it combines the two models | Parameters |
|---|---|---|
| equal | Average of the two probability vectors | none |
| product | Geometric mean of the probabilities (renormalised) | none |
| weighted | Weighted average; weights on a 0.1 grid, each at least 0.1, tuned on dev macro F1 | 1 weight |
| stacking | Multinomial logistic regression on the standardised log-probabilities, balanced class weights, L2 (lambda 0.01) | 136 |

## 3. Protocol

1. **Dev scores are out-of-sample.** The rules with parameters (weighted, stacking) are cross-fitted over 5 folds of dev films: fit on 40 films, predict the other 10. Equal and product have nothing to fit.
2. **One rule is frozen on dev.** The rule with the best dev macro F1 is chosen, refitted on all 50 dev films, and scored **once** on test.
3. **Intervals** are 95% film-level bootstrap intervals (1,000 resamples, seed 20261106). Comparisons with another system use the same resamples (paired bootstrap).

Everything is deterministic: fixed folds, fixed random stream, fixed solver steps.

## 4. Results

### Single models (macro F1)

| Model | Dev | Test | Test accuracy |
|---|---|---|---|
| Text | 0.299 | 0.305 (0.253 to 0.344) | 31.9% |
| Audio | 0.253 | 0.197 (0.152 to 0.235) | 20.7% |

### Dev, every rule (out-of-sample macro F1)

| Equal | Product | Weighted | Stacking | Chosen |
|---|---|---|---|---|
| 0.270 | 0.288 | **0.304** (text 0.7 / audio 0.3) | 0.221 | weighted |

### Test, frozen rule

| Rule | Test macro F1 (95%) | Test accuracy | Difference with video alone (95%) |
|---|---|---|---|
| weighted, text 0.7 / audio 0.3 | **0.291** (0.239 to 0.330) | 31.4% | -0.054 (-0.110 to -0.002) |

## 5. What the results show

- **Text + audio does not beat text alone.** On test it scores 0.291 against 0.305 for text alone. The audio model is much weaker (0.197 on test), so adding it mostly adds noise.
- **The tuned weights agree.** When tuned on dev, audio gets only 0.3 of the weight, and the rule that ignores the weak model least (equal) is the worst of the parameter-free rules.
- **Stacking is the worst rule** (0.221 on dev). It fits 136 parameters from 40 dev films per fold, which is too little data. Training it on the train out-of-fold predictions (2,207 clips) would be the fix.
- **This pair is the weakest combination.** It is the only fusion that is significantly below the video model alone (the interval of the difference excludes zero). The pairs that include video, and the three-way fusion, are in the other `Fusion` subfolders.

## 6. How to run

The fusion script `fuse_professor.py` runs every combination (text+audio, text+video, audio+video, text+audio+video) in one pass; the text+audio results are the `text+audio` rows of its outputs. It needs only `numpy` (and `matplotlib` for the figures).

```bash
pip install numpy matplotlib

python fuse_professor.py \
    --manifest splits/manifest.csv \
    --text  "results_text/predictions/text_both_char_C1p0_{split}.csv" \
    --audio "results/predictions/audio_ast_hand_{split}.csv" \
    --video "results/predictions/video_clip_meanstd_{split}.csv" \
    --out results_fusion
```

`{split}` is replaced by `dev` and `test` by the script. Add `--no-figures` to skip the confusion matrices.

### Outputs (in `results_fusion/`)

| File | Content |
|---|---|
| `fusion_report.md` | Tables for the report and slides |
| `fusion_report.json` | Every number |
| `fusion_dev_selection.csv` | Out-of-sample dev scores of every combination and rule |
| `fusion_test.csv` | The frozen rule of each combination, scored on test |
| `predictions/fusion_text_audio_weighted_{dev,test}.csv` | Fused probabilities, same format as the inputs |
