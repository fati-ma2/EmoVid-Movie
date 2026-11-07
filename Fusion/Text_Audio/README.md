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

| Rule | Test macro F1 (95%) | Test accuracy |
|---|---|---|
| weighted, text 0.7 / audio 0.3 | **0.291** (0.239 to 0.330) | 31.4% |

## 5. What the results show

- **Text + audio does not beat text alone.** On test it scores 0.291 against 0.305 for text alone. The audio model is much weaker (0.197 on test), so adding it mostly adds noise.
- **The tuned weights agree.** When tuned on dev, audio gets only 0.3 of the weight, and the rule that ignores the weak model least (equal) is the worst of the parameter-free rules.
- **Stacking is the worst rule** (0.221 on dev). It fits 136 parameters from 40 dev films per fold, which is too little data. Training it on the train out-of-fold predictions (2,207 clips) would be the fix.
- **Fusion stays well above audio alone.** The fused model (0.291) gains almost 0.10 macro F1 over audio alone (0.197), so most of its performance comes from the text side.

## 6. How to run

`confusion_text_audio.py` runs the text + audio fusion (same procedure as `fuse_professor.py`), prints the dev and test scores and writes the confusion matrices. It needs only the text and audio predictions, `numpy` and `matplotlib`, and `text_emotion.py` from `../../Text/` for the figure style.

```bash
pip install numpy matplotlib

python confusion_text_audio.py \
    --manifest ../../splits/manifest.csv \
    --text  "inputs/text_both_char_C1p0_{split}.csv" \
    --audio "inputs/audio_ast_hand_{split}.csv" \
    --out results
```

`{split}` is replaced by `dev` and `test` by the script. `fuse_professor.py` produces the full tables (`fusion_report.md` and the CSV files below); it is the shared fusion script of the project and also expects the other modalities' predictions.

### Outputs (in `results_fusion/`)

| File | Content |
|---|---|
| `fusion_report.md` | Tables for the report and slides |
| `fusion_report.json` | Every number |
| `fusion_dev_selection.csv` | Out-of-sample dev scores of every combination and rule |
| `fusion_test.csv` | The frozen rule of each combination, scored on test |
| `predictions/fusion_text_audio_weighted_{dev,test}.csv` | Fused probabilities, same format as the inputs |
| `confusion_text_audio_{dev,test}.png` / `.csv` | Confusion matrices of the text+audio fusion (from `confusion_text_audio.py`) |

## 7. Conclusion

Combining text and audio does not improve emotion recognition on this dataset. The rule chosen on dev, a weighted average with 0.7 on text and 0.3 on audio, reaches a test macro F1 of 0.291 (95% interval 0.239 to 0.330) and an accuracy of 31.4%. That is slightly below the text model alone (0.305) and well above the audio model alone (0.197).

The main reason is the gap in quality between the two inputs. The audio model is weak (0.197 macro F1 on test, against 0.305 for text), so giving it any weight pulls the fused prediction away from the better text model. The fusion procedure itself behaved as intended: the tuned weights favour text, and the parameter-free rules confirm the ranking (product 0.288 is above equal 0.270 on dev). For this project, text + audio is best seen as a negative result: it shows that fusion only helps when the second modality brings comparable and complementary information.

## 8. Limitations

- **Small dev set.** The fusion rule and its weights are chosen on 50 dev films (442 clips). Differences of about 0.01 to 0.02 macro F1 between rules are within noise, and the confidence intervals on test are wide (about 0.09 macro F1).
- **Weak audio input.** The audio model drops from 0.253 on dev to 0.197 on test, so the fusion was tuned on an audio signal that looked more useful on dev than it is on new films. A stronger audio model would be needed before this pair can be judged fairly.
- **Late fusion only.** The two models are combined at the level of their output probabilities. Early fusion of the features (for example the text sentence embeddings and the AST audio embeddings) was not tested.
- **Stacking trained on too little data.** Stacking fits 136 parameters from 40 dev films per fold, which explains its poor score (0.221 on dev). Training it on the train out-of-fold predictions (2,207 clips) would give it a fair chance.
- **The text input is not pure text.** The captions were written by an AI model that watched the video, so the text model already carries some visual information. Part of what text adds to audio here is indirectly visual.
- **Missing or unknown speech.** The transcriptions were provided by the professor with an unknown method, and clips without speech have an empty transcription. The audio model sees music and sound effects as well as speech, so the two models do not always describe the same content.
- **One test run, several comparisons.** Each combination was scored once on test, but four combinations and four rules were compared on dev, and no correction for multiple comparisons was applied.
