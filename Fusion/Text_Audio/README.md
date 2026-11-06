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
3. **Intervals** are 95% film-level bootstrap intervals (1,000 resamples): the 50 films are redrawn with replacement, macro F1 is recomputed each time, and the middle 95% is kept. Comparisons with another system use the same resamples (paired bootstrap).

Everything is deterministic: fixed folds, fixed random stream, fixed solver steps.

## 4. Results

### Single models (macro F1)

| Model | Dev (95%) | Test (95%) | Test accuracy |
|---|---|---|---|
| Text | 0.299 (0.235 to 0.350) | 0.305 (0.254 to 0.345) | 31.9% |
| Audio | 0.253 | 0.197 (0.152 to 0.235) | 20.7% |

The text intervals are those of `results_text/metrics.md`. `fuse_professor.py` draws its own resamples and reports 0.253 to 0.344 for the same 0.305 test score.

### Dev, every rule (out-of-sample macro F1)

| Equal | Product | Weighted | Stacking | Chosen |
|---|---|---|---|---|
| 0.270 | 0.288 | **0.304** (text 0.7 / audio 0.3) | 0.221 | weighted |

### Test, frozen rule

| Rule | Test macro F1 (95%) | Test accuracy | Difference with video alone (95%) |
|---|---|---|---|
| weighted, text 0.7 / audio 0.3 | **0.291** (0.239 to 0.330) | 31.4% | -0.054 (-0.110 to -0.002) |

## 5. What the results show

- **Text + audio does not beat text alone.** On test it scores 0.291 against 0.305 for text alone. The intervals overlap almost completely, so the fusion is not better, but it is not clearly worse either.
- **Audio is too weak to add information.** It scores 0.197 on test, not far above random guessing (0.125), and its errors are spread across all eight emotions. It is also overconfident: on dev its top guess has about 47% probability on average but is right only about 26% of the time, so in the average it can pull a correct text prediction toward a wrong emotion.
- **The labels follow what is seen, not what is heard.** The captions and most labels of EmoVid were produced by vision-language models that watched the clips. The caption-based text model is therefore close to the labels, while movie sound (music, effects and speech mixed) is only loosely related to them.
- **The tuned weights agree.** When tuned on dev, audio gets only 0.3 of the weight.
- **Stacking is the worst rule** (0.221 on dev). It fits 136 parameters from 40 dev films per fold, which is too little data. Training it on the train out-of-fold predictions (2,207 clips) would be the fix.
- **This pair is the weakest combination.** It is the only fusion that is significantly below the video model alone (the interval of the difference excludes zero). The pairs that include video, and the three-way fusion, are in the other `Fusion` subfolders.

## 6. Why the test score is higher than the dev score for text, but lower for the fusion

| System | Dev | Test | Test minus dev |
|---|---|---|---|
| Text alone | 0.299 | 0.305 | +0.006 |
| Audio alone | 0.253 | 0.197 | -0.056 |
| Text + audio (weighted 0.7 / 0.3) | 0.304 | 0.291 | -0.013 |

**Usually the test score is lower than the dev score.** Every choice (the text candidate, the audio features, the fusion rule and its weights) was made on dev, and the winner of a comparison tends to be the option that was a little lucky on those dev films. That luck does not carry over to new films, so the score usually drops on test. Audio and the text + audio fusion follow this pattern.

**The text model's test score is slightly higher (0.305 against 0.299). This is noise, not a problem:**

1. **Dev and test are two different samples of 50 films.** Some films are easier than others (a caption that says "dark room" or "gun" makes the emotion easy to read). The score moves by a few hundredths depending on which films land in each set. A difference of 0.006 is far smaller than the width of either interval (about 0.09 to 0.11).
2. **The choice on dev added very little optimism.** The three best of the 27 text candidates were within 0.004 of each other on dev (0.299, 0.298, 0.295), so the chosen model was not a lucky outlier, and there was almost nothing to lose on test.
3. **The two intervals overlap almost completely** (dev 0.235 to 0.350, test 0.254 to 0.345). The right reading is that dev and test agree: the text model scores about 0.30 on films it has never seen.
4. **Test was used once.** Everything was frozen before test was scored, so the higher number does not come from tuning on test.

**The fusion's test score is lower (0.291 against 0.304), as expected:**

1. **The rule was chosen on dev.** The weighted rule beat the other three by a small margin on dev, so part of that margin is luck. The weights themselves were cross-fitted over dev films, which limits but does not remove this effect.
2. **Audio drops on test.** Audio alone falls from 0.253 on dev to 0.197 on test, and the fusion inherits part of that drop through its 0.3 weight.
3. **The drop is within the noise.** The test interval of the fusion (0.239 to 0.330) still overlaps the text model's dev and test intervals.

## 7. How to run

`fuse_professor.py` runs every combination (text+audio, text+video, audio+video, text+audio+video) in one pass; the text+audio results are the `text+audio` rows of its outputs. `confusion_text_audio.py` reruns the text+audio part only and draws its confusion matrices. Both need `numpy` and `matplotlib`; `confusion_text_audio.py` also imports `text_emotion.py` from `../../Text/`.

```bash
pip install numpy matplotlib

python fuse_professor.py \
    --manifest path/to/splits/manifest.csv \
    --text  "path/to/text_both_char_C1p0_{split}.csv" \
    --audio "path/to/audio_ast_hand_{split}.csv" \
    --video "path/to/video_clip_meanstd_{split}.csv" \
    --out results_fusion

python confusion_text_audio.py \
    --manifest path/to/splits/manifest.csv \
    --text  "path/to/text_both_char_C1p0_{split}.csv" \
    --audio "path/to/audio_ast_hand_{split}.csv" \
    --out results_fusion
```

`{split}` is replaced by `dev` and `test` by the scripts. Add `--no-figures` to `fuse_professor.py` to skip its figures. `confusion_text_audio.py` prints the test macro F1 (0.291) and accuracy (31.4%), which confirms the run matches the results above.

### Outputs (in `results_fusion/`)

| File | Content |
|---|---|
| `fusion_report.md` | Tables for the report and slides |
| `fusion_report.json` | Every number |
| `fusion_dev_selection.csv` | Out-of-sample dev scores of every combination and rule |
| `fusion_test.csv` | The frozen rule of each combination, scored on test |
| `predictions/fusion_text_audio_weighted_{dev,test}.csv` | Fused probabilities, same format as the inputs |
| `confusion_text_audio_{dev,test}.png` | Confusion matrices of the text + audio fusion (% of each true class) |
| `confusion_text_audio_{dev,test}.csv` | The same matrices as raw counts (rows = true class) |
