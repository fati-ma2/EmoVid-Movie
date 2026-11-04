# Text modality

This folder holds the **text model** of the Group 3 project: recognising one of eight emotions in an EmoVid movie clip from its text alone. The text model's probabilities are one of the inputs to the fusion models in `../Fusion/`.

**Classes (fixed order, used in every file):** amusement, awe, contentment, excitement, anger, disgust, fear, sadness

## 1. Data

| Input | Where it comes from | Notes |
|---|---|---|
| Transcription | The professor's transcription CSV (`video_id`, `audio_path`, `transcription`) | What is said in the clip. Some clips have no speech, so their transcription is empty. |
| Caption | The `caption` field of each clip's JSON file in `PROJET_SID_2026/movie/annotation/` | A written description of the clip. |
| Label | The `emotion` field of the same JSON file | Used only as the target, never as an input. |

**Split:** the professor's shared split, by film, so no film appears in more than one set.

| Set | Films | Clips | Role |
|---|---|---|---|
| train | 250 | 2,207 | fitting the models |
| dev | 50 | 442 | choosing the model |
| test | 50 | 439 | final score, used **once** |

The split files list film IDs (for example `00024`). The script expands them to clip IDs (`00024_clip_003`) on its own.

## 2. Method

Every candidate is trained on **train only** and scored on **dev**. The candidate with the best dev macro F1 is kept, and only that model is scored on test.

| Choice | Options tried |
|---|---|
| Text source | transcription only, caption only, both (one vectorizer per field) |
| Representation | TF-IDF of words (1-grams), TF-IDF of words (1- and 2-grams), TF-IDF of character n-grams (2 to 5) |
| Classifier | Logistic regression with balanced class weights, C in {0.1, 1, 10} |
| Optional | Frozen sentence embeddings (`all-MiniLM-L6-v2`, 384 values) + logistic regression, with `--sbert` |

That gives 27 TF-IDF candidates (36 with `--sbert`).

Rules that keep the evaluation honest:
- Vectorizers sit inside the pipeline, so they only ever see train text.
- A clip with an empty transcription is marked `available = 0` and gets a uniform distribution (1/8 per class) when the model uses the transcription only.
- Ties in the predicted class go to the most frequent training class.
- Confidence intervals are 95% **film-level bootstrap** intervals (1,000 resamples), because clips from the same film are not independent.

## 3. Results

**Chosen model on dev: `both_char_C1.0`**, character n-gram TF-IDF of the caption and the transcription, logistic regression with C = 1.

| Set | Macro F1 (95% interval) | Accuracy |
|---|---|---|
| dev | 0.299 | |
| test | **0.305** (0.254 to 0.345) | 31.9% |

Chance level for 8 classes is about 0.125 macro F1. The full tables, with the majority-class and random baselines and per-class scores, are written to `results_text/metrics.md` and `results_text/per_class.csv`.

## 4. Files

| File | What it does |
|---|---|
| `text_emotion.py` | Trains every candidate, selects on dev, evaluates the chosen model on test, writes the predictions and figures. |
| `export_text.py` | Packages the text work for the fusion step: the chosen model's probabilities (dev, test, and train out-of-fold by film) and the sentence embeddings of the caption and the transcription. |

### Outputs of `text_emotion.py` (in `results_text/`)

| File | Content |
|---|---|
| `dev_selection.csv` | Dev macro F1, accuracy and coverage of every candidate |
| `metrics.json`, `metrics.md` | Dev and test scores of the chosen model and two baselines, with intervals |
| `per_class.csv` | Precision, recall, F1 and support per class |
| `confusion_dev.png`, `confusion_test.png` | Confusion matrices (rows sum to 100%) |
| `top_terms.csv` | The 15 strongest features per class |
| `predictions/text_both_char_C1p0_{dev,test}.csv` | Class probabilities per clip: `clip_id, p_amusement ... p_sadness, available` |
| `model.joblib` | The fitted model |

### Outputs of `export_text.py` (in `export_text/`)

| File | Content |
|---|---|
| `clips.csv` | `video_id, film_id, split, emotion, has_transcription` |
| `text_probabilities_{dev,test}.csv` | Same probabilities as `results_text/predictions` |
| `text_probabilities_train_oof.csv` | Train probabilities, out-of-fold by film (5 folds), for a fusion model trained on train |
| `text_embeddings_{caption,transcription}.csv` / `.npz` | 384 values per clip from `all-MiniLM-L6-v2` (not trained on our labels) |
| `text_metrics.json`, `summary.json`, `README.md` | Scores, settings and a description of the files |

## 5. How to run

Requirements: Python 3.10+, `numpy`, `pandas`, `scikit-learn`, `joblib`, `matplotlib`. Add `sentence-transformers` for `--sbert` and for the embeddings in `export_text.py`.

```bash
pip install numpy pandas scikit-learn joblib matplotlib
pip install sentence-transformers        # optional

# 1. Train, select on dev, evaluate on test
python text_emotion.py \
    --train splits/professor/train.txt \
    --dev splits/professor/dev.txt \
    --test splits/professor/test.txt \
    --transcripts splits/professor/transcriptions.csv \
    --annotations ../PROJET_SID_2026/movie/annotation

# 2. Export the probabilities and embeddings for fusion
python export_text.py \
    --train splits/professor/train.txt \
    --dev splits/professor/dev.txt \
    --test splits/professor/test.txt \
    --transcripts splits/professor/transcriptions.csv \
    --annotations ../PROJET_SID_2026/movie/annotation
```

Useful options of `text_emotion.py`: `--sources caption both` to limit the text inputs, `--sbert` to add sentence embeddings, `--no-test` to stop after the dev selection, `--out` to change the output folder. Run `python text_emotion.py --help` for the full list.

Adjust the paths to where the split files and the dataset are on your machine. The dataset itself is not in this repository.

## 6. Limitations

- **Captions carry visual information.** They were written by an AI model that watched the video, so the "text" model is partly a description of the images. This matters when comparing it with the video model and when reading the fusion results.
- **The transcription method is unknown.** The transcriptions were provided by the professor, and we could not check their quality or language.
- **Small dev set.** Selection is made on 50 dev films, so close candidates (differences of 0.01 macro F1) are within noise.
- **Simple models.** TF-IDF with logistic regression was the strongest option we tried; the frozen sentence embeddings did not beat it on dev.
