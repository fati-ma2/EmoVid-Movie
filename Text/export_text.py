#!/usr/bin/env python3
"""Package the text work for the teammate building the fusion model.

EN  Writes one folder (default export_text/) with:
      1. text probabilities: 8 emotion probabilities per clip from YOUR text model (the one
         text_emotion.py chose on dev, e.g. both_char_C1.0), retrained on the professor's train films:
         dev and test (identical to results_text/predictions) plus train out-of-fold by film;
      2. text embeddings: 384 numbers per clip for the caption and for the transcription, from the
         pretrained sentence model all-MiniLM-L6-v2 (no training on our labels);
      3. README.md in English and French describing every file.
FR  Crée un dossier (par défaut export_text/) contenant :
      1. les probabilités texte : 8 probabilités d'émotion par extrait issues de TON modèle texte (celui
         choisi sur dev par text_emotion.py), ré-entraîné sur les films train du professeur : dev et test
         (identiques à results_text/predictions) et train hors-pli par film ;
      2. les embeddings texte : 384 valeurs par extrait pour la légende (caption) et pour la transcription,
         issues du modèle de phrases pré-entraîné all-MiniLM-L6-v2 (aucun apprentissage sur nos étiquettes) ;
      3. un README.md en anglais et en français.

Run from the mmemo folder, with the same inputs as text_emotion.py:

    python scripts/export_text.py \\
        --train splits/professor/train.txt --dev splits/professor/dev.txt --test splits/professor/test.txt \\
        --transcripts splits/professor/transcriptions.csv --annotations ../PROJET_SID_2026/movie/annotation

The embeddings need `pip install sentence-transformers` (downloads a 90 MB model once).
Use --no-embeddings to export the probabilities only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import text_emotion as te  # noqa: E402  (same data loading and model definitions as the text system)

CLASSES = te.CLASSES
K = te.K
PROB_COLS = [f"p_{c}" for c in CLASSES]


def save_probs(path: Path, ids, proba, available) -> None:
    available = np.asarray(available, bool)
    proba = np.where(available[:, None], proba, 1 / K)
    df = pd.DataFrame(proba / proba.sum(axis=1, keepdims=True), columns=PROB_COLS)
    df.insert(0, "video_id", list(ids))
    df["available"] = available.astype(int)
    df.to_csv(path, index=False, float_format="%.8f")


def save_embeddings(out: Path, name: str, clips: pd.DataFrame, X: np.ndarray, available: np.ndarray) -> dict:
    cols = [f"emb_{i:03d}" for i in range(X.shape[1])]
    df = pd.concat([clips.reset_index(drop=True), pd.DataFrame({"available": available.astype(int)}),
                    pd.DataFrame(X, columns=cols)], axis=1)
    df.to_csv(out / f"{name}.csv", index=False, float_format="%.6g")
    np.savez_compressed(out / f"{name}.npz", video_id=clips["video_id"].to_numpy(str), X=X.astype(np.float32),
                        available=available)
    return {"file": f"{name}.csv / {name}.npz", "clips": int(len(df)), "numbers_per_clip": int(X.shape[1]),
            "empty_text": int((~available).sum())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transcripts", type=Path, required=True, help="the professor's transcription CSV")
    ap.add_argument("--annotations", type=Path, required=True, help="folder of the dataset JSON files")
    ap.add_argument("--train", type=Path)
    ap.add_argument("--dev", type=Path)
    ap.add_argument("--test", type=Path)
    ap.add_argument("--split-dir", type=Path)
    ap.add_argument("--results", type=Path, default=Path("results_text"),
                    help="output folder of text_emotion.py (to reuse its chosen model)")
    ap.add_argument("--model", default=None, help="candidate name, e.g. both_char_C1.0 (default: read from "
                                                  "results_text/metrics.json)")
    ap.add_argument("--no-embeddings", action="store_true", help="skip the sentence embeddings")
    ap.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out", type=Path, default=Path("export_text"))
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)
    if not (args.split_dir or (args.train and args.dev and args.test)):
        ap.error("give the professor's split with --train/--dev/--test or --split-dir")
    args.split = None
    t0 = time.time()

    # Which model? The one text_emotion.py chose on dev.
    chosen = args.model
    metrics_path = args.results / "metrics.json"
    if chosen is None:
        if not metrics_path.is_file():
            sys.exit(f"ERROR: {metrics_path} not found. Run text_emotion.py first, or pass --model both_char_C1.0")
        chosen = json.loads(metrics_path.read_text())["chosen"]
    cands = {c["name"]: c for c in te.build_candidates(["transcription", "caption", "both"], use_sbert=False)}
    if chosen not in cands:
        sys.exit(f"ERROR: model {chosen!r} is not a TF-IDF candidate of text_emotion.py "
                 f"(choose one of e.g. {list(cands)[:4]} ...)")
    cand = cands[chosen]
    print(f"Text model: {chosen}")

    # Data, exactly as text_emotion.py builds it.
    transcripts = te.load_transcripts(args.transcripts)
    split, _ = te.load_split(args)
    ann = te.load_annotations(args.annotations)
    split = te.expand_film_split(split, set(transcripts) | set(ann.index))
    ids = sorted(split)
    df = pd.DataFrame(index=pd.Index(ids, name="video_id"))
    df["split"] = [split[i] for i in ids]
    df["transcription"] = [transcripts.get(i, "") for i in ids]
    df["caption"] = ann["caption"].reindex(ids).fillna("").astype(str).to_numpy()
    df["emotion"] = ann["emotion"].reindex(ids).fillna("").to_numpy()
    bad = df[~df["emotion"].isin(CLASSES)]
    if len(bad):
        sys.exit(f"ERROR: {len(bad)} clips have no valid emotion label, e.g. {list(bad.index[:3])}")
    y = df["emotion"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    films = np.array([te.film_of(i) for i in ids])
    has_speech = df["transcription"].str.strip().astype(bool).to_numpy()
    rows = {s: np.flatnonzero(df["split"].to_numpy() == s) for s in ("train", "dev", "test")}
    prior = np.bincount(y[rows["train"]], minlength=K) / len(rows["train"])

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    clips = pd.DataFrame({"video_id": ids, "film_id": films, "split": df["split"].to_numpy(),
                          "emotion": df["emotion"].to_numpy(), "has_transcription": has_speech.astype(int)})
    clips.to_csv(out / "clips.csv", index=False)
    counts = {s: int(len(r)) for s, r in rows.items()}
    print(f"Clips: {counts}; transcription non-empty in {has_speech.mean():.1%}")
    summary = {"clips": int(len(ids)), "split": counts, "model": chosen, "source": cand["source"]}

    # 1. Probabilities.
    cols = ["transcription", "caption"] if cand["source"] == "both" else [cand["source"]]

    def avail(r):
        return has_speech[r] if cand["source"] == "transcription" else np.ones(len(r), bool)

    def fit(fit_r):
        fit_r = fit_r[avail(fit_r)]
        return cand["make"]().fit(df.iloc[fit_r][cols], y[fit_r])

    def predict(est, pred_r):
        p = np.full((len(pred_r), K), 1 / K)
        ok = avail(pred_r)
        if ok.any():
            p[ok] = te.proba_full(est, df.iloc[pred_r[ok]][cols])
        return p, ok

    print("Training on the professor's train films (same settings as text_emotion.py)...")
    est = fit(rows["train"])
    report = {"model": chosen, "source": cand["source"], "splits": {}}
    for s in ("dev", "test"):
        r = rows[s]
        p, ok = predict(est, r)
        m = te.metrics(y[r], te.labels_from(p, prior), films[r], args.n_boot)
        report["splits"][s] = {k: m[k] for k in ("n", "macro_f1", "macro_f1_ci95", "accuracy", "balanced_accuracy")}
        report["splits"][s]["coverage"] = float(ok.mean())
        save_probs(out / f"text_probabilities_{s}.csv", df.index[r], p, ok)
        print(f"  {s}: macro F1 {m['macro_f1']:.4f} (95% {m['macro_f1_ci95'][0]:.3f} to "
              f"{m['macro_f1_ci95'][1]:.3f}), accuracy {m['accuracy']:.4f}")
        check = args.results / "predictions" / f"text_{chosen.replace('.', 'p')}_{s}.csv"
        if check.is_file():
            old = pd.read_csv(check).set_index("clip_id").loc[df.index[r], PROB_COLS].to_numpy()
            diff = float(np.abs(old - p / p.sum(axis=1, keepdims=True)).max())
            report["splits"][s]["max_difference_vs_results_text"] = diff
            print(f"     same as {check.name}: {'yes' if diff < 1e-6 else 'NO, max difference ' + format(diff, '.2g')}")

    r = rows["train"]
    p_oof = np.full((len(r), K), 1 / K)
    ok_oof = avail(r)
    for fit_i, hold_i in GroupKFold(n_splits=5).split(r, y[r], films[r]):
        p_oof[hold_i], _ = predict(fit(r[fit_i]), r[hold_i])
    save_probs(out / "text_probabilities_train_oof.csv", df.index[r], p_oof, ok_oof)
    print("  train: out-of-fold probabilities written (5 folds by film)")
    (out / "text_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary["results"] = report["splits"]

    # 2. Embeddings.
    summary["embeddings"] = {}
    if not args.no_embeddings:
        try:
            emb = te.embed_fields(df, ["caption", "transcription"], args.sbert_model, args.results)
        except ImportError:
            print("WARNING: sentence-transformers is not installed, so no embeddings were written. "
                  "Run: python -m pip install sentence-transformers  (or use --no-embeddings)")
            emb = {}
        for field, X in emb.items():
            ok = df[field].str.strip().astype(bool).to_numpy()
            summary["embeddings"][field] = save_embeddings(out, f"text_embeddings_{field}", clips, X, ok)
            summary["embeddings"][field]["model"] = args.sbert_model
            print(f"  embeddings ({field}): {X.shape[1]} numbers per clip")

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_readme(out, summary)
    print(f"\nDone in {time.time() - t0:.0f} s. Send the whole folder: {out.resolve()}")
    return 0


def write_readme(out: Path, s: dict) -> None:
    src = {"both": "caption + transcription", "caption": "caption", "transcription": "transcription"}[s["source"]]
    src_fr = {"both": "légende + transcription", "caption": "légende", "transcription": "transcription"}[s["source"]]
    lines = ["# Text export for the fusion model / Export texte pour le modèle de fusion", "",
             f"{s['clips']} clips / extraits. Professor's split / split du professeur: "
             + ", ".join(f"{k} {v}" for k, v in s["split"].items()), "",
             "Every file is keyed by `video_id` (e.g. `00024_clip_003`), the same key as the audio export. / "
             "Chaque fichier est indexé par `video_id`, comme l'export audio.", "",
             "## Files / Fichiers", "",
             "| File / Fichier | EN | FR |", "|---|---|---|",
             "| `clips.csv` | video_id, film_id, split, emotion (the label: never use it as an input), "
             "has_transcription | video_id, film_id, split, emotion (l'étiquette : ne jamais l'utiliser comme "
             "entrée), has_transcription |",
             f"| `text_probabilities_dev.csv`, `_test.csv` | 8 emotion probabilities per clip (sum = 1) from the text "
             f"model `{s['model']}`: TF-IDF of character n-grams + logistic regression on the {src}, trained on the "
             "professor's TRAIN films only. Use dev to tune the fusion, test once at the end. "
             f"| 8 probabilités d'émotion par extrait (somme = 1) du modèle texte `{s['model']}` : TF-IDF de "
             f"n-grammes de caractères + régression logistique sur la {src_fr}, entraîné UNIQUEMENT sur les films "
             "TRAIN du professeur. Utiliser dev pour régler la fusion, test une seule fois à la fin. |",
             "| `text_probabilities_train_oof.csv` | Same, for train clips, out-of-fold by film (each clip predicted "
             "by a model that never saw its film). For a stacking fusion trained on train. | Idem pour les extraits "
             "train, hors-pli par film. Pour une fusion par stacking entraînée sur train. |"]
    for field, e in s["embeddings"].items():
        fr = "légende" if field == "caption" else "transcription"
        lines.append(f"| `text_embeddings_{field}.csv` / `.npz` | Embedding of the {field}: {e['numbers_per_clip']} "
                     f"numbers per clip from the pretrained model {e['model']} (not trained on our labels). "
                     f"{e['empty_text']} clips have an empty {field}: available = 0 and all zeros. "
                     f"| Embedding de la {fr} : {e['numbers_per_clip']} valeurs par extrait, modèle pré-entraîné "
                     f"{e['model']} (aucun apprentissage sur nos étiquettes). {e['empty_text']} extraits ont une "
                     f"{fr} vide : available = 0 et que des zéros. |")
    lines += ["| `text_metrics.json`, `summary.json` | Scores and settings. | Scores et paramètres. |", "",
              "## Text results / Résultats texte", "",
              "| Split | macro F1 (95%) | accuracy / exactitude |", "|---|---|---|"]
    for sp, v in s["results"].items():
        lines.append(f"| {sp} | {v['macro_f1']:.3f} ({v['macro_f1_ci95'][0]:.3f} to {v['macro_f1_ci95'][1]:.3f}) "
                     f"| {v['accuracy']:.1%} |")
    lines += ["", "Class order / ordre des classes: " + ", ".join(CLASSES), "",
              "Note: the captions were written by an AI model that watched the video, so the text model already "
              "carries some visual information. / Les légendes ont été écrites par un modèle d'IA à partir de la "
              "vidéo : le modèle texte contient donc déjà une part d'information visuelle.", "",
              "## Load in Python / Charger en Python", "",
              "```python",
              "import numpy as np, pandas as pd",
              "clips = pd.read_csv(\"clips.csv\", dtype={\"film_id\": str})",
              "p_dev = pd.read_csv(\"text_probabilities_dev.csv\").set_index(\"video_id\")",
              "z = np.load(\"text_embeddings_caption.npz\")",
              "emb = pd.DataFrame(z[\"X\"], index=z[\"video_id\"])",
              "```", ""]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
