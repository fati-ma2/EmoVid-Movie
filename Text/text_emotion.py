#!/usr/bin/env python3
"""Text emotion recognition on the professor's shared train / dev / test split.

Inputs
    --transcripts  CSV: column 1 = video ID, column 2 = audio_path, column 3 = transcription
                   (header row optional; comma, semicolon or tab separated)
    --split        the professor's split, any one of:
                     * one CSV with the video ID in column 1 and a column of train / dev / test
                       values (val / validation are read as dev); it may also hold the emotion
                     * --split-dir: a folder of files whose names contain train, dev (or val) and test,
                       video ID in column 1
                     * --train / --dev / --test: one file each, video ID in column 1
    --annotations  folder of the dataset's JSON files (emotion label + caption), e.g.
                   ../PROJET_SID_2026/movie/annotation. Needed unless the split file has the labels.

What it does
    1. Joins the three sources on the video ID and reports anything missing.
    2. Trains every candidate on TRAIN only:
         text source:     transcription | caption | both (caption needs --annotations)
         representation:  TF-IDF words (1-gram, 1-2-grams), TF-IDF characters (2-5),
                          optional frozen sentence embeddings (--sbert)
         classifier:      logistic regression, balanced class weights, C in {0.1, 1, 10}
    3. Picks the candidate with the best DEV macro F1 (the test set plays no part in the choice).
    4. Evaluates that one model on TEST once (skip with --no-test).

Outputs (in --out, default results_text/)
    dev_selection.csv           every candidate's dev scores
    metrics.json / metrics.md   dev and test scores of the chosen model and two baselines,
                                with 95% bootstrap intervals (by film when the ID is <film>_clip_<n>)
    per_class.csv, confusion_dev.png, confusion_test.png, top_terms.csv
    predictions/text_<name>_{dev,test}.csv   class probabilities (clip_id, p_amusement..p_sadness,
                                available), the same format the fusion step uses
    model.joblib

Example
    python scripts/text_emotion.py \\
        --transcripts transcriptions.csv \\
        --split professor_split.csv \\
        --annotations ../PROJET_SID_2026/movie/annotation

Needs: numpy, pandas, scikit-learn, joblib; matplotlib for the figures;
sentence-transformers only with --sbert.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

CLASSES = ("amusement", "awe", "contentment", "excitement", "anger", "disgust", "fear", "sadness")
K = len(CLASSES)
SPLIT_WORDS = {"train": "train", "training": "train", "trn": "train",
               "dev": "dev", "val": "dev", "valid": "dev", "validation": "dev", "development": "dev",
               "test": "test", "tst": "test", "testing": "test"}
HEADER_WORDS = {"video_id", "videoid", "video", "id", "clip_id", "clip", "name", "file", "filename", "key"}
SEED = 0


# ------------------------------------------------------------------ reading files
def norm_id(value: str) -> str:
    """'path/to/00012_clip_003.mp4' -> '00012_clip_003'; also strips spaces and quotes."""
    v = str(value).strip().strip('"').strip("'")
    v = v.replace("\\", "/").split("/")[-1]
    return re.sub(r"\.(mp4|mp3|wav|json|avi|mkv|txt)$", "", v, flags=re.IGNORECASE)


def read_rows(path: Path) -> list[list[str]]:
    if not path.is_file():
        sys.exit(f"ERROR: file not found: {path}")
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    sample = text[:20000]
    try:
        delim = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        delim = "\t" if "\t" in sample else ","
    rows = [r for r in csv.reader(text.splitlines(), delimiter=delim) if any(c.strip() for c in r)]
    if not rows:
        sys.exit(f"ERROR: {path} is empty")
    return rows


def drop_header(rows: list[list[str]]) -> tuple[list[list[str]], list[str] | None]:
    first = rows[0][0].strip().lower().replace(" ", "_")
    if first in HEADER_WORDS or (first.endswith("id") and not re.search(r"\d", first)):
        return rows[1:], rows[0]
    return rows, None


def load_transcripts(path: Path) -> dict[str, str]:
    rows, header = drop_header(read_rows(path))
    if any(len(r) < 3 for r in rows[:50]):
        sys.exit(f"ERROR: {path}: expected 3 columns (video ID, audio_path, transcription); "
                 f"first row has {len(rows[0])}: {rows[0][:3]}")
    out: dict[str, str] = {}
    dups = []
    for r in rows:
        vid = norm_id(r[0])
        text = ",".join(r[2:]).strip() if len(r) > 3 else r[2].strip()  # tolerate unquoted commas
        if text.lower() in {"nan", "none", "null"}:
            text = ""
        if vid in out:
            dups.append(vid)
        out[vid] = text
    if dups:
        print(f"WARNING: {len(dups)} duplicate video IDs in {path.name} (last one kept), e.g. {dups[:3]}")
    print(f"Transcripts: {len(out)} videos ({sum(bool(t) for t in out.values())} with text)"
          + (f"; header: {header}" if header else ""))
    return out


def _column_matching(rows, values: set[str], skip: int = 0) -> int | None:
    width = min(len(r) for r in rows)
    for j in range(width):
        if j == skip:
            continue
        hits = sum(r[j].strip().lower() in values for r in rows)
        if hits >= 0.9 * len(rows):
            return j
    return None


def load_split(args) -> tuple[dict[str, str], dict[str, str]]:
    """Returns {video_id: train|dev|test} and, if the split file has them, {video_id: emotion}."""
    split, labels = {}, {}
    if args.split:
        rows, header = drop_header(read_rows(args.split))
        j = _column_matching(rows, set(SPLIT_WORDS))
        if j is None:
            sys.exit(f"ERROR: no column of train/dev/test values found in {args.split}. "
                     f"First row: {rows[0]}. Use --split-dir or --train/--dev/--test instead.")
        lj = _column_matching(rows, set(CLASSES))
        for r in rows:
            split[norm_id(r[0])] = SPLIT_WORDS[r[j].strip().lower()]
            if lj is not None:
                labels[norm_id(r[0])] = r[lj].strip().lower()
    else:
        files: dict[str, list[Path]] = {"train": [], "dev": [], "test": []}
        if args.split_dir:
            for f in sorted(Path(args.split_dir).iterdir()):
                name = f.stem.lower()
                if not f.is_file():
                    continue
                if "test" in name:
                    files["test"].append(f)
                elif re.search(r"(dev|val)", name):
                    files["dev"].append(f)
                elif "train" in name:
                    files["train"].append(f)
        for s in ("train", "dev", "test"):
            if getattr(args, s):
                files[s].append(Path(getattr(args, s)))
        if not all(files.values()):
            sys.exit(f"ERROR: could not find all three split files; found {({k: [p.name for p in v] for k, v in files.items()})}")
        for s, paths in files.items():
            for p in paths:
                rows, _ = drop_header(read_rows(p))
                lj = _column_matching(rows, set(CLASSES))
                for r in rows:
                    split[norm_id(r[0])] = s
                    if lj is not None:
                        labels[norm_id(r[0])] = r[lj].strip().lower()
    counts = pd.Series(split).value_counts().to_dict()
    print(f"Split: {counts}" + (" (labels found in the split file)" if labels else ""))
    return split, labels


def film_of(clip_id: str) -> str:
    return clip_id.split("_clip_")[0] if "_clip_" in clip_id else clip_id


def expand_film_split(split: dict[str, str], known_clips: set[str]) -> dict[str, str]:
    """If the split lists FILM IDs (e.g. '00024') rather than clip IDs ('00024_clip_003'),
    give every clip of a film that film's split. Clip-level splits are returned unchanged."""
    if not known_clips or sum(k in known_clips for k in split) >= 0.5 * len(split):
        return split
    clips_by_film: dict[str, list[str]] = {}
    for c in known_clips:
        clips_by_film.setdefault(film_of(c), []).append(c)
    if sum(k in clips_by_film for k in split) < 0.5 * len(split):
        return split  # neither clip nor film IDs: the label check below reports it
    missing = [f for f in split if f not in clips_by_film]
    if missing:
        print(f"WARNING: {len(missing)} films in the split have no clips in the data, e.g. {missing[:3]}")
    out = {c: s for f, s in split.items() for c in clips_by_film.get(f, [])}
    unassigned = len(known_clips) - len(out)
    counts = pd.Series(out).value_counts().reindex(["train", "dev", "test"]).fillna(0).astype(int).to_dict()
    print(f"The split lists {len(split)} film IDs: expanded to {len(out)} clips {counts}"
          + (f"; {unassigned} clips belong to films not in the split" if unassigned else ""))
    return out


def load_annotations(folder: Path | None) -> pd.DataFrame:
    if folder is None:
        return pd.DataFrame(columns=["emotion", "caption"])
    if not folder.is_dir():
        sys.exit(f"ERROR: annotation folder not found: {folder}")
    rows = []
    for p in folder.rglob("*.json"):
        with p.open(encoding="utf-8") as fh:
            rec = json.load(fh)
        rows.append({"video_id": norm_id(rec.get("video_id") or p.stem),
                     "emotion": str(rec.get("emotion", "")).strip().lower(),
                     "caption": str(rec.get("caption") or "")})
    df = pd.DataFrame(rows).drop_duplicates("video_id").set_index("video_id")
    print(f"Annotations: {len(df)} JSON files")
    return df


# ------------------------------------------------------------------ models
def word_tfidf(ngram):
    return TfidfVectorizer(lowercase=True, strip_accents="unicode", sublinear_tf=True, ngram_range=ngram,
                           min_df=2, max_features=50000)


def char_tfidf():
    return TfidfVectorizer(lowercase=True, strip_accents="unicode", sublinear_tf=True, analyzer="char_wb",
                           ngram_range=(2, 5), min_df=3, max_features=100000)


def logreg(C):
    return LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=SEED)


def build_candidates(sources: list[str], use_sbert: bool) -> list[dict]:
    cands = []
    for src in sources:
        fields = ["transcription", "caption"] if src == "both" else [src]
        for rep, maker in (("word1", lambda: word_tfidf((1, 1))), ("word12", lambda: word_tfidf((1, 2))),
                           ("char", char_tfidf)):
            for C in (0.1, 1.0, 10.0):
                def make(fields=fields, maker=maker, C=C):
                    return Pipeline([("tfidf", ColumnTransformer([(f, maker(), f) for f in fields])),
                                     ("lr", logreg(C))])
                cands.append({"name": f"{src}_{rep}_C{C}", "source": src, "kind": "text", "make": make})
        if use_sbert:
            for C in (0.01, 0.1, 1.0):
                cands.append({"name": f"{src}_sbert_C{C}", "source": src, "kind": "emb", "fields": fields,
                              "make": lambda C=C: make_pipeline(StandardScaler(), logreg(C))})
    return cands


def embed_fields(df: pd.DataFrame, fields: list[str], model_name: str, cache_dir: Path) -> dict[str, np.ndarray]:
    from sentence_transformers import SentenceTransformer

    model = None
    out = {}
    for f in fields:
        cache = cache_dir / f"emb_{f}_{model_name.split('/')[-1]}.npz"
        if cache.is_file():
            z = np.load(cache, allow_pickle=False)
            if list(z["ids"]) == list(df.index):
                out[f] = z["X"]
                continue
        if model is None:
            model = SentenceTransformer(model_name)
        texts = df[f].tolist()
        X = np.zeros((len(texts), model.get_sentence_embedding_dimension()), dtype=np.float32)
        idx = [i for i, t in enumerate(texts) if t.strip()]
        if idx:
            X[idx] = model.encode([texts[i] for i in idx], batch_size=64, normalize_embeddings=True,
                                  show_progress_bar=True)
        np.savez_compressed(cache, ids=np.array(df.index, dtype=str), X=X)
        out[f] = X
    return out


def proba_full(est, X) -> np.ndarray:
    p = est.predict_proba(X)
    full = np.zeros((p.shape[0], K))
    full[:, np.asarray(est.classes_, dtype=int)] = p
    return full


def labels_from(proba, prior):
    return (np.asarray(proba) + 1e-9 * prior).argmax(axis=1)  # uniform rows -> most frequent class


# ------------------------------------------------------------------ evaluation
def metrics(y, pred, groups, n_boot: int = 1000) -> dict:
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=list(range(K)), zero_division=0)
    out = {"n": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
           "macro_f1": float(f1_score(y, pred, labels=list(range(K)), average="macro", zero_division=0)),
           "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
           "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                         for i, c in enumerate(CLASSES)},
           "confusion": confusion_matrix(y, pred, labels=list(range(K))).tolist()}
    uniq = np.unique(groups)
    rows = [np.flatnonzero(groups == g) for g in uniq]
    rng = np.random.default_rng(SEED)
    f1s, accs = [], []
    for _ in range(n_boot):
        pick = np.concatenate([rows[i] for i in rng.integers(0, len(uniq), len(uniq))])
        f1s.append(f1_score(y[pick], pred[pick], labels=list(range(K)), average="macro", zero_division=0))
        accs.append(accuracy_score(y[pick], pred[pick]))
    out["macro_f1_ci95"] = [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))]
    out["accuracy_ci95"] = [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))]
    out["bootstrap_unit_count"] = int(len(uniq))
    return out


def plot_confusion(cm, title, path, counts: bool = False) -> None:
    """counts=False: % of each true class (rows sum to 100). counts=True: number of clips in each cell."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        return
    cm = np.asarray(cm, float)
    support = cm.sum(axis=1)
    norm = cm / np.maximum(support[:, None], 1)
    cmap = LinearSegmentedColormap.from_list("b", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
    fig, ax = plt.subplots(figsize=(6.4, 5.6), facecolor="#fcfcfb")
    shade = cm / max(cm.max(), 1) if counts else norm
    ax.imshow(shade, cmap=cmap, vmin=0, vmax=1)
    for i in range(K):
        for j in range(K):
            if counts:
                txt = f"{int(cm[i, j])}"
            else:
                txt = "-" if support[i] == 0 else f"{norm[i, j] * 100:.0f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if shade[i, j] > 0.55 else "#0b0b0b")
    ax.set_xticks(range(K), CLASSES, rotation=40, ha="right")
    ax.set_yticks(range(K), [f"{c} (n={int(n)})" for c, n in zip(CLASSES, support)])
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(title, loc="left", fontweight="bold", pad=20)
    note = (f"number of clips (each row sums to its n; {int(cm.sum())} clips, {int(np.trace(cm))} correct on the diagonal)"
            if counts else "% of each true class (rows sum to 100)")
    ax.text(0, 1.01, note, transform=ax.transAxes, fontsize=8.5, color="#52514e")
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_predictions(path: Path, ids, proba, available) -> None:
    uniform = np.full(K, 1 / K)
    proba = np.where(np.asarray(available)[:, None] == 1, proba, uniform)
    proba = proba / proba.sum(axis=1, keepdims=True)
    df = pd.DataFrame(proba, columns=[f"p_{c}" for c in CLASSES])
    df.insert(0, "clip_id", list(ids))
    df["available"] = np.asarray(available, dtype=int)
    df.sort_values("clip_id").to_csv(path, index=False, float_format="%.8f")


# ------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transcripts", type=Path, required=True)
    ap.add_argument("--split", type=Path)
    ap.add_argument("--split-dir", type=Path)
    ap.add_argument("--train", type=Path)
    ap.add_argument("--dev", type=Path)
    ap.add_argument("--test", type=Path)
    ap.add_argument("--annotations", type=Path, help="folder of the dataset JSON files (labels and captions)")
    ap.add_argument("--sources", nargs="+", default=None, choices=["transcription", "caption", "both"],
                    help="text inputs to try (default: all available)")
    ap.add_argument("--sbert", action="store_true", help="also try frozen sentence embeddings")
    ap.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--no-test", action="store_true", help="stop after dev selection")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=Path("results_text"))
    args = ap.parse_args(argv)
    if not (args.split or args.split_dir or (args.train and args.dev and args.test)):
        ap.error("give the professor's split with --split, --split-dir, or --train/--dev/--test")

    t0 = time.time()
    transcripts = load_transcripts(args.transcripts)
    split, split_labels = load_split(args)
    ann = load_annotations(args.annotations)
    known_clips = set(transcripts) | set(ann.index)
    split = expand_film_split(split, known_clips)
    if split_labels and not (set(split_labels) & set(split)):
        split_labels = {}  # labels given per film cannot be used for clips

    ids = sorted(split)
    df = pd.DataFrame(index=pd.Index(ids, name="video_id"))
    df["split"] = [split[i] for i in ids]
    df["transcription"] = [transcripts.get(i, "") for i in ids]
    df["caption"] = ann["caption"].reindex(ids).fillna("").astype(str).to_numpy() if len(ann) else ""
    label_src = split_labels or (ann["emotion"].to_dict() if len(ann) else {})
    df["emotion"] = [label_src.get(i, "") for i in ids]
    if split_labels and len(ann):
        a = ann["emotion"].reindex(ids)
        clash = [i for i in ids if i in split_labels and isinstance(a.get(i), str) and a[i] and a[i] != split_labels[i]]
        if clash:
            print(f"WARNING: {len(clash)} videos have a different emotion in the split file and the JSON, e.g. {clash[:3]}; "
                  "using the split file")

    no_text = [i for i in ids if i not in transcripts]
    if no_text:
        print(f"WARNING: {len(no_text)} split videos are missing from the transcription CSV (treated as empty), "
              f"e.g. {no_text[:3]}")
    extra = len(set(transcripts) - set(ids))
    if extra:
        print(f"Note: {extra} transcription rows are not in the split and are ignored.")
    bad = df[~df["emotion"].isin(CLASSES)]
    if len(bad):
        sys.exit(f"ERROR: {len(bad)} split videos have no valid emotion label (e.g. {list(bad.index[:3])}). "
                 "Pass --annotations, or check that the video IDs match the JSON file names.")
    df["y"] = df["emotion"].map({c: i for i, c in enumerate(CLASSES)})
    df["group"] = [i.split("_clip_")[0] if "_clip_" in i else i for i in ids]
    films_in_many = df.groupby("group")["split"].nunique()
    unit = "film" if (df["group"] != df.index).any() else "clip"
    if unit == "film" and (films_in_many > 1).any():
        print(f"WARNING: {int((films_in_many > 1).sum())} films appear in more than one split "
              "(check the professor's split; scores may be optimistic)")

    available_sources = ["transcription"] + (["caption", "both"] if df["caption"].str.strip().any() else [])
    sources = args.sources or available_sources
    missing_src = set(sources) - set(available_sources)
    if missing_src:
        sys.exit(f"ERROR: {sorted(missing_src)} need captions: pass --annotations")

    tr, dv, te = (np.flatnonzero(df["split"].to_numpy() == s) for s in ("train", "dev", "test"))
    if not len(tr) or not len(dv):
        sys.exit("ERROR: the split must contain train and dev videos")
    y = df["y"].to_numpy()
    prior = np.bincount(y[tr], minlength=K) / len(tr)
    has_speech = df["transcription"].str.strip().astype(bool).to_numpy()
    print(f"Data: train {len(tr)}, dev {len(dv)}, test {len(te)} videos; transcription non-empty in "
          f"{has_speech.mean():.1%}; bootstrap unit: {unit}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "predictions").mkdir(exist_ok=True)
    emb = {}
    if args.sbert:
        emb = embed_fields(df, ["transcription", "caption"] if "caption" in available_sources else ["transcription"],
                           args.sbert_model, args.out)

    def inputs(cand, rows):
        if cand["kind"] == "emb":
            return np.hstack([emb[f][rows] for f in cand["fields"]])
        cols = ["transcription", "caption"] if cand["source"] == "both" else [cand["source"]]
        return df.iloc[rows][cols]

    def avail(cand, rows):
        return has_speech[rows] if cand["source"] == "transcription" else np.ones(len(rows), bool)

    def predict(est, cand, rows):
        ok = avail(cand, rows)
        p = np.full((len(rows), K), 1 / K)
        if ok.any():
            p[ok] = proba_full(est, inputs(cand, rows[ok]))
        return p, ok

    # Selection on dev.
    results, best = [], None
    print(f"\nTraining {len(build_candidates(sources, args.sbert))} candidates on train, scoring on dev:")
    for cand in build_candidates(sources, args.sbert):
        fit_rows = tr[avail(cand, tr)]
        est = cand["make"]()
        try:
            est.fit(inputs(cand, fit_rows), y[fit_rows])
        except ValueError as exc:  # e.g. empty vocabulary
            print(f"  {cand['name']:<34} skipped: {exc}")
            continue
        p_dev, ok_dev = predict(est, cand, dv)
        pred = labels_from(p_dev, prior)
        f1 = f1_score(y[dv], pred, labels=list(range(K)), average="macro", zero_division=0)
        acc = accuracy_score(y[dv], pred)
        results.append({"candidate": cand["name"], "source": cand["source"], "dev_macro_f1": round(f1, 4),
                        "dev_accuracy": round(acc, 4), "dev_coverage": round(float(ok_dev.mean()), 3)})
        print(f"  {cand['name']:<34} dev macro F1 {f1:.4f}  accuracy {acc:.4f}")
        if best is None or f1 > best[0]:
            best = (f1, cand, est, p_dev, ok_dev)
    if best is None:
        sys.exit("ERROR: no candidate could be trained")
    table = pd.DataFrame(results).sort_values("dev_macro_f1", ascending=False)
    table.to_csv(args.out / "dev_selection.csv", index=False)
    print("\nBest per text source on dev:")
    print(table.groupby("source").head(1).to_string(index=False))
    f1_best, cand, est, p_dev, ok_dev = best
    print(f"\nChosen on dev: {cand['name']} (dev macro F1 {f1_best:.4f})")

    # Final evaluation.
    groups = df["group"].to_numpy()
    report = {"chosen": cand["name"], "source": cand["source"], "selected_on": "dev macro F1",
              "bootstrap_unit": unit, "splits": {}}
    pred_name = f"text_{cand['name'].replace('.', 'p')}"
    eval_sets = [("dev", dv, p_dev, ok_dev)]
    if len(te) and not args.no_test:
        p_te, ok_te = predict(est, cand, te)
        eval_sets.append(("test", te, p_te, ok_te))
    per_class_rows = []
    rng = np.random.default_rng(SEED)
    for name, rows, proba, ok in eval_sets:
        pred = labels_from(proba, prior)
        m = metrics(y[rows], pred, groups[rows], args.n_boot)
        m["coverage"] = float(ok.mean())
        maj = np.full(len(rows), prior.argmax())
        rnd = rng.choice(K, size=len(rows), p=prior)
        report["splits"][name] = {
            "model": m,
            "baseline_majority": {k: v for k, v in metrics(y[rows], maj, groups[rows], args.n_boot).items()
                                  if k in ("macro_f1", "macro_f1_ci95", "accuracy")},
            "baseline_random": {k: v for k, v in metrics(y[rows], rnd, groups[rows], args.n_boot).items()
                                if k in ("macro_f1", "macro_f1_ci95", "accuracy")},
        }
        for c, v in m["per_class"].items():
            per_class_rows.append({"split": name, "class": c, **{k: round(x, 4) if isinstance(x, float) else x
                                                                for k, x in v.items()}})
        write_predictions(args.out / "predictions" / f"{pred_name}_{name}.csv", df.index[rows], proba, ok.astype(int))
        plot_confusion(m["confusion"], f"Text model ({cand['name']}), {name}", args.out / f"confusion_{name}.png")
        plot_confusion(m["confusion"], f"Text model ({cand['name']}), {name}", args.out / f"confusion_{name}_counts.png",
                       counts=True)
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(per_class_rows).to_csv(args.out / "per_class.csv", index=False)

    lines = ["| split | system | macro F1 | 95% interval | accuracy |", "|---|---|---|---|---|"]
    for name, r in report["splits"].items():
        for sys_name, key in (("text model", "model"), ("majority class", "baseline_majority"),
                              ("random (train frequencies)", "baseline_random")):
            v = r[key]
            lines.append(f"| {name} | {sys_name} | {v['macro_f1']:.4f} | "
                         f"{v['macro_f1_ci95'][0]:.4f} to {v['macro_f1_ci95'][1]:.4f} | {v['accuracy']:.4f} |")
    (args.out / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))

    if cand["kind"] == "text":
        names = est.named_steps["tfidf"].get_feature_names_out()
        lr = est.named_steps["lr"]
        top = [{"class": CLASSES[int(c)], "rank": k + 1, "term": names[j], "weight": round(float(row[j]), 4)}
               for row, c in zip(lr.coef_, lr.classes_) for k, j in enumerate(np.argsort(row)[::-1][:15])]
        pd.DataFrame(top).to_csv(args.out / "top_terms.csv", index=False)
    joblib.dump({"model": est, "candidate": cand["name"], "classes": CLASSES}, args.out / "model.joblib")
    print(f"\nDone in {time.time() - t0:.0f} s. Outputs in {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
