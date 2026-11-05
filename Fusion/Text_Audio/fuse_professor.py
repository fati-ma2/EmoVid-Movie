#!/usr/bin/env python3
"""Late fusion of the text, audio and video models on the professor's split.

Four combinations, four fusion rules each:

    combinations   text+audio, text+video, audio+video, text+audio+video
    rules          equal     average of the modalities' probabilities (no parameters)
                   product   geometric mean of the probabilities (no parameters)
                   weighted  weighted average, weights on a 0.1 grid (each >= 0.1), tuned on dev macro F1
                   stacking  multinomial logistic regression on the modalities' log-probabilities
                             (standardised, balanced class weights, L2 lambda 0.01), trained on dev

Protocol
    * Inputs are each model's dev and test probabilities, all trained on the professor's TRAIN films only.
    * Rules with parameters are scored on dev by 5-fold cross-fitting over dev FILMS (fit on 40 films,
      predict the other 10), so every dev score is out-of-sample. Folds: dev films sorted, film i in fold i % 5.
    * For each combination, the rule with the best (out-of-sample) dev macro F1 is frozen, refitted on all
      50 dev films and scored ONCE on test. The final system is the combination with the best dev score.
    * Every interval is a 95% film-level bootstrap (1,000 resamples, seed 20261106). Comparisons with the best
      single model use the same resamples (paired bootstrap).
    * Missing inputs (available = 0) are left out of the average for that clip.

Everything is deterministic (fixed folds, fixed random stream, fixed solver steps): the script was checked
against an independent JavaScript implementation of the same procedure, which gave identical numbers on
synthetic data and on the committed predictions (commit 9306643e). The stacking solver reaches the same
optimum as scikit-learn's LogisticRegression.

Run from the mmemo folder (defaults point at the committed prediction files):

    python scripts/fuse_professor.py
    python scripts/fuse_professor.py --out results_fusion --no-figures

Outputs (in --out, default results_fusion/):
    fusion_report.md          tables for the report and slides
    fusion_report.json        every number
    fusion_dev_selection.csv  4 combinations x 4 rules, out-of-sample dev scores
    fusion_test.csv           the frozen rule of each combination, scored on test
    predictions/fusion_<combination>_<rule>_{dev,test}.csv   (clip_id, p_amusement .. p_sadness, available)
    confusion_final_{dev,test}.png

Needs: numpy (matplotlib for the figures).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLASSES = ("amusement", "awe", "contentment", "excitement", "anger", "disgust", "fear", "sadness")
K = len(CLASSES)
MODS = ("text", "audio", "video")
COMBOS = (("text", "audio"), ("text", "video"), ("audio", "video"), ("text", "audio", "video"))
METHODS = ("equal", "product", "weighted", "stacking")
CFG = {"eps": 1e-6, "folds": 5, "lambda": 0.01, "iters": 1000, "powerIters": 100, "seed": 20261106,
       "nBoot": 1000, "tie": 1e-12}
DEFAULT_INPUTS = {
    "text": "results_text/predictions/text_both_char_C1p0_{split}.csv",
    "audio": "results/predictions/audio_ast_hand_{split}.csv",
    "video": "results/predictions/video_clip_meanstd_{split}.csv",
}
M32 = 0xFFFFFFFF


# ------------------------------------------------------------------ random numbers (same stream as JavaScript)
class Mulberry32:
    def __init__(self, seed: int):
        self.a = seed & M32

    def __call__(self) -> float:
        self.a = (self.a + 0x6D2B79F5) & M32
        t = self.a
        t = ((t ^ (t >> 15)) * (t | 1)) & M32
        t = (t ^ ((t + (((t ^ (t >> 7)) * (t | 61)) & M32)) & M32)) & M32
        return ((t ^ (t >> 14)) & M32) / 4294967296


def seq_sum(values) -> float:
    s = 0.0
    for v in values:
        s += v
    return s


# ------------------------------------------------------------------ labels and metrics
def labels_of(P: np.ndarray, prior: np.ndarray) -> np.ndarray:
    return (P + 1e-9 * prior).argmax(axis=1)  # ties: the more frequent training class


def counts(y, pred, idx=None):
    if idx is not None:
        y, pred = y[idx], pred[idx]
    hit = y == pred
    return (np.bincount(y[hit], minlength=K), np.bincount(pred[~hit], minlength=K),
            np.bincount(y[~hit], minlength=K))


def per_class_f1(c) -> list[float]:
    tp, fp, fn = c
    out = []
    for k in range(K):
        d = 2 * int(tp[k]) + int(fp[k]) + int(fn[k])
        out.append((2 * int(tp[k])) / d if d > 0 else 0)
    return out


def macro_f1(y, pred, idx=None) -> float:
    return seq_sum(per_class_f1(counts(y, pred, idx))) / K


def accuracy(y, pred) -> float:
    return int((y == pred).sum()) / len(y)


def balanced_accuracy(y, pred) -> float:
    tp, _fp, fn = counts(y, pred)
    s, m = 0.0, 0
    for k in range(K):
        sup = int(tp[k]) + int(fn[k])
        if sup > 0:
            s += int(tp[k]) / sup
            m += 1
    return s / m


def percentile(sorted_vals, q: float) -> float:
    pos = (q / 100) * (len(sorted_vals) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def film_groups(films):
    uniq = sorted(set(films))
    pos = {f: i for i, f in enumerate(uniq)}
    rows = [[] for _ in uniq]
    for i, f in enumerate(films):
        rows[pos[f]].append(i)
    return uniq, rows


_PICKS: dict[tuple, list[np.ndarray]] = {}


def bootstrap_picks(films) -> list[np.ndarray]:
    """Film-level resamples; identical for every system scored on the same split (paired)."""
    key = tuple(films)
    if key not in _PICKS:
        uniq, rows = film_groups(films)
        rnd, nf, picks = Mulberry32(CFG["seed"]), len(uniq), []
        for _ in range(CFG["nBoot"]):
            pick = []
            for _j in range(nf):
                pick.extend(rows[int(rnd() * nf)])
            picks.append(np.array(pick, dtype=int))
        _PICKS[key] = picks
    return _PICKS[key]


def boot_f1(y, preds, films):
    picks = bootstrap_picks(films)
    return [[macro_f1(y, p, pick) for pick in picks] for p in preds]


def metrics(y, pred, films) -> dict:
    c = counts(y, pred)
    boot = sorted(boot_f1(y, [pred], films)[0])
    return {"n": int(len(y)), "macro_f1": seq_sum(per_class_f1(c)) / K,
            "macro_f1_ci95": [percentile(boot, 2.5), percentile(boot, 97.5)],
            "accuracy": accuracy(y, pred), "balanced_accuracy": balanced_accuracy(y, pred),
            "per_class_f1": per_class_f1(c)}


def paired_delta(y, pred_a, pred_b, films) -> dict:
    a, b = boot_f1(y, [pred_a, pred_b], films)
    d = [u - v for u, v in zip(a, b)]
    ds = sorted(d)
    return {"delta": macro_f1(y, pred_a) - macro_f1(y, pred_b), "ci95": [percentile(ds, 2.5), percentile(ds, 97.5)],
            "share_le_0": sum(1 for v in d if v <= 0) / len(d)}


# ------------------------------------------------------------------ fusion rules
def softmax_rows(Z: np.ndarray) -> np.ndarray:
    E = np.exp(Z - Z.max(axis=1, keepdims=True))
    s = E[:, 0].copy()
    for k in range(1, K):
        s = s + E[:, k]
    return E / s[:, None]


def fuse_mean(Ps, As, w, rows) -> np.ndarray:
    num, den = np.zeros((len(rows), K)), np.zeros(len(rows))
    for m in range(len(Ps)):
        a = w[m] * As[m][rows]
        den = den + a
        num = num + a[:, None] * Ps[m][rows]
    out = np.full((len(rows), K), 1 / K)
    ok = den > 0
    out[ok] = num[ok] / den[ok, None]
    return out


def fuse_product(Ps, As, w, rows) -> np.ndarray:
    z, den = np.zeros((len(rows), K)), np.zeros(len(rows))
    for m in range(len(Ps)):
        a = w[m] * As[m][rows]
        den = den + a
        z = z + a[:, None] * np.log(np.maximum(Ps[m][rows], CFG["eps"]))
    out = np.full((len(rows), K), 1 / K)
    ok = den > 0
    out[ok] = softmax_rows(z[ok] / den[ok, None])
    return out


def weight_grid(M: int) -> list[list[float]]:
    if M == 2:
        return [[a / 10, (10 - a) / 10] for a in range(1, 10)]
    return [[a / 10, b / 10, (10 - a - b) / 10] for a in range(1, 9) for b in range(1, 10 - a)]


def fit_weights(Ps, As, y, prior, rows) -> list[float]:
    ysub, M = y[rows], len(Ps)
    best, best_f, best_d = None, -1.0, math.inf
    for w in weight_grid(M):
        f = macro_f1(ysub, labels_of(fuse_mean(Ps, As, w, rows), prior))
        d = seq_sum(abs(w[m] - 1 / M) for m in range(M))
        if f > best_f + CFG["tie"] or (abs(f - best_f) <= CFG["tie"] and d < best_d - CFG["tie"]):
            best, best_f, best_d = w, f, d
    return best


def log_features(Ps, rows) -> np.ndarray:
    return np.hstack([np.log(np.maximum(P[rows], CFG["eps"])) for P in Ps])


def standardise(X, mu, sd) -> np.ndarray:
    return np.hstack([(X - mu) / sd, np.ones((len(X), 1))])


def fit_stack(X: np.ndarray, y: np.ndarray) -> dict:
    """Multinomial logistic regression, balanced class weights, L2 on the weights (not the intercepts),
    Nesterov-accelerated gradient descent with a fixed number of steps (deterministic)."""
    n, D = X.shape
    lam_reg = CFG["lambda"]
    mu = np.cumsum(X, axis=0)[-1] / n
    sd = np.sqrt(np.cumsum((X - mu) * (X - mu), axis=0)[-1] / n)
    sd[sd < 1e-8] = 1.0
    Xs = standardise(X, mu, sd)
    D1 = D + 1
    cnt = np.bincount(y, minlength=K)
    present = int((cnt > 0).sum())
    w = n / (present * cnt[y])
    sw = float(np.cumsum(w)[-1])
    vec, lam = np.full(D1, 1 / math.sqrt(D1)), 0.0
    for _ in range(CFG["powerIters"]):
        u = Xs.T @ (w * (Xs @ vec) / sw)
        lam = math.sqrt(float(u @ u))
        vec = u / lam
    step = 1 / (0.5 * lam + lam_reg)
    Y = np.eye(K)[y]

    def grad(th):
        R = w[:, None] * (softmax_rows(Xs @ th) - Y) / sw
        G = Xs.T @ R
        G[:D] += lam_reg * th[:D]
        return G

    theta, yk = np.zeros((D1, K)), np.zeros((D1, K))
    for t in range(CFG["iters"]):
        nxt = yk - step * grad(yk)
        yk = nxt + (t / (t + 3)) * (nxt - theta)
        theta = nxt
    return {"mu": mu, "sd": sd, "theta": theta}


def predict_stack(model, X) -> np.ndarray:
    return softmax_rows(standardise(X, model["mu"], model["sd"]) @ model["theta"])


def apply_method(method, Ps, As, y, prior, fit_rows, pred_rows):
    M = len(Ps)
    if method == "equal":
        return fuse_mean(Ps, As, [1] * M, pred_rows), None
    if method == "product":
        return fuse_product(Ps, As, [1] * M, pred_rows), None
    if method == "weighted":
        wt = fit_weights(Ps, As, y, prior, fit_rows)
        return fuse_mean(Ps, As, wt, pred_rows), wt
    model = fit_stack(log_features(Ps, fit_rows), y[fit_rows])
    return predict_stack(model, log_features(Ps, pred_rows)), None


def dev_predictions(method, Ps, As, y, prior, films):
    """Out-of-sample dev probabilities: parameter-free rules directly, the others cross-fitted by film."""
    allr = np.arange(len(y))
    if method in ("equal", "product"):
        return apply_method(method, Ps, As, y, prior, allr, allr)[0], None
    uniq, _ = film_groups(films)
    fold_of = {f: i % CFG["folds"] for i, f in enumerate(uniq)}
    fold = np.array([fold_of[f] for f in films])
    out, fold_params = np.zeros((len(y), K)), []
    for k in range(CFG["folds"]):
        fit_rows, pred_rows = allr[fold != k], allr[fold == k]
        P, params = apply_method(method, Ps, As, y, prior, fit_rows, pred_rows)
        out[pred_rows] = P
        fold_params.append(params)
    return out, fold_params


# ------------------------------------------------------------------ main procedure
def run(manifest: list[dict], preds: dict) -> tuple[dict, dict]:
    prior = np.zeros(K)
    n_train = 0
    for r in manifest:
        if r["split"] == "train":
            prior[CLASSES.index(r["emotion"])] += 1
            n_train += 1
    prior = prior / n_train
    data = {}
    for s in ("dev", "test"):
        rows = sorted((r for r in manifest if r["split"] == s), key=lambda r: r["clip_id"])
        d = {"ids": [r["clip_id"] for r in rows], "films": [r["film_id"] for r in rows],
             "y": np.array([CLASSES.index(r["emotion"]) for r in rows]), "P": {}, "A": {}}
        for m in MODS:
            by = {r["clip_id"]: r for r in preds[m][s]}
            miss = [i for i in d["ids"] if i not in by]
            if miss:
                sys.exit(f"ERROR: {m} {s}: {len(miss)} clips of the split missing from the predictions, e.g. {miss[0]}")
            if len(preds[m][s]) != len(d["ids"]):
                sys.exit(f"ERROR: {m} {s}: {len(preds[m][s])} rows for {len(d['ids'])} clips (wrong split?)")
            d["P"][m] = np.array([by[i]["p"] for i in d["ids"]], dtype=float)
            d["A"][m] = np.array([by[i]["available"] for i in d["ids"]], dtype=int)
        data[s] = d

    report = {"config": CFG, "classes": list(CLASSES), "prior_train": prior.tolist(), "data": {}, "single": {},
              "combos": {}, "frozen": {}, "final": None}
    for s, d in data.items():
        report["data"][s] = {"clips": len(d["ids"]), "films": len(set(d["films"])),
                             "available": {m: int(d["A"][m].sum()) for m in MODS}}
    lab = {"dev": {}, "test": {}}
    for m in MODS:
        report["single"][m] = {}
        for s in ("dev", "test"):
            lab[s][m] = labels_of(data[s]["P"][m], prior)
            report["single"][m][s] = metrics(data[s]["y"], lab[s][m], data[s]["films"])
    ref = sorted(MODS, key=lambda m: -report["single"][m]["dev"]["macro_f1"])[0]
    report["reference_single"] = ref
    dv, te = data["dev"], data["test"]
    fused = {}
    for combo in COMBOS:
        name = "+".join(combo)
        Ps, As = [dv["P"][m] for m in combo], [dv["A"][m] for m in combo]
        entry = {"modalities": list(combo), "methods": {}}
        best, best_f = None, -1.0
        for method in METHODS:
            P, fold_params = dev_predictions(method, Ps, As, dv["y"], prior, dv["films"])
            lb = labels_of(P, prior)
            e = {"dev": metrics(dv["y"], lb, dv["films"]), "dev_vs_ref": paired_delta(dv["y"], lb, lab["dev"][ref], dv["films"])}
            if method == "weighted":
                e["fold_weights"] = fold_params
                e["weights_all_dev"] = fit_weights(Ps, As, dv["y"], prior, np.arange(len(dv["y"])))
            entry["methods"][method] = e
            fused[(name, method, "dev")] = P
            if e["dev"]["macro_f1"] > best_f + CFG["tie"]:
                best, best_f = method, e["dev"]["macro_f1"]
        L = np.stack([lab["dev"][m] for m in combo])
        entry["dev_oracle"] = float((L == dv["y"]).any(axis=0).mean())
        entry["dev_all_agree"] = float((L == L[0]).all(axis=0).mean())
        entry["chosen"] = best
        report["frozen"][name] = best
        report["combos"][name] = entry

    # Test: the frozen rule of each combination, fitted on all dev films, scored once.
    nd = len(dv["y"])
    y_all = np.concatenate([dv["y"], te["y"]])
    for combo in COMBOS:
        name = "+".join(combo)
        entry, method = report["combos"][name], report["combos"][name]["chosen"]
        Ps = [np.vstack([dv["P"][m], te["P"][m]]) for m in combo]
        As = [np.concatenate([dv["A"][m], te["A"][m]]) for m in combo]
        P, params = apply_method(method, Ps, As, y_all, prior, np.arange(nd), np.arange(nd, nd + len(te["y"])))
        lb = labels_of(P, prior)
        entry["test"] = metrics(te["y"], lb, te["films"])
        entry["test_vs_ref"] = paired_delta(te["y"], lb, lab["test"][ref], te["films"])
        entry["test_params"] = params
        fused[(name, method, "test")] = P

    fin, fin_f = None, -1.0
    for combo in COMBOS:
        nm = "+".join(combo)
        f = report["combos"][nm]["methods"][report["combos"][nm]["chosen"]]["dev"]["macro_f1"]
        if f > fin_f + CFG["tie"]:
            fin, fin_f = nm, f
    report["final"] = {"combination": fin, "method": report["combos"][fin]["chosen"]}
    extras = {"data": data, "fused": fused, "prior": prior}
    return report, extras


# ------------------------------------------------------------------ files
def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        sys.exit(f"ERROR: file not found: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_predictions(path: Path) -> list[dict]:
    rows = read_csv(path)
    need = {"clip_id", "available", *(f"p_{c}" for c in CLASSES)}
    if rows and not need <= set(rows[0]):
        sys.exit(f"ERROR: {path} lacks columns {sorted(need - set(rows[0]))}")
    out = []
    for r in rows:
        p = [float(r[f"p_{c}"]) for c in CLASSES]
        if abs(sum(p) - 1) > 1e-4:
            sys.exit(f"ERROR: {path}: probabilities of {r['clip_id']} sum to {sum(p):.6f}")
        out.append({"clip_id": r["clip_id"], "p": p, "available": int(float(r["available"]))})
    return out


def fmt_ci(m: dict) -> str:
    return f"{m['macro_f1']:.3f} ({m['macro_f1_ci95'][0]:.3f} to {m['macro_f1_ci95'][1]:.3f})"


def fmt_delta(d: dict) -> str:
    return f"{d['delta']:+.3f} ({d['ci95'][0]:+.3f} to {d['ci95'][1]:+.3f}); share <= 0: {d['share_le_0']:.0%}"


def fmt_w(combo, w) -> str:
    return ", ".join(f"{m} {v:.1f}" for m, v in zip(combo, w)) if w else ""


def write_outputs(report: dict, extras: dict, out: Path, figures: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions").mkdir(exist_ok=True)
    (out / "fusion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    ref = report["reference_single"]

    with (out / "fusion_dev_selection.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["combination", "rule", "dev_macro_f1", "ci95_low", "ci95_high", "dev_accuracy",
                    "dev_balanced_accuracy", f"delta_vs_{ref}", "delta_ci95_low", "delta_ci95_high",
                    "weights_all_dev", "chosen"])
        for name, e in report["combos"].items():
            for method, r in e["methods"].items():
                m, d = r["dev"], r["dev_vs_ref"]
                w.writerow([name, method, round(m["macro_f1"], 4), round(m["macro_f1_ci95"][0], 4),
                            round(m["macro_f1_ci95"][1], 4), round(m["accuracy"], 4), round(m["balanced_accuracy"], 4),
                            round(d["delta"], 4), round(d["ci95"][0], 4), round(d["ci95"][1], 4),
                            fmt_w(e["modalities"], r.get("weights_all_dev")), int(method == e["chosen"])])
    with (out / "fusion_test.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["combination", "rule", "test_macro_f1", "ci95_low", "ci95_high", "test_accuracy",
                    "test_balanced_accuracy", f"delta_vs_{ref}", "delta_ci95_low", "delta_ci95_high", "weights", "final"])
        for name, e in report["combos"].items():
            m, d = e["test"], e["test_vs_ref"]
            w.writerow([name, e["chosen"], round(m["macro_f1"], 4), round(m["macro_f1_ci95"][0], 4),
                        round(m["macro_f1_ci95"][1], 4), round(m["accuracy"], 4), round(m["balanced_accuracy"], 4),
                        round(d["delta"], 4), round(d["ci95"][0], 4), round(d["ci95"][1], 4),
                        fmt_w(e["modalities"], e["test_params"]), int(name == report["final"]["combination"])])

    for (name, method, split), P in extras["fused"].items():
        if method != report["combos"][name]["chosen"]:
            continue
        ids = extras["data"][split]["ids"]
        path = out / "predictions" / f"fusion_{name.replace('+', '_')}_{method}_{split}.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["clip_id", *(f"p_{c}" for c in CLASSES), "available"])
            for i, row in zip(ids, P):
                w.writerow([i, *(f"{v:.8f}" for v in row), 1])

    lines = ["# Fusion on the professor's split", "",
             f"Dev {report['data']['dev']['clips']} clips / {report['data']['dev']['films']} films; "
             f"test {report['data']['test']['clips']} clips / {report['data']['test']['films']} films. "
             "Macro F1 with 95% film-level bootstrap intervals.", "",
             "## Single models", "", "| model | dev macro F1 | test macro F1 | test accuracy |", "|---|---|---|---|"]
    for m, r in report["single"].items():
        lines.append(f"| {m} | {fmt_ci(r['dev'])} | {fmt_ci(r['test'])} | {r['test']['accuracy']:.1%} |")
    lines += ["", "## Dev: every rule (out-of-sample; weighted and stacking cross-fitted over 5 folds of films)", "",
              "| combination | equal | product | weighted | stacking | chosen |", "|---|---|---|---|---|---|"]
    for name, e in report["combos"].items():
        cells = [f"{e['methods'][mt]['dev']['macro_f1']:.3f}" for mt in METHODS]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {e['chosen']} |")
    lines += ["", f"## Test: the rule chosen on dev for each combination (difference with {ref} alone, paired)", "",
              "| combination | rule | test macro F1 | test accuracy | difference with " + ref + " |", "|---|---|---|---|---|"]
    for name, e in report["combos"].items():
        mark = " **(final)**" if name == report["final"]["combination"] else ""
        wtxt = f" ({fmt_w(e['modalities'], e['test_params'])})" if e["test_params"] else ""
        lines.append(f"| {name}{mark} | {e['chosen']}{wtxt} | {fmt_ci(e['test'])} | {e['test']['accuracy']:.1%} | "
                     f"{fmt_delta(e['test_vs_ref'])} |")
    lines += ["", "## Complementarity on dev", "", "| combination | oracle (at least one model right) | all models agree |",
              "|---|---|---|"]
    for name, e in report["combos"].items():
        lines.append(f"| {name} | {e['dev_oracle']:.1%} | {e['dev_all_agree']:.1%} |")
    fin = report["combos"][report["final"]["combination"]]
    lines += ["", f"Final system (best dev score): {report['final']['combination']}, {report['final']['method']} rule.",
              "", "Per-class F1 of the final system:", "", "| class | dev | test |", "|---|---|---|"]
    fin_dev = fin["methods"][fin["chosen"]]["dev"]["per_class_f1"]
    for c, a, b in zip(CLASSES, fin_dev, fin["test"]["per_class_f1"]):
        lines.append(f"| {c} | {a:.3f} | {b:.3f} |")
    (out / "fusion_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if figures:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from text_emotion import plot_confusion  # same figure style as the text system
        except Exception as exc:  # noqa: BLE001
            print(f"Figures skipped ({exc})")
            return
        nm, mt = report["final"]["combination"], report["final"]["method"]
        for split in ("dev", "test"):
            y = extras["data"][split]["y"]
            pred = labels_of(extras["fused"][(nm, mt, split)], extras["prior"])
            cm = np.zeros((K, K), dtype=int)
            np.add.at(cm, (y, pred), 1)
            plot_confusion(cm.tolist(), f"Fusion {nm} ({mt}), {split}", out / f"confusion_final_{split}.png")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "splits" / "manifest.csv",
                    help="clip_id, film_id, emotion, split (the professor's split)")
    for m in MODS:
        ap.add_argument(f"--{m}", default=str(PROJECT_ROOT / DEFAULT_INPUTS[m]),
                        help=f"{m} predictions, with {{split}} in place of dev/test (default: %(default)s)")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "results_fusion")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    manifest = read_csv(args.manifest)
    counts_split = {s: sum(r["split"] == s for r in manifest) for s in ("train", "dev", "test")}
    print(f"Manifest: {counts_split}")
    preds = {m: {s: read_predictions(Path(getattr(args, m).format(split=s))) for s in ("dev", "test")} for m in MODS}
    report, extras = run(manifest, preds)
    write_outputs(report, extras, args.out, not args.no_figures)
    print((args.out / "fusion_report.md").read_text(encoding="utf-8"))
    print(f"Wrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
