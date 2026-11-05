#!/usr/bin/env python3
"""Confusion matrices of the Text + Audio fusion (dev and test).

Uses exactly the procedure of fuse_professor.py: the rule chosen on dev for text+audio
(out-of-sample, 5 folds of dev films), refitted on all dev films and applied once to test.
Only the text and audio predictions are needed.

Run from the mmemo folder (same defaults as fuse_professor.py):

    python scripts/confusion_text_audio.py
    python scripts/confusion_text_audio.py --out results_fusion

Outputs (in --out, default results_fusion/):
    confusion_text_audio_dev.png, confusion_text_audio_test.png   (% of each true class)
    confusion_text_audio_dev.csv, confusion_text_audio_test.csv   (raw counts, rows = true class)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (_HERE.parent.parent / "Text", _HERE):  # GitHub layout (Text/ two levels up) or mmemo/scripts/
    if _p.is_dir():
        sys.path.insert(0, str(_p))
import fuse_professor as fp  # noqa: E402
from text_emotion import plot_confusion  # noqa: E402  (same figure style as the other matrices)

COMBO = ("text", "audio")
NAME = "+".join(COMBO)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=fp.PROJECT_ROOT / "splits" / "manifest.csv")
    for m in COMBO:
        ap.add_argument(f"--{m}", default=str(fp.PROJECT_ROOT / fp.DEFAULT_INPUTS[m]),
                        help=f"{m} predictions, with {{split}} in place of dev/test (default: %(default)s)")
    ap.add_argument("--out", type=Path, default=fp.PROJECT_ROOT / "results_fusion")
    args = ap.parse_args(argv)

    # Restrict the fusion procedure to text + audio (each combination is computed independently,
    # so the text+audio numbers are identical to those of the full fuse_professor.py run).
    fp.MODS = COMBO
    fp.COMBOS = (COMBO,)

    manifest = fp.read_csv(args.manifest)
    preds = {m: {s: fp.read_predictions(Path(getattr(args, m).format(split=s))) for s in ("dev", "test")}
             for m in COMBO}
    report, extras = fp.run(manifest, preds)
    entry = report["combos"][NAME]
    rule = entry["chosen"]
    wtxt = f", {fp.fmt_w(COMBO, entry['test_params'])}" if entry["test_params"] else ""

    args.out.mkdir(parents=True, exist_ok=True)
    for split in ("dev", "test"):
        y = extras["data"][split]["y"]
        pred = fp.labels_of(extras["fused"][(NAME, rule, split)], extras["prior"])
        cm = np.zeros((fp.K, fp.K), dtype=int)
        np.add.at(cm, (y, pred), 1)
        f1 = entry["methods"][rule]["dev"]["macro_f1"] if split == "dev" else entry["test"]["macro_f1"]

        with (args.out / f"confusion_text_audio_{split}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["true \\ predicted", *fp.CLASSES])
            for c, row in zip(fp.CLASSES, cm):
                w.writerow([c, *row.tolist()])
        plot_confusion(cm.tolist(), f"Fusion text+audio ({rule}{wtxt}), {split}: macro F1 {f1:.3f}",
                       args.out / f"confusion_text_audio_{split}.png")
        print(f"{split}: rule {rule}{wtxt}, macro F1 {f1:.3f}, accuracy {np.trace(cm) / cm.sum():.1%}")
        print("\n".join(f"  {c:<12} " + " ".join(f"{v:4d}" for v in row) for c, row in zip(fp.CLASSES, cm)))

    print(f"Wrote confusion_text_audio_{{dev,test}}.png/.csv in {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
