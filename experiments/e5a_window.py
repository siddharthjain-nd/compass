"""E5a: does tracking only around the object (compass.window) give the same outlines as the whole frame, and how much
faster is it? Tracker only (no holder logic), on the E0 clips, every 5th frame.

For each clip both trackers start from the same box and track the same frames; per frame we compare the two masks
(IoU, both empty counts as agreeing) and time each tracker step.

Run:  python experiments/e5a_window.py [--margin 0.5]
Writes experiments/output/e5/window_tracker_only.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

import cv2
import numpy as np
import torch

from compass.models import device, video_predictor
from compass.tracker import StreamingTracker
from e0_real_clips import CLIPS

STRIDE = 5


def timed(fn):
    t = time.perf_counter()
    out = fn()
    torch.mps.synchronize()
    return out, (time.perf_counter() - t) * 1000


def iou(a, b):
    union = (a | b).sum()
    return 1.0 if union == 0 else float((a & b).sum() / union)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=0.5)
    args = ap.parse_args()
    dev = device()
    full = StreamingTracker(video_predictor(dev))
    win = StreamingTracker(video_predictor(dev), window=True)
    win.window.margin = args.margin
    report = {"margin": args.margin, "clips": {}}
    for name, clip in CLIPS.items():
        cap = cv2.VideoCapture(str(ROOT / "clips" / f"{name}.mp4"))
        stride = max(1, round(STRIDE * (cap.get(cv2.CAP_PROP_FPS) or 15) / 15))  # 3 pictures/s for 15 and 30 fps
        i = k = 0
        rows = []
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if i >= clip["start"] and (i - clip["start"]) % stride == 0:
                if k == 0:
                    a, b = full.start(img, clip["box"])[0], win.start(img, clip["box"])[0]
                else:
                    (a, _), ta = timed(lambda: full.track(img))
                    (b, _), tb = timed(lambda: win.track(img))
                    cell = win.window.cell
                    rows.append({"frame": i, "iou": round(iou(a, b), 3), "full_ms": round(ta, 1), "win_ms": round(tb, 1),
                                 "window": cell, "area_full": int(a.sum()), "area_win": int(b.sum())})
                k += 1
            i += 1
        ious = [r["iou"] for r in rows]
        w_rows = [r for r in rows if r["window"] is not None]
        summary = {"frames": len(rows), "mean_iou": round(float(np.mean(ious)), 3),
                   "share_iou_below_0.8": round(float(np.mean([x < 0.8 for x in ious])), 3),
                   "share_windowed": round(len(w_rows) / max(len(rows), 1), 2),
                   "median_full_ms": round(float(np.median([r["full_ms"] for r in rows])), 1),
                   "median_win_ms": round(float(np.median([r["win_ms"] for r in rows])), 1),
                   "median_win_ms_when_windowed": round(float(np.median([r["win_ms"] for r in w_rows])), 1) if w_rows else None,
                   "median_window_cells": float(np.median([r["window"][2] for r in w_rows])) if w_rows else None}
        report["clips"][name] = {"summary": summary, "frames": rows}
        print(name, summary)
    allr = [r for c in report["clips"].values() for r in c["frames"]]
    report["all"] = {"frames": len(allr), "mean_iou": round(float(np.mean([r["iou"] for r in allr])), 3),
                     "share_iou_below_0.8": round(float(np.mean([r["iou"] < 0.8 for r in allr])), 3),
                     "median_full_ms": round(float(np.median([r["full_ms"] for r in allr])), 1),
                     "median_win_ms": round(float(np.median([r["win_ms"] for r in allr])), 1)}
    print("ALL", report["all"])
    out = ROOT / "experiments" / "output" / "e5"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"window_tracker_only_m{args.margin}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
