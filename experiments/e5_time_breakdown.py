"""E5 baseline: where the time goes per frame in the full system (one object), measured on the E0 clips.

Replays the six E0 clips exactly like experiments/e0_real_clips.py (every 5th frame) and times each part of
ObjectState.update(). Parts are timed at the lowest level so nothing is counted twice:
  tracker         EdgeTAM (follow the object to the next frame; restarts)
  appearance      DINOv2 at 224 px: "does this outline still look like my object?" (every call)
  hands           MediaPipe hand detector (since step 1 it runs in parallel with the tracker: shown, not summed)
  explainer       blocked / turned / closer... (ORB details, colours; CPU)
  finder scan     DINOv2 at 448 px over the whole picture: "where could my object be?"
  finder outlines EdgeTAM image model: outline each candidate spot
  clothes         MediaPipe clothes segmenter (pocket check)
  other           the rest of the logic (mask maths, rules)
plus the live app's drawing step, timed separately.

Run:  python experiments/e5_time_breakdown.py [--no-skip]
Writes experiments/output/e5/time_breakdown.json (or time_breakdown_noskip.json: tracker on every frame)
"""
import json
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

import cv2
import numpy as np
import torch

from compass.appearance import Appearance
from compass.body import ClothesSegmenter
from compass.explain import Explainer
from compass.hands import HandDetector
from compass.holder import Finder, ObjectState, Segmenter
from compass.live import draw
from compass.models import device, video_predictor
from compass.tracker import StreamingTracker
from e0_real_clips import CLIPS, FPS

STRIDE = 5
frame_times = defaultdict(float)  # part -> ms in the current frame
frame_calls = defaultdict(int)


def timed(cls, method, part):
    original = getattr(cls, method)

    def wrapper(*a, **k):
        t = time.perf_counter()
        out = original(*a, **k)
        # sync the GPU only from the main thread: the hand detector runs on a worker thread (in parallel with the
        # tracker), and a GPU sync from there races the tracker's GPU work (Metal aborts)
        if torch.backends.mps.is_available() and threading.current_thread() is threading.main_thread():
            torch.mps.synchronize()
        frame_times[part] += (time.perf_counter() - t) * 1000
        frame_calls[part] += 1
        return out

    setattr(cls, method, wrapper)


def state_name(step):
    if step.state == "in_pocket":
        return "in pocket"
    if step.state == "held":
        return "in hand"
    return "visible" if step.mask.any() else step.state


def main():
    for cls, method, part in [(StreamingTracker, "track", "tracker"), (StreamingTracker, "reinit", "tracker"),
                              (Appearance, "_embed_crops", "appearance"), (HandDetector, "detect", "hands"),
                              (Explainer, "explain", "explainer"), (Explainer, "remember", "explainer"),
                              (Explainer, "detail_matches", "explainer"), (Finder, "tokens", "finder scan"),
                              (Segmenter, "segment_points", "finder outlines"),
                              (Segmenter, "segment_point", "finder outlines"), (ClothesSegmenter, "clothes", "clothes")]:
        timed(cls, method, part)

    dev = device()
    skip = "--no-skip" not in sys.argv
    window = "--window" in sys.argv
    obj = ObjectState(StreamingTracker(video_predictor(dev), window=window), Appearance(dev), Finder(dev), Segmenter(dev),
                      hands=HandDetector(), clothes=ClothesSegmenter(), skip_when_still=skip,
                      precheck="--precheck" in sys.argv)
    rows = []
    only = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--clip=")]
    for name, clip in CLIPS.items():
        if only and name not in only:
            continue
        cap = cv2.VideoCapture(str(ROOT / "clips" / f"{name}.mp4"))
        fps = cap.get(cv2.CAP_PROP_FPS) or FPS
        stride = max(1, round(STRIDE * fps / FPS))  # 3 pictures per second of video for 15 and 30 fps clips
        i = k = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if i >= clip["start"] and (i - clip["start"]) % stride == 0:
                t_ms = i / fps * 1000
                if k == 0:
                    obj.start(img, clip["box"], t_ms=t_ms)
                else:
                    frame_times.clear()
                    frame_calls.clear()
                    t = time.perf_counter()
                    step = obj.update(img, t_ms=t_ms)
                    total = (time.perf_counter() - t) * 1000
                    t = time.perf_counter()
                    draw(img, step, "", "", "", False)
                    draw_ms = (time.perf_counter() - t) * 1000
                    parts = dict(frame_times)
                    # hands run in parallel with the tracker (worker thread), so they are not part of the sum
                    parts["other"] = total - sum(v for p, v in parts.items() if p != "hands")
                    rows.append({"clip": name, "frame": i, "state": state_name(step), "total": total, "draw": draw_ms,
                                 "skipped": bool(step.debug.get("skipped")),
                                 "parts": parts, "calls": dict(frame_calls)})
                k += 1
            i += 1
        print(f"{name}: {k} frames")

    names = ["tracker", "appearance", "hands", "explainer", "finder scan", "finder outlines", "clothes", "other"]

    def summary(rs):
        n = len(rs)
        return {"frames": n, "total_ms": round(sum(r["total"] for r in rs) / n, 1),
                "fps": round(1000 / (sum(r["total"] for r in rs) / n), 2),
                "draw_ms": round(sum(r["draw"] for r in rs) / n, 1),
                "parts_ms": {p: round(sum(r["parts"].get(p, 0) for r in rs) / n, 1) for p in names},
                "share_of_frames_with": {p: round(sum(p in r["calls"] for r in rs) / n, 2)
                                         for p in ("finder scan", "finder outlines", "clothes")},
                "appearance_calls_per_frame": round(sum(r["calls"].get("appearance", 0) for r in rs) / n, 2),
                "share_skipped": round(sum(r.get("skipped", False) for r in rs) / n, 2),
                "median_total_ms": round(float(np.median([r["total"] for r in rs])), 1),
                "p90_total_ms": round(float(np.percentile([r["total"] for r in rs], 90)), 1)}

    result = {"all": summary(rows)}
    for s in sorted({r["state"] for r in rows}):
        result[f"state: {s}"] = summary([r for r in rows if r["state"] == s])
    print(json.dumps(result, indent=1))
    out = ROOT / "experiments" / "output" / "e5"
    out.mkdir(parents=True, exist_ok=True)
    tag = ('_clips' if only else '') + ('' if skip else '_noskip') + ('_window' if window else '') + ('_precheck' if '--precheck' in sys.argv else '')
    (out / f"time_breakdown{tag}.json").write_text(json.dumps({"summary": result, "frames": rows}, indent=1))


if __name__ == "__main__":
    main()
