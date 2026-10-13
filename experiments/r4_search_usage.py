"""Round 4 groundwork: what does the in-hand search actually do on the E0 clips?

While the object is held, ObjectState searches on every frame for "was it put down?" (DINOv2 scan + an outline per
candidate spot + look and detail checks, ~150-200 ms). This replays the E0 clips with the default settings and, for
each held frame, records: how many spots were outlined, why each was rejected, whether one was accepted (put down /
still in the hand / maybe), and whether the frame's answer came from the search or from the tracker.

Run:  python experiments/r4_search_usage.py
Writes experiments/output/e5/r4_search_usage.json and prints a summary.
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

import cv2

from compass.appearance import Appearance
from compass.body import ClothesSegmenter
from compass.hands import HandDetector
from compass.holder import Finder, ObjectState, Segmenter
from compass.models import device, video_predictor
from compass.tracker import StreamingTracker
from e0_real_clips import CLIPS, FPS

STRIDE = 5


def main():
    dev = device()
    obj = ObjectState(StreamingTracker(video_predictor(dev), window=True, encoder="coreml"), Appearance(dev),
                      Finder(dev), Segmenter(dev), hands=HandDetector(), clothes=ClothesSegmenter())
    rows = []
    for name, clip in CLIPS.items():
        cap = cv2.VideoCapture(str(ROOT / "clips" / f"{name}.mp4"))
        fps = cap.get(cv2.CAP_PROP_FPS) or FPS
        stride = max(1, round(STRIDE * fps / FPS))
        i = k = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if i >= clip["start"] and (i - clip["start"]) % stride == 0:
                t_ms = i / fps * 1000
                was = obj.state if k else None
                step = obj.start(img, clip["box"], t_ms=t_ms) if k == 0 else obj.update(img, t_ms=t_ms)
                if was == "held":
                    cands = step.debug.get("search_candidates") or []
                    on_hand = 0
                    for c in cands:
                        x, y = c["at"]
                        on_hand += any(hd.mask[y, x] for hd in step.hands)
                    rows.append({"clip": name, "frame": i, "searched": bool(step.debug.get("hand_off_searched")),
                                 "n": len(cands), "on_hand": on_hand,
                                 "rejected": Counter(c.get("rejected", "passed") for c in cands),
                                 "accepted": next((c["accepted"] for c in cands if "accepted" in c), None),
                                 "events": [e for _, e in step.events], "state_after": step.state,
                                 "note": step.note, "hands": len(step.hands)})
                k += 1
            i += 1
        print(name, "held frames:", sum(r["clip"] == name for r in rows), flush=True)
    n = len(rows)
    total = Counter()
    for r in rows:
        total.update(r["rejected"])
    outlined = sum(r["n"] for r in rows)
    ev = Counter(e.split(":")[0].split("(")[0].strip() for r in rows for e in r["events"])
    summary = {
        "held_frames": n, "searched_frames": sum(r["searched"] for r in rows),
        "outlines_per_frame": round(outlined / max(n, 1), 2),
        "outlines_on_a_hand_per_frame": round(sum(r["on_hand"] for r in rows) / max(n, 1), 2),
        "candidate_outcomes": dict(total.most_common()),
        "frames_with_an_accepted_candidate": sum(r["accepted"] is not None for r in rows),
        "accepted_kinds": dict(Counter(r["accepted"] for r in rows if r["accepted"]).most_common()),
        "events_in_held_frames": dict(ev.most_common()),
    }
    print(json.dumps(summary, indent=1))
    out = ROOT / "experiments" / "output" / "e5"
    out.mkdir(parents=True, exist_ok=True)
    (out / "r4_search_usage.json").write_text(json.dumps({"summary": summary, "frames": rows}, indent=1, default=dict))


if __name__ == "__main__":
    main()
