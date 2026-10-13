"""Round 4 groundwork, part 2: which cheap tests, done on a search spot *before* outlining it, would drop the spots
that come to nothing, without dropping one that matters?

Replays the E0 clips (default settings). For every spot the in-hand search outlines, records cheap features known
before outlining, and what became of the outline:
  lookalike   a known lookalike spot lies within 25 px of it
  far         it is more than one object size away from where the hand is or was
  unchanged   the area around it looks as it did when the hand took the object (as the "already lying there" check)
  on_hand     it lies on a detected hand
  outcome     accepted / passed / rejected (reason) / dropped (wrong size, or the same object as an earlier spot)

Run:  python experiments/r4_spot_filters.py
Writes experiments/output/e5/r4_spot_filters.json and prints, per test, how many outlines it would save and how many
useful spots it would lose.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

import cv2
import numpy as np

from compass.appearance import Appearance
from compass.body import ClothesSegmenter
from compass.hands import HandDetector
from compass.holder import Finder, ObjectState, Segmenter, _dilate, mask_iou
from compass.models import device, video_predictor
from compass.tracker import StreamingTracker
from e0_real_clips import CLIPS, FPS

STRIDE = 5


def patch_unchanged(img, before, x, y, r=24):
    """Does the area around (x, y) look as it did in `before`? (smoothed Lab, as ObjectState._already_there)"""
    if before is None or before.shape != img.shape:
        return False
    h, w = img.shape[:2]
    y0, y1, x0, x1 = max(int(y) - r, 0), min(int(y) + r + 1, h), max(int(x) - r, 0), min(int(x) + r + 1, w)
    a = cv2.cvtColor(cv2.GaussianBlur(img[y0:y1, x0:x1], (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(cv2.GaussianBlur(before[y0:y1, x0:x1], (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    return float((np.linalg.norm(a - b, axis=-1) < 18).mean()) >= 0.6


def main():
    dev = device()
    obj = ObjectState(StreamingTracker(video_predictor(dev), window=True, encoder="coreml"), Appearance(dev),
                      Finder(dev), Segmenter(dev), hands=HandDetector(), clothes=ClothesSegmenter())
    seg_calls = []
    orig_seg = obj.seg.segment_points

    def seg_spy(img, points, window=None):
        masks = orig_seg(img, points, window=window)
        seg_calls.append((list(points), masks))
        return masks

    obj.seg.segment_points = seg_spy
    spots = []
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
                held = k > 0 and obj.state == "held"
                if held:  # what is known before this frame's search
                    before, others = getattr(obj, "take_img", None), list(obj.known_others)
                    size = float(np.sqrt(np.median(obj.gate.areas)))
                seg_calls.clear()
                step = obj.start(img, clip["box"], t_ms=t_ms) if k == 0 else obj.update(img, t_ms=t_ms)
                if held and seg_calls:
                    points, masks = seg_calls[-1]
                    near = obj._reach() if obj.holders else None  # as the search saw it (with this frame's hand)
                    cands = step.debug.get("search_candidates") or []
                    dist = cv2.distanceTransform((~near).astype(np.uint8), cv2.DIST_L2, 3) if near is not None else None
                    kept = []
                    for (x, y), m in zip(points, masks):
                        xi, yi = min(int(x), img.shape[1] - 1), min(int(y), img.shape[0] - 1)
                        if any(kk[yi, xi] or mask_iou(kk, m) > 0.5 for kk in kept):
                            outcome = "dropped: same object"
                        elif not 0.25 * obj.finder.ref_area <= m.sum() <= 3 * obj.finder.ref_area:
                            outcome = "dropped: size"
                        else:
                            kept.append(m)
                            ys, xs = np.nonzero(m)
                            c = next((c for c in cands if c["at"] == [int(xs.mean()), int(ys.mean())]), {})
                            outcome = ("accepted" if "accepted" in c else c.get("rejected", "passed"))
                        spots.append({"clip": name, "frame": i, "outcome": outcome,
                                      "lookalike": any(abs(x - ox) + abs(y - oy) < 25 for ox, oy in others),
                                      "far": bool(dist is not None and dist[yi, xi] > size),
                                      "unchanged": patch_unchanged(img, before, x, y),
                                      "on_hand": any(hd.mask[yi, xi] for hd in step.hands)})
                k += 1
            i += 1
        print(name, "spots so far:", len(spots), flush=True)
    useful = {"accepted", "passed"}
    print(f"\n{len(spots)} outlined spots; useful (accepted or passed the checks): {sum(s['outcome'] in useful for s in spots)}")
    for test in ("lookalike", "far", "unchanged", "on_hand"):
        hit = [s for s in spots if s[test]]
        print(f"  skip if {test:10s}: saves {len(hit):4d} outlines, would lose {sum(s['outcome'] in useful for s in hit)} useful "
              f"({sum(s['outcome'] == 'accepted' for s in hit)} accepted)")
    combo = [s for s in spots if s["lookalike"] or s["far"] or s["unchanged"]]
    print(f"  skip if lookalike/far/unchanged: saves {len(combo)}, would lose {sum(s['outcome'] in useful for s in combo)} useful "
          f"({sum(s['outcome'] == 'accepted' for s in combo)} accepted)")
    out = ROOT / "experiments" / "output" / "e5"
    (out / "r4_spot_filters.json").write_text(json.dumps(spots, indent=1))


if __name__ == "__main__":
    main()
