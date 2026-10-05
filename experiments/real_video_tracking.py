"""Track several real objects in a real video with EdgeTAM and render the result to watch.

Default clip: EdgeTAM's 02_cups.mp4. A yellow ball goes under one of three identical red cups,
the cups are shuffled, and the ball is revealed at the end. Each object gets one box on frame 0;
after that the tracker is on its own. There is no ground truth here: you check by watching
whether each label stays on the same physical object.

Run from repo root:
  python experiments/real_video_tracking.py            # every 2nd frame (15 fps of the 30 fps clip)
  python experiments/real_video_tracking.py --stride 6 # every 6th frame (~5 fps, like live on the Mac)
Writes experiments/output/real_cups_stride<N>.mp4, .png and .json (per-frame area + box per object).
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "third_party" / "EdgeTAM")]

import cv2
import numpy as np
import torch

from check_env import patch_attention_for_mps

OUT = ROOT / "experiments" / "output"
VIDEO = ROOT / "third_party" / "EdgeTAM" / "examples" / "02_cups.mp4"


def bgr(hex_):
    h = hex_.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (4, 2, 0))


# name -> (box on frame 0 in pixels, colour). Three cups use the three categorical slots that stay
# distinct when all are on screen together; the ball uses white. Every mask also carries its label.
OBJECTS = {
    "BALL": ((365, 425, 437, 500), bgr("#ffffff")),
    "CUP 1": ((262, 0, 445, 198), bgr("#3987e5")),
    "CUP 2": ((570, 292, 738, 545), bgr("#d95926")),
    "CUP 3": ((830, 292, 998, 545), bgr("#199e70")),
}
PLANE, SURFACE, INK, INK2, GRID = bgr("#0d0d0d"), bgr("#1a1a19"), bgr("#ffffff"), bgr("#c3c2b7"), bgr("#2c2c2a")
FONT = cv2.FONT_HERSHEY_SIMPLEX


def text(img, s, org, color=INK, scale=0.6, thick=1):
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def pill(img, s, org, fill, ink=PLANE, scale=0.6, thick=2):
    (w, h), _ = cv2.getTextSize(s, FONT, scale, thick)
    x, y = org
    cv2.rectangle(img, (x, y - h - 8), (x + w + 14, y + 8), fill, -1)
    text(img, s, (x + 7, y), ink, scale, thick)


def overlay(img, masks):
    out = img.copy()
    for name, (_, color) in OBJECTS.items():
        m = masks[name]
        if m.any():
            tint = out.copy()
            tint[m] = color
            out = cv2.addWeighted(tint, 0.45, out, 0.55, 0)
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, color, 3, cv2.LINE_AA)
    placed = []  # labels last so masks never cover them; stack labels that would overlap
    for name, (_, color) in OBJECTS.items():
        ys, xs = np.nonzero(masks[name])
        if len(xs):
            x, y = max(int(xs.min()), 4), max(int(ys.min()) - 10, 26)
            while any(abs(x - px) < 90 and abs(y - py) < 30 for px, py in placed):
                y += 32
            placed.append((x, y))
            pill(out, name, (x, y), color)
    return out


def status_rows(width, visible, cur):
    """One row per object: a cell per processed frame, filled in the object's colour while visible."""
    rows = np.full((34 * len(OBJECTS) + 40, width, 3), PLANE, np.uint8)
    n = len(next(iter(visible.values())))
    cx0, cx1 = 230, width - 16
    cw = (cx1 - cx0) / n
    for r, (name, (_, color)) in enumerate(OBJECTS.items()):
        y = 8 + r * 34
        text(rows, name, (14, y + 19), INK2, 0.55)
        for i in range(n):
            fill = (color if visible[name][i] else SURFACE) if i <= cur else GRID
            cv2.rectangle(rows, (int(cx0 + i * cw), y + 4), (max(int(cx0 + (i + 1) * cw) - 1, int(cx0 + i * cw)), y + 24), fill, -1)
        cv2.line(rows, (int(cx0 + (cur + 0.5) * cw), y), (int(cx0 + (cur + 0.5) * cw), y + 28), INK, 1)
    text(rows, "Filled = tracker says it can see that object. Dark = 'not visible' (e.g. ball under a cup).",
         (14, 34 * len(OBJECTS) + 26), INK2, 0.5)
    return rows


def render(frame, masks, visible, cur, src_idx, fps_label):
    h, w = frame.shape[:2]
    head = np.full((44, 2 * w, 3), PLANE, np.uint8)
    text(head, f"Real video: 3 identical cups + a ball. Frame {src_idx}  ({fps_label})", (14, 30), INK, 0.75)
    left, right = frame.copy(), overlay(frame, masks)
    pill(left, "What the camera sees", (14, 40), SURFACE, INK, 0.7, 2)
    pill(right, "What the tracker follows (one colour + label per object)", (14, 40), SURFACE, INK, 0.7, 2)
    return np.vstack([head, np.hstack([left, right]), status_rows(2 * w, visible, cur)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=2, help="use every Nth frame of the 30 fps clip")
    args = ap.parse_args()

    cap = cv2.VideoCapture(str(VIDEO))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    frames, src_idx = [], []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % args.stride == 0:
            frames.append(f)
            src_idx.append(i)
        i += 1
    fps_label = f"every {args.stride} frame(s) = {src_fps / args.stride:.0f} fps"
    print(f"{VIDEO.name}: {i} frames, using {len(frames)} ({fps_label})")

    patch_attention_for_mps()
    from sam2.build_sam import build_sam2_video_predictor

    predictor = build_sam2_video_predictor(
        "configs/edgetam.yaml", str(ROOT / "third_party/EdgeTAM/checkpoints/edgetam.pt"), device=torch.device("mps")
    )
    names = list(OBJECTS)
    with tempfile.TemporaryDirectory() as d:
        for k, f in enumerate(frames):
            cv2.imwrite(f"{d}/{k:05d}.jpg", f)
        with torch.inference_mode():
            state = predictor.init_state(video_path=d, offload_video_to_cpu=True)
            for obj_id, name in enumerate(names, start=1):
                predictor.add_new_points_or_box(state, frame_idx=0, obj_id=obj_id, box=np.array(OBJECTS[name][0]))
            masks = {}
            t = time.perf_counter()
            for fi, obj_ids, logits in predictor.propagate_in_video(state):
                masks[fi] = {names[o - 1]: (logits[j, 0] > 0).cpu().numpy() for j, o in enumerate(obj_ids)}
            dt = time.perf_counter() - t
    print(f"tracked {len(names)} objects: {dt / len(frames) * 1000:.0f} ms/frame ({len(frames) / dt:.1f} fps)")

    visible = {n: [bool(masks[k][n].any()) for k in range(len(frames))] for n in names}
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"real_cups_stride{args.stride}"
    first = render(frames[0], masks[0], visible, 0, src_idx[0], fps_label)
    scale = 0.5  # 2560-wide side-by-side -> 1280 for a reasonable file
    size = (int(first.shape[1] * scale), int(first.shape[0] * scale))
    writer = cv2.VideoWriter(f"{stem}.mp4", cv2.VideoWriter_fourcc(*"avc1"), min(src_fps / args.stride, 15), size)
    for k in range(len(frames)):
        writer.write(cv2.resize(render(frames[k], masks[k], visible, k, src_idx[k], fps_label), size, interpolation=cv2.INTER_AREA))
    writer.release()

    keys = np.linspace(0, len(frames) - 1, 6).astype(int)
    tiles = []
    for k in keys:
        tile = cv2.resize(overlay(frames[k], masks[k]), (426, 240), interpolation=cv2.INTER_AREA)
        text(tile, f"frame {src_idx[k]}", (8, 232), INK, 0.5, 1)
        tiles.append(tile)
    rows = status_rows(426 * 6, visible, len(frames) - 1)
    sheet = np.vstack([np.hstack(tiles), rows])
    cv2.imwrite(f"{stem}.png", sheet)
    for n in names:
        gone = [src_idx[k] for k in range(len(frames)) if not visible[n][k]]
        print(f"  {n:6s} not visible in {len(gone)}/{len(frames)} frames" + (f" (source frames {gone[0]}-{gone[-1]})" if gone else ""))
    stats = {}
    for n in names:
        stats[n] = []
        for k in range(len(frames)):
            ys, xs = np.nonzero(masks[k][n])
            box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None
            stats[n].append({"frame": src_idx[k], "area": int(len(xs)), "box": box})
    Path(f"{stem}.json").write_text(json.dumps(stats))
    print(f"wrote {stem.relative_to(ROOT)}.mp4 / .png / .json")


if __name__ == "__main__":
    main()
