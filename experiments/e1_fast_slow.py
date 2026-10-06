"""E1: fast/slow tracking. Can 5 fps EdgeTAM + 30 fps optical flow keep identities like 30 fps EdgeTAM?

Hypothesis (RESEARCH.md §4.1): identity of identical objects is lost *between* EdgeTAM keyframes, when
objects move more than their own width. Cheap optical flow on every frame never makes that jump, so
it can check EdgeTAM at each keyframe and re-prompt it when they disagree.

Setup, on the real cups clip (3 identical red cups + a ball, 300 frames at 30 fps):
  reference  EdgeTAM on all 300 frames (identities verified by eye at the hard moments)
  baseline   EdgeTAM on every Nth frame only; between keyframes the last result is held
  fast/slow  same EdgeTAM keyframes + DIS optical flow on all frames. At each keyframe, a cup whose
             EdgeTAM mask has left the place the flow predicts (IoU < 0.3), or sits on another cup's
             predicted place, or duplicates another cup's mask, is re-prompted with the flow box.
Baseline and fast/slow run the same code; only the correction step differs.

Scoring: a cup is correct on a frame if its mask overlaps the reference mask of the same cup (IoU >= 0.5),
"on another cup" if it overlaps a different cup's reference mask, "lost" if empty, else "off".

E1b (added after E1a's results): the same EdgeTAM compute, spent on evenly spaced frames vs on the frames
where optical flow says things move fast (a causal rule: frame t is processed if motion t-1 -> t exceeds a
threshold, and at least every 6th frame). Flow decides *when* to look, never *where* an object is.

Run from repo root:  python experiments/e1_fast_slow.py [--part flow|gated|all] [--stride 6 ...]
Writes experiments/output/e1/: e1a_results.json, e1a_stride<N>.png, e1b_results.json, e1b_even_vs_gated.mp4/.png,
index.html (open it; rebuild alone with --part index)
"""
import argparse
import copy
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "third_party" / "EdgeTAM"), str(ROOT / "experiments")]

import cv2
import numpy as np
import torch

from check_env import patch_attention_for_mps
from real_video_tracking import OBJECTS, VIDEO, bgr, overlay, pill, text
from compass.video_io import make_web_playable

OUT = ROOT / "experiments" / "output" / "e1"
SMALL = (320, 180)  # flow + scoring resolution (1/4 of 1280x720)
CUPS = ["CUP 1", "CUP 2", "CUP 3"]
PLANE, SURFACE, INK, INK2, GRID = bgr("#0d0d0d"), bgr("#1a1a19"), bgr("#ffffff"), bgr("#c3c2b7"), bgr("#2c2c2a")
VERDICT = {"correct": bgr("#0ca30c"), "other": bgr("#d03b3b"), "off": bgr("#d03b3b"), "lost": bgr("#898781")}


# ---------------------------------------------------------------- helpers


def small(mask):
    return cv2.resize(mask.astype(np.uint8), SMALL, interpolation=cv2.INTER_NEAREST).astype(bool)


def full(mask, size):
    return cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union else 0.0


def box_of(mask, scale):
    ys, xs = np.nonzero(mask)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32) * scale


def read_frames():
    cap = cv2.VideoCapture(str(VIDEO))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            return frames
        frames.append(f)


class Flow:
    """DIS optical flow on small grey frames, used to carry each mask from frame t-1 to frame t.

    mode "warp":  move every pixel by its own flow vector (backward warp). Follows deformation, but
                  smears masks when objects move fast, blur, or cross each other.
    mode "shift": move the whole mask by the median flow of its own pixels. Keeps the shape; EdgeTAM
                  fixes the shape at the next keyframe anyway.
    """

    def __init__(self, mode):
        self.mode = mode
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        gx, gy = np.meshgrid(np.arange(SMALL[0]), np.arange(SMALL[1]))
        self.grid = np.stack([gx, gy], -1).astype(np.float32)

    def carry(self, prev_bgr, cur_bgr, masks):
        g = lambda f: cv2.cvtColor(cv2.resize(f, SMALL, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        if self.mode == "warp":
            field = self.grid + self.dis.calc(g(cur_bgr), g(prev_bgr), None)  # where each cur pixel came from
            return {n: cv2.remap(m.astype(np.float32), field[..., 0], field[..., 1], cv2.INTER_LINEAR) > 0.5
                    for n, m in masks.items()}
        fwd = self.dis.calc(g(prev_bgr), g(cur_bgr), None)  # where each prev pixel went
        out = {}
        for n, m in masks.items():
            if not m.any():
                out[n] = m
                continue
            dx, dy = np.median(fwd[m], axis=0)
            shift = np.float32([[1, 0, dx], [0, 1, dy]])
            out[n] = cv2.warpAffine(m.astype(np.uint8), shift, SMALL, flags=cv2.INTER_NEAREST).astype(bool)
        return out


class PerObjectTracker:
    """One EdgeTAM state per object (so re-prompting one object never touches another), sharing the
    loaded video and the image-encoder features, so the encoder still runs once per frame."""

    def __init__(self, predictor, frame_dir, boxes):
        self.p = predictor
        predictor.__dict__.pop("_get_image_feature", None)  # drop a previous run's sharing hook
        base = predictor.init_state(video_path=frame_dir, offload_video_to_cpu=True)
        self.states = []
        for _ in boxes:
            memo = {id(base["images"]): base["images"], id(base["cached_features"]): {}}
            self.states.append(copy.deepcopy(base, memo))
        self._share_encoder()
        self.first = [(predictor.add_new_points_or_box(s, frame_idx=0, obj_id=1, box=b)[2][0, 0] > 0).cpu().numpy()
                      for s, b in zip(self.states, boxes)]

    def _share_encoder(self):
        orig, shared = type(self.p)._get_image_feature.__get__(self.p), {}

        def get(inference_state, frame_idx, batch_size):
            if frame_idx in shared and frame_idx not in inference_state["cached_features"]:
                inference_state["cached_features"] = {frame_idx: shared[frame_idx]}
            out = orig(inference_state, frame_idx, batch_size)
            shared.clear()
            shared[frame_idx] = inference_state["cached_features"][frame_idx]
            return out

        self.p._get_image_feature = get

    def step(self, k):
        """Track every object one keyframe forward. Returns full-res bool masks."""
        outs = []
        for s in self.states:
            _, _, m = next(iter(self.p.propagate_in_video(s, start_frame_idx=k, max_frame_num_to_track=0)))
            outs.append((m[0, 0] > 0).cpu().numpy())
        return outs

    def reprompt(self, i, k, box):
        """Re-initialise object i on keyframe k from a box: segment the box with no memory (as on the first
        frame), make it a conditioning frame, and forget the object's recent non-conditioning memory, which
        is what carried the wrong identity. A plain prompt on a tracked frame gets overridden by that memory."""
        s = self.states[i]
        s["frames_already_tracked"].pop(k, None)  # -> treated as an initial conditioning frame (no memory)
        for d in (s["output_dict"], s["output_dict_per_obj"][0]):
            for f in [f for f in d["non_cond_frame_outputs"] if f <= k]:
                d["non_cond_frame_outputs"].pop(f)
        _, _, m = self.p.add_new_points_or_box(s, frame_idx=k, obj_id=1, box=box)
        return (m[0, 0] > 0).cpu().numpy()


# ---------------------------------------------------------------- runs


def build_predictor():
    patch_attention_for_mps()
    import sam2.sam2_video_predictor as vp
    from sam2.build_sam import build_sam2_video_predictor

    vp.tqdm = lambda it, **_: it  # silence per-step progress bars
    return build_sam2_video_predictor(
        "configs/edgetam.yaml", str(ROOT / "third_party/EdgeTAM/checkpoints/edgetam.pt"), device=torch.device("mps")
    )


def reference_run(predictor, frames):
    """EdgeTAM on every frame (batched, as in real_video_tracking.py). Returns {name: [small masks]}."""
    names = list(OBJECTS)
    with tempfile.TemporaryDirectory() as d:
        for k, f in enumerate(frames):
            cv2.imwrite(f"{d}/{k:05d}.jpg", f)
        with torch.inference_mode():
            st = predictor.init_state(video_path=d, offload_video_to_cpu=True)
            for oid, n in enumerate(names, 1):
                predictor.add_new_points_or_box(st, frame_idx=0, obj_id=oid, box=np.array(OBJECTS[n][0]))
            ref = {n: [None] * len(frames) for n in names}
            for fi, ids, logits in predictor.propagate_in_video(st):
                for j, oid in enumerate(ids):
                    ref[names[oid - 1]][fi] = small((logits[j, 0] > 0).cpu().numpy())
    return ref


def keyframe_run(predictor, frames, stride, correct, flow_mode="shift", oracle=None, keys=None):
    """EdgeTAM on every `stride`-th frame (or on the frames listed in `keys`). EdgeTAM only ever sees those
    frames, so its memory holds its last processed frames. If `correct`, add 30 fps flow + keyframe corrections.
    If `oracle` (reference masks) is given, the prior at each keyframe is the reference itself: an upper
    bound for the correction mechanism with a perfect fast tracker."""
    names = list(OBJECTS)
    h, w = frames[0].shape[:2]
    scale = w / SMALL[0]
    keys = keys or list(range(0, len(frames), stride))
    kidx = {t: k for k, t in enumerate(keys)}
    flow = Flow(flow_mode)
    log = {"corrections": [], "edgetam_ms": [], "flow_ms": []}
    out = {n: [None] * len(frames) for n in names}  # small masks shown on each source frame
    visible = {n: [False] * len(frames) for n in names}
    with tempfile.TemporaryDirectory() as d:
        for k, t in enumerate(keys):
            cv2.imwrite(f"{d}/{k:05d}.jpg", frames[t])
        with torch.inference_mode():
            trk = PerObjectTracker(predictor, d, [np.array(OBJECTS[n][0]) for n in names])
            cur = {n: small(m) for n, m in zip(names, trk.first)}  # current belief, warped by flow
            was_visible = {n: bool(cur[n].any()) for n in names}
            key_area = {n: int(cur[n].sum()) for n in names}
            for t in range(len(frames)):
                if t > 0 and correct:
                    t0 = time.perf_counter()
                    cur = flow.carry(frames[t - 1], frames[t], cur)
                    log["flow_ms"].append((time.perf_counter() - t0) * 1000)
                if t in kidx and t > 0:
                    k = kidx[t]
                    torch.mps.synchronize()
                    t0 = time.perf_counter()
                    edge = dict(zip(names, (small(m) for m in trk.step(k))))
                    if oracle is not None:
                        cur = {n: oracle[n][t] for n in names}
                        was_visible = {n: bool(oracle[n][keys[k - 1]].any()) for n in names}
                    if correct:
                        for i, n in enumerate(names):
                            if not (edge[n].any() and was_visible[n] and cur[n].any()):
                                continue  # only check objects that are visible now and were before
                            if not 0.5 <= cur[n].sum() / max(key_area[n], 1) <= 1.6:
                                continue  # flow smeared or lost the mask; don't trust it
                            own = iou(edge[n], cur[n])
                            other = max((iou(edge[n], cur[o]) for o in names if o != n and cur[o].any()), default=0)
                            dup = any(iou(edge[n], edge[o]) > 0.5 and iou(edge[o], cur[o]) > own
                                      for o in names if o != n and edge[o].any())
                            if own < 0.3 or (other > 0.5 and other > own) or dup:
                                fixed = small(trk.reprompt(i, k, box_of(cur[n], scale)))
                                log["corrections"].append({"frame": t, "object": n, "iou_before": round(own, 2),
                                                           "iou_after": round(iou(fixed, cur[n]), 2)})
                                edge[n] = fixed
                    torch.mps.synchronize()
                    log["edgetam_ms"].append((time.perf_counter() - t0) * 1000)
                    for n in names:
                        was_visible[n] = bool(edge[n].any())
                        if edge[n].any() or not correct:
                            cur[n] = edge[n]  # hidden objects keep their flow-carried belief (fast/slow only)
                        key_area[n] = int(cur[n].sum())
                for n in names:
                    if t in kidx:
                        visible[n][t] = was_visible[n]
                        out[n][t] = cur[n] if was_visible[n] else np.zeros_like(cur[n])
                    elif correct:
                        visible[n][t] = was_visible[n]
                        out[n][t] = cur[n] if was_visible[n] else np.zeros_like(cur[n])
                    else:  # baseline holds the last keyframe result
                        visible[n][t], out[n][t] = visible[n][t - 1], out[n][t - 1]
            log["ghost"] = {n: cur[n] for n in names}  # where flow thinks hidden objects are at the end
    return out, visible, log


def motion_scores(frames):
    """Per-frame motion: 99th percentile of optical-flow speed (px/frame at 1/4 resolution) between frame
    t-1 and t. Only says *how much* is moving, never *where an object is*."""
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    g = lambda f: cv2.cvtColor(cv2.resize(f, SMALL, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    scores = [0.0]
    for t in range(1, len(frames)):
        scores.append(float(np.percentile(np.linalg.norm(dis.calc(g(frames[t - 1]), g(frames[t]), None), axis=-1), 99)))
    return scores


def gated_keys(scores, thr, max_gap=6):
    """Causal frame selection: run EdgeTAM on frame t if motion(t-1 -> t) > thr, else at least every max_gap."""
    keys, last = [0], 0
    for t in range(1, len(scores)):
        if scores[t] > thr or t - last >= max_gap:
            keys.append(t)
            last = t
    return keys


def threshold_for_budget(scores, budget, max_gap=6):
    """Motion threshold whose selection processes the fraction `budget` of frames (for a matched comparison)."""
    return min(sorted(set(scores)), key=lambda thr: abs(len(gated_keys(scores, thr, max_gap)) / len(scores) - budget))


def flow_diagnostic(frames, ref, stride, mode):
    """Start each gap from the reference masks, carry them `stride` frames with flow alone, and compare with
    the reference at the end of the gap. Returns per-gap IoUs for each cup that stays visible."""
    flow, ious = Flow(mode), []
    for t0 in range(0, len(frames) - stride, stride):
        cur = {c: ref[c][t0] for c in CUPS}
        for t in range(t0 + 1, t0 + stride + 1):
            cur = flow.carry(frames[t - 1], frames[t], cur)
        for c in CUPS:
            if ref[c][t0].sum() > 500 and ref[c][t0 + stride].sum() > 500:  # visible at both ends
                ious.append({"frame": t0, "cup": c, "iou": round(float(iou(cur[c], ref[c][t0 + stride])), 2)})
    return ious


def score(pred, ref):
    """Per cup, per frame: correct | other | lost | off (see module docstring)."""
    res = {}
    for n in CUPS:
        row = []
        for t in range(len(ref[n])):
            p = pred[n][t]
            if not p.any():
                row.append("lost")
            elif iou(p, ref[n][t]) >= 0.5:
                row.append("correct")
            elif any(iou(p, ref[o][t]) >= 0.5 for o in CUPS if o != n):
                row.append("other")
            else:
                row.append("off")
        res[n] = row
    return res


def summarise(verdicts, frames_idx):
    cells = [verdicts[n][t] for n in CUPS for t in frames_idx]
    return {v: cells.count(v) for v in ("correct", "other", "off", "lost")} | {"total": len(cells)}


# ---------------------------------------------------------------- rendering


def rows(width, runs, cur, labels):
    """Timeline per method x cup: green right cup, red on another cup / off, grey lost."""
    img = np.full((24 + len(labels) * (len(CUPS) * 22 + 34) + 30, width, 3), PLANE, np.uint8)
    cx0, cx1 = 300, width - 16
    n = len(runs[labels[0][1]]["verdicts"][CUPS[0]])
    cw = (cx1 - cx0) / n
    y, ticks = 8, False
    for title, key in labels:
        text(img, title, (14, y + 16), INK, 0.55)
        y += 26
        for c in CUPS:
            text(img, c, (34, y + 14), INK2, 0.45)
            for t in range(n):
                color = VERDICT[runs[key]["verdicts"][c][t]] if t <= cur else GRID
                cv2.rectangle(img, (int(cx0 + t * cw), y + 2), (max(int(cx0 + (t + 1) * cw) - 1, int(cx0 + t * cw)), y + 18), color, -1)
            y += 22
        for t in runs[key].get("marks", []):  # frames where EdgeTAM ran (gated) or was corrected (flow)
            x = int(cx0 + (t + 0.5) * cw)
            cv2.line(img, (x, y - len(CUPS) * 22 - 6), (x, y - len(CUPS) * 22 - 1), INK2, 1)
            ticks = True
        y += 8
    x = 14
    legend = [("on the right cup", VERDICT["correct"]), ("on another cup / off", VERDICT["other"]), ("lost", VERDICT["lost"])]
    if ticks:
        legend.append(("| above a row = frame EdgeTAM processed", INK2))
    for label, color in legend:
        cv2.circle(img, (x + 6, y + 8), 6, color, -1, cv2.LINE_AA)
        text(img, label, (x + 18, y + 13), INK2, 0.45)
        x += 40 + cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
    cv2.line(img, (int(cx0 + (cur + 0.5) * cw), 6), (int(cx0 + (cur + 0.5) * cw), y - 6), INK, 1)
    return img


def panel(frame, masks_small, verdict_at, title):
    h, w = frame.shape[:2]
    img = overlay(frame, {n: full(m, (w, h)) for n, m in masks_small.items()})
    pill(img, title, (14, 40), SURFACE, INK, 0.8, 2)
    y = 84
    for c in CUPS:
        v = verdict_at[c]
        word = {"correct": "right cup", "other": "WRONG cup", "off": "WRONG (off)", "lost": "lost"}[v]
        pill(img, f"{c}: {word}", (14, y), VERDICT[v], INK if v != "correct" else PLANE, 0.7, 2)
        y += 40
    return cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)


def render(frames, runs, t, left, right, head):
    """left/right = (title, run key). Two labelled views of the same frame + the verdict timelines."""
    w = frames[0].shape[1]
    top = np.full((44, w, 3), PLANE, np.uint8)
    text(top, f"{head}  |  frame {t}", (14, 30), INK, 0.7)
    views = [panel(frames[t], {n: runs[k]["masks"][n][t] for n in OBJECTS}, {c: runs[k]["verdicts"][c][t] for c in CUPS}, title)
             for title, k in (left, right)]
    return np.vstack([top, np.hstack(views), rows(w, runs, t, [left, right])])


def write_video(path, frames, runs, left, right, head):
    size = render(frames, runs, 0, left, right, head).shape[1::-1]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), 10, size)
    for t in range(0, len(frames), 2):
        writer.write(render(frames, runs, t, left, right, head))
    writer.release()
    make_web_playable(path)


def compute_ms(log, n_runs, n_frames, flow_ms=0.0):
    """Average compute per second of 30 fps video: EdgeTAM runs + flow on every frame (if used)."""
    edge = float(np.mean(log["edgetam_ms"])) if log["edgetam_ms"] else 0.0
    return round(edge * 30 * n_runs / n_frames + flow_ms * 30)


def flow_part(predictor, frames, ref, strides):
    """E1a: fixed-rate EdgeTAM, with and without flow-based corrections, plus the perfect-prior upper bound."""
    summary = {}
    variants = (("base", False, None, "EdgeTAM alone"), ("warp", True, "warp", "+ flow corrections (warp)"),
                ("fast", True, "shift", "+ flow corrections (shift)"), ("oracle", True, "shift", "+ perfect prior"))
    for stride in strides:
        runs, entry, keyframes = {}, {}, list(range(0, len(frames), stride))
        for key, correct, mode, label in variants:
            masks, _, log = keyframe_run(predictor, frames, stride, correct, mode, oracle=ref if key == "oracle" else None)
            runs[key] = {"masks": masks, "verdicts": score(masks, ref), "marks": [c["frame"] for c in log["corrections"]]}
            flow_ms = float(np.mean(log["flow_ms"])) if log["flow_ms"] else 0.0
            entry[key] = {"keyframes": summarise(runs[key]["verdicts"], keyframes),
                          "all_frames": summarise(runs[key]["verdicts"], range(len(frames))),
                          "compute_ms_per_second_of_video": compute_ms(log, len(keyframes), len(frames), flow_ms),
                          "corrections": log["corrections"]}
        for mode in ("warp", "shift"):
            diag = flow_diagnostic(frames, ref, stride, mode)
            vals = np.array([d["iou"] for d in diag])
            entry[f"flow_alone_{mode}"] = {"gaps": len(vals), "median_iou": round(float(np.median(vals)), 2),
                                           "gaps_below_0.5": int((vals < 0.5).sum()),
                                           "worst": sorted(diag, key=lambda d: d["iou"])[:5]}
        summary[f"stride{stride}"] = entry
        print(f"\n== E1a, EdgeTAM every {stride} frames ({30 // stride} fps)")
        for key, _, _, label in variants:
            e = entry[key]
            print(f"  {label:28s} all frames correct {e['all_frames']['correct']}/{e['all_frames']['total']} "
                  f"(keyframes {e['keyframes']['correct']}/{e['keyframes']['total']}) | {e['compute_ms_per_second_of_video']} ms "
                  f"compute per second of video | corrections {len(e['corrections'])}")
        for mode in ("warp", "shift"):
            d = entry[f"flow_alone_{mode}"]
            print(f"  flow alone ({mode}) across {stride}-frame gaps: median IoU {d['median_iou']}, {d['gaps_below_0.5']}/{d['gaps']} below 0.5")
        cv2.imwrite(str(OUT / f"e1a_stride{stride}.png"),
                    rows(frames[0].shape[1], runs, len(frames) - 1, [(f"{lbl}, {30 // stride} fps", k) for k, _, _, lbl in variants]))
    return summary


def gated_part(predictor, frames, ref):
    """E1b: same EdgeTAM compute, spent evenly vs where the motion is."""
    scores = motion_scores(frames)
    summary, runs_for_video = {}, None
    for budget, stride in ((1 / 3, 3), (1 / 2, 2)):
        thr = threshold_for_budget(scores, budget)
        keys = gated_keys(scores, thr)
        # "boost": never wait more than 3 frames, spend the rest of the budget on the fastest moments
        boost_thr = threshold_for_budget(scores, budget, max_gap=3)
        boost = gated_keys(scores, boost_thr, max_gap=3)
        runs, entry = {}, {}
        variants = [("even", list(range(0, len(frames), stride))), ("gated", keys)]
        if len(boost) > len(variants[0][1]) * 0.9:  # only when the budget allows a 1-in-3 floor
            variants.append(("boost", boost))
        for key, sel in variants:
            masks, _, log = keyframe_run(predictor, frames, None, False, keys=sel)
            runs[key] = {"masks": masks, "verdicts": score(masks, ref), "marks": sel}
            entry[key] = {"frames_processed": len(sel), "all_frames": summarise(runs[key]["verdicts"], range(len(frames))),
                          "compute_ms_per_second_of_video": compute_ms(log, len(sel), len(frames), 4.0 if key == "gated" else 0.0)}
        entry["motion_threshold_px_per_frame"] = round(thr, 2)
        entry["boost_threshold_px_per_frame"] = round(boost_thr, 2)
        summary[f"budget_{len(keys)}_of_{len(frames)}"] = entry
        print(f"\n== E1b, EdgeTAM on about {budget:.0%} of frames (motion threshold {thr:.1f} px/frame)")
        for key, label in (("even", f"evenly spaced ({30 // stride} fps)"), ("gated", "where the motion is"),
                           ("boost", "every 3rd + fast moments")):
            if key not in entry:
                continue
            e = entry[key]
            a = e["all_frames"]
            print(f"  {label:24s} {e['frames_processed']:3d} frames processed | all frames correct {a['correct']}/{a['total']} "
                  f"(wrong cup {a['other']}, off {a['off']}, lost {a['lost']}) | {e['compute_ms_per_second_of_video']} ms compute per second of video")
        runs_for_video = (runs, stride, budget)
    runs, stride, budget = runs_for_video
    left, right = (f"evenly spaced ({30 // stride} fps)", "even"), ("motion-gated (same compute)", "gated")
    write_video(OUT / "e1b_even_vs_gated.mp4", frames, runs, left, right,
                f"E1b: EdgeTAM on {budget:.0%} of frames, evenly vs where the motion is")
    cv2.imwrite(str(OUT / "e1b_even_vs_gated.png"), rows(frames[0].shape[1], runs, len(frames) - 1, [left, right]))
    return summary


def write_index():
    """Results page built from e1a_results.json + e1b_results.json (no GPU needed)."""
    import html as _html
    a = json.loads((OUT / "e1a_results.json").read_text())
    b = json.loads((OUT / "e1b_results.json").read_text())
    pct = lambda d: f"{d['correct'] / d['total']:.0%}"
    rows_a = []
    for stride in (6, 3, 2):
        e = a.get(f"stride{stride}")
        if not e:
            continue
        cells = "".join(f"<td><b>{pct(e[k]['all_frames'])}</b> <small>({e[k]['all_frames']['correct']}/{e[k]['all_frames']['total']})</small></td>"
                        for k in ("base", "warp", "fast", "oracle"))
        fa = e["flow_alone_shift"]
        rows_a.append(f"<tr><td>{30 // stride} fps (every {stride})</td>{cells}"
                      f"<td>{fa['gaps_below_0.5']} of {fa['gaps']} gaps</td></tr>")
    rows_b = []
    for key, e in b.items():
        n = e["even"]["frames_processed"]
        cells = "".join(f"<td><b>{pct(e[k]['all_frames'])}</b> <small>({e[k]['all_frames']['correct']}/900)</small></td>"
                        if k in e else "<td>same as even</td>" for k in ("even", "gated", "boost"))
        rows_b.append(f"<tr><td>{n} of 300 frames ({n / 300:.0%})</td>{cells}"
                      f"<td>{e['even']['compute_ms_per_second_of_video'] / 1000:.1f} s</td></tr>")
    vids = "".join(f"<figure><figcaption>{_html.escape(cap)}</figcaption><video src='{src}' controls loop muted playsinline "
                   f"preload='metadata'></video></figure>" for src, cap in (
                       ("e1b_even_vs_gated.mp4", "E1b: EdgeTAM on 50% of frames, evenly spaced (left) vs where the motion is (right)"),
                       ("e1a_stride6_base_vs_flow.mp4", "E1a: EdgeTAM at 5 fps alone (left) vs + flow corrections, shift (right)")))
    (OUT / "index.html").write_text(f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>E1 Fast Slow Tracking</title>
<style>
:root{{--plane:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e1e0d9}}
@media (prefers-color-scheme:dark){{:root{{--plane:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--line:#2c2c2a}}}}
body{{margin:0;padding:24px 16px;background:var(--plane);color:var(--ink);font:15px/1.55 system-ui,sans-serif}}
main{{max-width:1200px;margin:auto}} h1{{font-size:22px;margin:0 0 8px}} h2{{font-size:17px;margin:0 0 6px}}
p,li,figcaption,small{{color:var(--ink2)}} b{{color:var(--ink)}}
section{{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px;margin:16px 0}}
.wrap{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-size:14px}}
td,th{{text-align:left;padding:8px;border-bottom:1px solid var(--line);vertical-align:top}} th{{color:var(--ink2);font-weight:600}}
.vids{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:12px}} figure{{margin:0}}
video,img{{width:100%;border-radius:6px;background:#000}} figcaption{{font-size:13px;margin-bottom:4px}}
</style></head><body><main>
<h1>E1: can cheap motion tracking replace heavy-tracker frames?</h1>
<p>Real cups clip (3 identical cups + a ball, 300 frames at 30 fps). "Correct" = a cup's mask overlaps the
reference mask of the same cup (IoU &ge; 0.5), counted over all 300 frames x 3 cups. Reference = EdgeTAM on every
frame, checked by eye (and the ball is revealed under CUP 1, the cup that covered it).</p>
<section class="wrap"><h2>E1a: flow corrections at a fixed EdgeTAM rate</h2><table>
<tr><th>EdgeTAM rate</th><th>EdgeTAM alone</th><th>+ flow corrections (warp)</th><th>+ flow corrections (shift)</th>
<th>+ perfect prior (upper bound)</th><th>Flow alone loses the cup in</th></tr>{''.join(rows_a)}</table>
<p><b>Takeaway:</b> the correction mechanism works (perfect prior: 40% &rarr; 91% at 5 fps), but cheap optical flow
loses cups exactly in the fast, blurred swaps, and a wrong correction makes things worse.</p></section>
<section class="wrap"><h2>E1b: same EdgeTAM compute, spent evenly vs where the motion is</h2><table>
<tr><th>Budget</th><th>Evenly spaced</th><th>Where the motion is</th><th>Every 3rd + fast moments</th>
<th>Compute per 1 s of video (M2, 4 objects)</th></tr>{''.join(rows_b)}</table>
<p><b>Takeaway:</b> nothing beats evenly spaced frames at 50% budget. This clip moves almost all the time, so
there are no calm moments to save on. Real-time needs under 1 s of compute per 1 s of video.</p></section>
<section><h2>Videos</h2><div class="vids">{vids}
<figure><figcaption>Reference check (EdgeTAM on every frame) at 9 moments</figcaption><img src="reference_check.jpg" alt="Reference masks at nine frames"></figure>
</div></section>
</main></body></html>""")


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["flow", "gated", "index", "all"], default="all")
    ap.add_argument("--stride", type=int, action="append", help="E1a keyframe stride(s); default 6, 3, 2")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.part == "index":
        write_index()
        print(f"wrote {(OUT / 'index.html').relative_to(ROOT)}")
        return
    frames = read_frames()
    predictor = build_predictor()

    cache = OUT / "reference_masks.npz"
    if cache.exists():  # the reference is deterministic; delete the file to recompute it
        ref = {n: list(v) for n, v in np.load(cache).items()}
        print(f"reference: loaded {cache.relative_to(ROOT)}")
    else:
        print("reference: EdgeTAM on all 300 frames ...", flush=True)
        ref = reference_run(predictor, frames)
        np.savez_compressed(cache, **{n: np.stack(v) for n, v in ref.items()})

    if args.part in ("flow", "all"):
        res = flow_part(predictor, frames, ref, args.stride or [6, 3, 2])
        (OUT / "e1a_results.json").write_text(json.dumps(res, indent=1))
    if args.part in ("gated", "all"):
        res = gated_part(predictor, frames, ref)
        (OUT / "e1b_results.json").write_text(json.dumps(res, indent=1))
    if (OUT / "e1a_results.json").exists() and (OUT / "e1b_results.json").exists():
        write_index()
    print(f"\nwrote {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
