"""E5 probe: where EdgeTAM's time goes on the M2, and how much the cheap knobs from the speed research would save.

Timing only (accuracy of each knob is a separate test). It tracks the wallet in one of the user's clips for 20
frames so the memory is full, captures the real inputs of the memory-attention step, then times variants of it:
  - fewer memory frames (EdgeTAM uses 7; the SAM 2 paper found 4 about as accurate)
  - queries only in a window around the object ("region of interest"), instead of the whole 64x64 feature grid
  - several objects at once (EdgeTAM batches objects: does the cost grow with N?)
  - half precision (float16)
and the image encoder at smaller input sizes.

Run:  python experiments/e5_speed_probe.py
"""
import json
import time

import cv2
import torch

from compass.models import ROOT, device, video_predictor
from compass.tracker import StreamingTracker

CLIP, BOX = ROOT / "clips" / "20261009-002032.mp4", [597, 387, 978, 596]
REPEATS = 15


def timed(fn, repeats=REPEATS):
    """Median ms of fn() on the GPU (synchronised), after 3 warm-up calls."""
    for _ in range(3):
        fn()
    torch.mps.synchronize()
    out = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        torch.mps.synchronize()
        out.append((time.perf_counter() - t) * 1000)
    return round(sorted(out)[len(out) // 2], 1)


def main():
    dev = device()
    pred = video_predictor(dev)
    captured = {}

    def hook(_module, args, kwargs):
        captured.update(args=args, kwargs=kwargs)

    handle = pred.memory_attention.register_forward_pre_hook(hook, with_kwargs=True)
    cap = cv2.VideoCapture(str(CLIP))
    tracker = StreamingTracker(pred)
    frames = []
    for k in range(20):
        ok, img = cap.read()
        frames.append(img)
        tracker.start(img, BOX) if k == 0 else tracker.track(img)
    handle.remove()

    kw = captured["kwargs"]
    curr, memory = kw["curr"], kw["memory"]
    curr, curr_pos = (curr[0], kw["curr_pos"][0]) if isinstance(curr, list) else (curr, kw["curr_pos"])
    mem_pos, n_ptr, n_spatial = kw["memory_pos"], kw["num_obj_ptr_tokens"], kw["num_spatial_mem"]
    per_frame = (memory.shape[0] - n_ptr) // n_spatial
    shapes = {"query_tokens": curr.shape[0], "memory_tokens": memory.shape[0], "memory_frames": n_spatial,
              "tokens_per_memory_frame": per_frame, "object_pointer_tokens": n_ptr}
    print("memory attention inputs:", shapes)

    ma = pred.memory_attention
    cross = [layer.cross_attn_image for layer in ma.layers]

    def set_query_grid(side):
        for c in cross:
            c.freqs_cis_q = c.compute_cis(end_x=side, end_y=side).to(dev)

    def run(c, cp, m, mp, n_sp, n_objects=1):
        if n_objects > 1:
            c, cp, m, mp = (t.repeat(1, n_objects, 1) for t in (c, cp, m, mp))
        with torch.inference_mode():
            return ma(curr=c, memory=m, curr_pos=cp, memory_pos=mp, num_obj_ptr_tokens=n_ptr, num_spatial_mem=n_sp)

    def keep_frames(n):
        """The last n spatial memory frames + all object pointers."""
        lo = (n_spatial - n) * per_frame
        hi = n_spatial * per_frame
        sel = torch.cat([torch.arange(lo, hi), torch.arange(hi, memory.shape[0])]).to(dev)
        return memory[sel], mem_pos[sel], n

    def window(side):
        """Queries of a side x side window centred on the 64x64 grid (stands in for 'around the object')."""
        g = int(curr.shape[0] ** 0.5)
        idx = torch.arange(g * g, device=dev).view(g, g)
        o = (g - side) // 2
        sel = idx[o:o + side, o:o + side].reshape(-1)
        return curr[sel], curr_pos[sel]

    results = {"shapes": shapes, "memory_attention_ms": {}, "encoder_ms": {}}
    r = results["memory_attention_ms"]
    r["full (7 memory frames, 64x64 queries)"] = timed(lambda: run(curr, curr_pos, memory, mem_pos, n_spatial))
    for n in (4, 2):
        m, mp, ns = keep_frames(n)
        r[f"{n} memory frames"] = timed(lambda: run(curr, curr_pos, m, mp, ns))
    for side in (32, 24, 16):
        c, cp = window(side)
        set_query_grid(side)
        r[f"window {side}x{side} queries (object ~{side * 16}px in the 1024 input)"] = \
            timed(lambda: run(c, cp, memory, mem_pos, n_spatial))
        m, mp, ns = keep_frames(4)
        r[f"window {side}x{side} + 4 memory frames"] = timed(lambda: run(c, cp, m, mp, ns))
    set_query_grid(64)
    for n_obj in (2, 4, 8):
        r[f"{n_obj} objects batched (full)"] = timed(lambda: run(curr, curr_pos, memory, mem_pos, n_spatial, n_obj))
    c, cp = window(24)
    set_query_grid(24)
    m, mp, ns = keep_frames(4)
    for n_obj in (4, 8):
        r[f"{n_obj} objects batched (window 24 + 4 frames)"] = timed(lambda: run(c, cp, m, mp, ns, n_obj))
    set_query_grid(64)
    with torch.autocast("mps", dtype=torch.float16):
        try:
            r["full, float16 (autocast)"] = timed(lambda: run(curr, curr_pos, memory, mem_pos, n_spatial))
        except Exception as e:  # noqa: BLE001 - report, don't crash the probe
            r["full, float16 (autocast)"] = f"failed: {e}"[:120]

    e = results["encoder_ms"]
    rgb = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2RGB)
    for size in (1024, 768, 512):
        x = torch.from_numpy(cv2.resize(rgb, (size, size))).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
        with torch.inference_mode():
            e[f"{size}px"] = timed(lambda: pred.forward_image(x))
            with torch.autocast("mps", dtype=torch.float16):
                try:
                    e[f"{size}px float16"] = timed(lambda: pred.forward_image(x))
                except Exception as ex:  # noqa: BLE001
                    e[f"{size}px float16"] = f"failed: {ex}"[:120]

    print(json.dumps(results, indent=1))
    out = ROOT / "experiments" / "output" / "e5"
    out.mkdir(parents=True, exist_ok=True)
    (out / "speed_probe.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
