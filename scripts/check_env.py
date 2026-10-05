"""Sanity check: SAM 2 loads on MPS and segments/tracks a synthetic moving object.

Run:  python scripts/check_env.py [edgetam|tiny|small] [--stock-attn]

EdgeTAM and SAM 2 both ship a package named `sam2`, so EdgeTAM is loaded by
putting its repo first on sys.path instead of pip-installing it.
"""
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SAM2_DIR = ROOT / "third_party" / "sam2"
MODELS = {
    "edgetam": (ROOT / "third_party" / "EdgeTAM", "configs/edgetam.yaml", "edgetam.pt"),
    "tiny": (SAM2_DIR, "configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
    "small": (SAM2_DIR, "configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
}


def patch_attention_for_mps():
    """F.scaled_dot_product_attention is ~2x slower than plain matmul+softmax on MPS
    for SAM 2's memory-attention shapes (4096 queries x ~28k keys), so swap it out."""
    import types

    import sam2.modeling.sam.transformer as transformer

    def sdpa(q, k, v, dropout_p=0.0, **_):
        return torch.softmax((q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5, dim=-1) @ v

    transformer.F = types.SimpleNamespace(scaled_dot_product_attention=sdpa)


def make_frames(n=30, size=512):
    """A yellow 'key' blob moving right, then half-hidden behind a grey 'hand'."""
    frames = []
    for i in range(n):
        img = np.full((size, size, 3), 40, np.uint8)
        x = 100 + i * 8
        cv2.ellipse(img, (x, 256), (40, 18), 20, 0, 360, (0, 200, 255), -1)
        if i > n // 2:
            cv2.rectangle(img, (x - 10, 200), (x + 80, 320), (150, 150, 150), -1)
        frames.append(img)
    return frames


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    name = args[0] if args else "edgetam"
    repo, cfg, ckpt = MODELS[name]
    sys.path.insert(0, str(repo))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    patched = device.type == "mps" and "--stock-attn" not in sys.argv
    if patched:
        patch_attention_for_mps()
    print(f"torch {torch.__version__} | device: {device} | model: {name} | mps attention patch: {patched}")

    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    ckpt_path = str(repo / "checkpoints" / ckpt)
    frames = make_frames()

    # 1) Single-image segmentation from a box prompt
    t = time.perf_counter()
    image_predictor = SAM2ImagePredictor(build_sam2(cfg, ckpt_path, device=device))
    print(f"image model load: {time.perf_counter() - t:.1f}s")
    rgb = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
    for i in range(4):  # first call includes warm-up
        t = time.perf_counter()
        image_predictor.set_image(rgb)
        masks, scores, _ = image_predictor.predict(box=np.array([50, 220, 150, 290]), multimask_output=False)
        dt = time.perf_counter() - t
    print(f"image predict: {dt * 1000:.0f} ms/frame | mask px={int(masks[0].sum())} score={scores[0]:.2f}")

    # 2) Video tracking through partial occlusion
    video_predictor = build_sam2_video_predictor(cfg, ckpt_path, device=device)
    with tempfile.TemporaryDirectory() as d:
        for i, f in enumerate(frames):
            cv2.imwrite(f"{d}/{i:05d}.jpg", f)
        with torch.inference_mode():
            state = video_predictor.init_state(video_path=d)
            video_predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1, box=np.array([50, 220, 150, 290]))
            t = time.perf_counter()
            areas = [int((logits[0] > 0).sum()) for _, _, logits in video_predictor.propagate_in_video(state)]
            dt = time.perf_counter() - t
    print(f"video tracking: {len(frames) / dt:.1f} fps ({dt / len(frames) * 1000:.0f} ms/frame)")
    print(f"mask area first/mid/last: {areas[0]} / {areas[len(areas) // 2]} / {areas[-1]}  (should shrink once occluded)")
    print("OK")


if __name__ == "__main__":
    main()
