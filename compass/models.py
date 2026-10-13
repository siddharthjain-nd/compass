"""Model loading for the Mac: EdgeTAM (video tracker + image segmenter) on the GPU (MPS).

EdgeTAM ships a package named `sam2`, like SAM 2 itself, so it is imported from third_party/EdgeTAM by
putting that folder first on sys.path (see scripts/setup_third_party.sh).
"""
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

ROOT = Path(__file__).resolve().parents[1]
EDGETAM = ROOT / "third_party" / "EdgeTAM"
EDGETAM_CONFIG = "configs/edgetam.yaml"
EDGETAM_CKPT = EDGETAM / "checkpoints" / "edgetam.pt"

if str(EDGETAM) not in sys.path:
    sys.path.insert(0, str(EDGETAM))


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def patch_attention_for_mps():
    """F.scaled_dot_product_attention is ~2x slower than plain matmul+softmax on MPS for SAM 2's
    memory-attention shapes (4096 queries x ~28k keys), so swap it out."""
    import sam2.modeling.sam.transformer as transformer

    def sdpa(q, k, v, dropout_p=0.0, **_):
        return torch.softmax((q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5, dim=-1) @ v

    transformer.F = types.SimpleNamespace(scaled_dot_product_attention=sdpa)


def video_predictor(dev=None, image_size=1024):
    """EdgeTAM video predictor, with the MPS attention patch and per-step progress bars silenced.
    image_size: EdgeTAM was trained at 1024; other sizes (multiples of 256) change the feature grid (image_size / 16)
    and the memory step's query position table to match (E5c test)."""
    dev = dev or device()
    if dev.type == "mps":
        patch_attention_for_mps()
    import sam2.sam2_video_predictor as vp
    from sam2.build_sam import build_sam2_video_predictor

    vp.tqdm = lambda it, **_: it
    return build_sam2_video_predictor(EDGETAM_CONFIG, str(EDGETAM_CKPT), device=dev,
                                      hydra_overrides_extra=_size_overrides(image_size))


def _size_overrides(image_size):
    g = image_size // 16
    return [] if image_size == 1024 else [f"++model.image_size={image_size}",
                                          f"++model.memory_attention.layer.cross_attention.q_sizes=[{g},{g}]"]


def image_predictor(dev=None, image_size=1024):
    """EdgeTAM's single-image segmenter (SAM-style box / point prompts); image_size as for video_predictor."""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    pred = SAM2ImagePredictor(build_sam2(EDGETAM_CONFIG, str(EDGETAM_CKPT), device=dev or device(),
                                         hydra_overrides_extra=_size_overrides(image_size)))
    g = image_size // 16
    pred._bb_feat_sizes = [(4 * g, 4 * g), (2 * g, 2 * g), (g, g)]  # hard-coded for 1024 in SAM2ImagePredictor
    return pred
