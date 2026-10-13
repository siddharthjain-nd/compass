"""EdgeTAM's image encoder through Core ML (Apple's own runtime: Neural Engine + GPU), for the video tracker (E5, round 3a).

The image encoder is the biggest single cost of a tracked frame (~88 ms on the M2 in PyTorch/MPS). Everything else
(memory, decoder) stays in PyTorch. Core ML returns exactly what EdgeTAM's `forward_image` returns: the three feature
maps (after the decoder's high-resolution projections conv_s0 / conv_s1). The positional encodings don't depend on
the image, so they are computed once in PyTorch and reused.

EdgeTAM ships its own Core ML export (third_party/EdgeTAM/coreml), but for single images: it skips the ImageNet
normalisation and adds the "no memory" embedding to the features, which the video tracker must not have. So this is
our own export of `forward_image`, with the normalisation inside the model (the input is the 1024x1024 RGB picture).

Export once (~15 s):  python -m compass.coreml_encoder [image_size, default 1024]
"""
import sys

import numpy as np
import torch
from PIL import Image

from compass.models import ROOT


def model_path(image_size=1024):
    return ROOT / "third_party" / "models" / f"edgetam_image_encoder_video{'' if image_size == 1024 else f'_{image_size}'}.mlpackage"


MODEL = model_path()
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class _Encoder(torch.nn.Module):
    """RGB 0-255 picture -> EdgeTAM backbone_fpn (3 levels), as forward_image computes them."""

    def __init__(self, sam):
        super().__init__()
        self.sam = sam
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1) * 255)

    def forward(self, image):
        out = self.sam.forward_image((image - self.mean) / self.std)
        f0, f1, f2 = out["backbone_fpn"]
        return f0, f1, f2


def export(image_size=1024):
    import coremltools as ct

    from compass.models import video_predictor

    path = model_path(image_size)
    sam = video_predictor(torch.device("cpu"), image_size).eval()
    enc = _Encoder(sam).eval()
    with torch.no_grad():
        traced = torch.jit.trace(enc, torch.rand(1, 3, image_size, image_size) * 255)
    ml = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=(1, 3, image_size, image_size), scale=1.0, bias=[0, 0, 0],
                             color_layout=ct.colorlayout.RGB)],
        outputs=[ct.TensorType(name="fpn0"), ct.TensorType(name="fpn1"), ct.TensorType(name="fpn2")],
        minimum_deployment_target=ct.target.macOS14,
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
    )
    ml.short_description = "EdgeTAM image encoder for video tracking (normalisation inside; backbone_fpn outputs)"
    path.parent.mkdir(parents=True, exist_ok=True)
    ml.save(str(path))
    print(f"saved {path}")


class CoreMLEncoder:
    """Drop-in for the video predictor's forward_image: give it the (image_size x image_size) RGB picture first."""

    def __init__(self, predictor, compute_units="ALL"):
        import coremltools as ct

        size = predictor.image_size
        path = model_path(size)
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing: run  python -m compass.coreml_encoder {size}")
        self.ml = ct.models.MLModel(str(path), compute_units=getattr(ct.ComputeUnit, compute_units))
        self.device = predictor.device
        with torch.inference_mode():  # positional encodings: the same for every picture of this size
            out = predictor.forward_image(torch.zeros(1, 3, size, size, device=self.device))
        self.pos = [p.clone() for p in out["vision_pos_enc"]]
        self.rgb = None

    def set_image(self, rgb_1024):
        self.rgb = rgb_1024

    def forward_image(self, img_batch=None):
        out = self.ml.predict({"image": Image.fromarray(self.rgb)})
        fpn = [torch.from_numpy(np.asarray(out[k], dtype=np.float32)).to(self.device) for k in ("fpn0", "fpn1", "fpn2")]
        return {"vision_features": fpn[2], "vision_pos_enc": list(self.pos), "backbone_fpn": fpn}


if __name__ == "__main__":
    export(int(sys.argv[1]) if len(sys.argv) > 1 else 1024)
