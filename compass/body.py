"""Where the person's clothes are, per pixel: MediaPipe's selfie multiclass segmenter (Google, Apache-2.0).

Used to tell "the object went into a pocket" from other disappearances: a pocket is part of the clothes. Body pose was
tried first (hips would give left/right pockets), but MediaPipe's pose detector needs a face: in a clip where the
camera saw only torso and legs it put "shoulders" and "hips" on the table. The segmenter works per pixel and does
not need a face; it labels shirt and shorts as clothes (also a patterned tablecloth, which is fabric too, so callers
only use it while a hand is holding the object).

Classes: 0 background, 1 hair, 2 body skin, 3 face skin, 4 clothes, 5 other (accessories). Runs at 640 px wide,
~120 ms on an M2 CPU, so callers run it only when needed.
"""
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from compass.models import ROOT

MODEL = ROOT / "third_party" / "models" / "selfie_multiclass_256x256.tflite"
CLOTHES = 4


class ClothesSegmenter:
    WIDTH = 640
    MAX_AGE_MS = 1000  # reuse the last mask for up to 1 s of video: clothes barely move in that time, and each run
    # costs ~200 ms (measured on the E0 clips: it ran on 27% of in-hand frames, 56 ms per in-hand frame on average)

    def __init__(self):
        if not MODEL.exists():
            raise FileNotFoundError(f"{MODEL} is missing: run  bash scripts/setup_third_party.sh")
        self.seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL)), running_mode=vision.RunningMode.IMAGE,
            output_category_mask=True))
        self._last = (None, None)
        self._last_t = None

    def clothes(self, img_bgr, t_ms=None):
        """Bool mask (full size) of clothes; cached for the same frame, and for up to MAX_AGE_MS if `t_ms` is given."""
        if self._last[0] is img_bgr:
            return self._last[1]
        if (t_ms is not None and self._last_t is not None and self._last[1] is not None
                and 0 <= t_ms - self._last_t <= self.MAX_AGE_MS and self._last[1].shape == img_bgr.shape[:2]):
            return self._last[1]
        h, w = img_bgr.shape[:2]
        small = cv2.resize(img_bgr, (self.WIDTH, round(h * self.WIDTH / w)), interpolation=cv2.INTER_AREA)
        cat = self.seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        mask = cv2.resize((cat.category_mask.numpy_view().squeeze() == CLOTHES).astype(np.uint8), (w, h),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
        self._last, self._last_t = (img_bgr, mask), t_ms
        return mask

    def close(self):
        self.seg.close()
