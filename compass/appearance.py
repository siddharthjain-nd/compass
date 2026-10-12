"""The "does the tracker's mask still look like my object?" check (validated in experiment E2, variant B).

Compares each frame's mask with the object's known looks, using size and a DINOv2 (ViT-S/14) embedding of the
masked crop. No training and no ground truth.

Known looks (the "bank"): the registration view plus rotated and mirrored copies of it (turning an object by
30-45 degrees drops similarity to ~0.8, measured on a real block and a drawn wallet), plus new views learned while
the object is confidently tracked. Similarity = best match in the bank.

Rule ("typical area" = median of recently accepted areas):
  grown    area > 1.5x typical                       -> don't show (it has spread onto a holder)
  partial  area < 0.7x typical (object partly hidden) -> show if similarity >= 0.75, never remember
  normal                                              -> show and remember if similarity >= 0.85
A shown normal view that is new (best match < 0.95) is added to the bank (up to 48 learned views).

History: v1 calibrated "typical similarity" on the first 5 checked frames; in live use those frames can already
show the object half-hidden in a hand (a real replay measured 0.15), which corrupted the calibration. The bank
makes thresholds absolute against the registration view instead. The 0.75 partial threshold was set on drawn
scenes; real clips must confirm it.
"""
from collections import deque

import cv2
import numpy as np
import timm
import torch

AREA_GROWTH, AREA_PARTIAL, SIM_NORMAL, SIM_PARTIAL = 1.5, 0.7, 0.85, 0.75
NEW_VIEW, MAX_LEARNED = 0.95, 48
REGISTRATION_ANGLES = (0, 15, -15, 30, -30, 45, -45, 90, 180, 270)


class Appearance:
    """DINOv2 ViT-S/14 embedding of the masked object crop (background greyed out)."""

    def __init__(self, device):
        self.device = device
        self.model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0,
                                       img_size=224).eval().to(device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    @staticmethod
    def _crop(img_bgr, mask):
        ys, xs = np.nonzero(mask)
        y0, y1, x0, x1 = max(ys.min() - 8, 0), ys.max() + 9, max(xs.min() - 8, 0), xs.max() + 9
        crop = img_bgr[y0:y1, x0:x1].copy()
        crop[~mask[y0:y1, x0:x1]] = 128
        return cv2.cvtColor(cv2.resize(crop, (224, 224)), cv2.COLOR_BGR2RGB)

    @torch.no_grad()
    def _embed_crops(self, crops):
        x = torch.from_numpy(np.stack(crops)).to(self.device).permute(0, 3, 1, 2).float() / 255
        e = self.model((x - self.mean) / self.std)
        return e / e.norm(dim=-1, keepdim=True)

    def embed(self, img_bgr, mask):
        return self._embed_crops([self._crop(img_bgr, mask)])[0]

    def embed_masks(self, img_bgr, masks):
        """Embeddings of several masked crops of one frame, in one batch."""
        return self._embed_crops([self._crop(img_bgr, m) for m in masks])

    def embed_boxes(self, img_bgr, boxes):
        """Embeddings of plain box crops (no mask), in one batch: used to check whether a spot looks like the object."""
        crops = [cv2.cvtColor(cv2.resize(img_bgr[y0:y1, x0:x1], (224, 224)), cv2.COLOR_BGR2RGB)
                 for x0, y0, x1, y1 in boxes]
        return self._embed_crops(crops)

    def embed_views(self, img_bgr, mask, angles=REGISTRATION_ANGLES, mirror=True):
        """Embeddings of the object rotated by each angle (about its centre), plus a mirrored copy, in one batch."""
        ys, xs = np.nonzero(mask)
        centre = (float(xs.mean()), float(ys.mean()))
        size = img_bgr.shape[1::-1]
        crops = []
        for a in angles:
            m = cv2.getRotationMatrix2D(centre, a, 1.0)
            img_r = cv2.warpAffine(img_bgr, m, size, borderMode=cv2.BORDER_REFLECT)
            mask_r = cv2.warpAffine(mask.astype(np.uint8), m, size, flags=cv2.INTER_NEAREST) > 0
            crops.append(self._crop(img_r, mask_r))
        if mirror:
            crops.append(self._crop(img_bgr[:, ::-1].copy(), mask[:, ::-1].copy()))
        return self._embed_crops(crops)


class Gate:
    """Decides per frame whether the mask still looks like the registered object."""

    def __init__(self, appearance, img0, mask0):
        self.app = appearance
        self.registered = appearance.embed_views(img0, mask0)  # registration view + rotated/mirrored copies
        self.learned = deque(maxlen=MAX_LEARNED)
        self.areas = deque([int(mask0.sum())], maxlen=10)

    def _bank(self):
        return torch.cat([self.registered, torch.stack(list(self.learned))]) if self.learned else self.registered

    def similarity(self, img, mask):
        """Best match against all known looks; also returns the embedding."""
        e = self.app.embed(img, mask)
        return float((self._bank() @ e).max()), e

    def similarities(self, img, masks):
        """Best match against all known looks for several masks of one frame, in one batch."""
        if not masks:
            return []
        return [float(s) for s in (self.app.embed_masks(img, masks) @ self._bank().T).max(dim=1).values]

    def check(self, img, mask):
        """Returns (show, remember, info)."""
        area = int(mask.sum())
        sim, e = self.similarity(img, mask)
        typical_area = float(np.median(self.areas))
        if area > AREA_GROWTH * typical_area:
            kind, show = "grown", False
        elif area < AREA_PARTIAL * typical_area:
            kind, show = "partial", sim >= SIM_PARTIAL
        else:
            kind, show = "normal", sim >= SIM_NORMAL
        remember = show and kind == "normal"
        learned = False
        if remember:
            self.areas.append(area)
            if sim < NEW_VIEW:  # a new angle of the object: learn it
                self.learned.append(e)
                learned = True
        return show, remember, {"area": area, "sim": round(sim, 3), "kind": kind, "show": show,
                                "remember": remember, "learned_view": learned}

    def learn(self, img, mask):
        """Add this view to the known looks if it is new (used when another cue says it IS the object, e.g. it was
        seen turning). Also counts its size as normal."""
        sim, e = self.similarity(img, mask)
        self.areas.append(int(mask.sum()))
        if sim < NEW_VIEW:
            self.learned.append(e)

    def confirm(self, img, mask, sim=None):
        """Stateless version for candidates: object-sized and looks like the object (partial views allowed).
        `sim`: the similarity if already computed."""
        area = int(mask.sum())
        typical_area = float(np.median(self.areas))
        if not 0.25 * typical_area <= area <= AREA_GROWTH * typical_area:
            return False
        return (self.similarity(img, mask)[0] if sim is None else sim) >= SIM_PARTIAL
