"""Why does the object look different? blocked / turned / closer-farther / something else (plan E9).

Keeps the object's last confirmed view (image + outline). For a new frame it:
  1. estimates how the object moved since then: rotation, scale and shift, from matching small details (ORB
     keypoints + RANSAC similarity fit). If too few details match (plain object, other side up), it assumes the
     outline moved with the tracker's mask;
  2. predicts the EXPECTED outline (last outline moved that way) and its expected look;
  3. measures:
       coverage       share of the expected outline the tracker's mask still covers
       missing part   what is now where the object should be but the mask isn't: the object's old look (it is
                      still there), background (it moved/turned away), a detected hand, or something new
       visible match  do the visible pixels still look like the same pixels before (after the motion)?
       rotation/scale from step 1
  4. decides:
       closer/farther  details match at a scale >= 1.25x / <= 0.8x, the outline is mostly covered, and the mask's
                       area changed accordingly (~ scale squared)
       blocked         part missing, the visible part matches, and something covers the missing part: a detected
                       hand, or the area just outside the missing part changed since the reference view (a blocker
                       reaches beyond the object; a turned or flipped object leaves its surroundings alone)
       turned          outline mostly intact and details match at an angle >= 12 deg, or the mask stayed in place
                       with no sign of a blocker: >= 75% of the outline, or what is visible lies inside the old
                       outline (mid-flip: no details to match)
       same            it looks like it did a moment ago (only different from the registration view)
       something_else  the mask is mostly away from where the object should be
       (mask empty)    hand over the spot -> "blocked" by a hand; something new on the spot -> "blocked";
                       background on the spot -> "gone"; the old look still there -> "same" (tracker lost it)
"""
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

SCALE_CLOSER, SCALE_FARTHER, MIN_TURN_DEG = 1.25, 0.8, 12.0
MIN_DETAILS = 10  # matched details needed before a rotation / scale estimate is trusted
SAME_COLOUR, BACKGROUND_COLOUR = 18.0, 15.0  # Lab distances


@dataclass
class Explanation:
    label: str  # same | blocked | turned | closer | farther | something_else | gone | unsure
    visible_fraction: float = 0.0
    angle: float | None = None  # degrees, counter-clockwise in the picture
    scale: float | None = None
    blocker: str | None = None  # "hand" | "something"
    matched_details: int = 0
    missing: dict = field(default_factory=dict)  # shares of the missing part: object / background / hand / new
    expected: np.ndarray | None = None

    def summary(self):
        d = {"label": self.label, "visible": round(self.visible_fraction, 2), "details": self.matched_details,
             "missing": {k: round(float(v), 2) for k, v in self.missing.items()}}
        if self.angle is not None:
            d["angle"] = round(self.angle, 1)
            d["scale"] = round(self.scale, 2)
        if self.blocker:
            d["blocker"] = self.blocker
        return d


def _dilate(mask, r):
    return cv2.dilate(mask.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0


class Explainer:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=600, fastThreshold=10)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.img = self.mask = None

    def remember(self, img, mask):
        """Store the object's last confirmed view. Works in a window around the object (whole-frame processing
        cost ~37-134 ms per frame at 1280x720)."""
        self.img, self.mask = img, mask  # callers pass a fresh frame each time; no copy needed
        ys, xs = np.nonzero(mask)
        self.size = max(int(np.ptp(xs)), int(np.ptp(ys)), 8) if len(xs) else 8
        h, w = mask.shape
        pad = max(64, self.size)  # room for ORB's coarser pyramid levels (needed to match a resized object)
        y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, h)
        x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, w)
        crop, cmask = img[y0:y1, x0:x1], mask[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        kp, self.des = self.orb.detectAndCompute(gray, _dilate(cmask, 4).astype(np.uint8) * 255)
        self.kp = [cv2.KeyPoint(k.pt[0] + x0, k.pt[1] + y0, k.size, k.angle, k.response, k.octave, k.class_id)
                   for k in kp]
        # what was around the object then (a blocker that arrives later is not part of it)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        ring = _dilate(cmask, 30) & ~_dilate(cmask, 8)
        colours = lab[ring]
        if len(colours) > 1500:
            colours = colours[np.random.default_rng(0).choice(len(colours), 1500, replace=False)]
        self.background = colours

    def detail_matches(self, img, mask):
        """How many of the remembered view's small details (ORB keypoints) reappear in this candidate with a
        consistent geometry (RANSAC similarity inliers). An identity check: a "T" block matches the T, not an "E"."""
        if getattr(self, "des", None) is None:
            return 0
        _, n = self._motion(img, _dilate(mask, 6))
        return n

    def _motion(self, img, search):
        """Similarity transform (2x3) from the remembered view to this frame, and how many details agree."""
        if self.des is None or len(self.kp) < 6:
            return None, 0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp2, des2 = self.orb.detectAndCompute(gray, search.astype(np.uint8) * 255)
        if des2 is None or len(kp2) < 6:
            return None, 0
        good = [m for m, n in (p for p in self.matcher.knnMatch(self.des, des2, k=2) if len(p) == 2)
                if m.distance < 0.8 * n.distance]
        if len(good) < 6:
            return None, len(good)
        src = np.float32([self.kp[m.queryIdx].pt for m in good])
        dst = np.float32([kp2[m.trainIdx].pt for m in good])
        M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0)
        n = int(inliers.sum()) if inliers is not None else 0
        if M is None or n < MIN_DETAILS:
            return None, n
        return M, n

    def _missing_content(self, img, old_look, missing, hand_mask):
        """Shares of the missing part that look like: the object's old look / what was around the object before /
        a detected hand / something new."""
        n = int(missing.sum())
        if n == 0:
            return {}
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        old = cv2.cvtColor(old_look, cv2.COLOR_BGR2LAB).astype(np.float32)
        hand_share = float((missing & hand_mask).sum() / n)
        rest = missing & ~hand_mask
        ys, xs = np.nonzero(rest)
        if len(ys) > 2000:  # estimate the shares from a sample
            pick = np.random.default_rng(0).choice(len(ys), 2000, replace=False)
            ys, xs = ys[pick], xs[pick]
        px, before = lab[ys, xs], old[ys, xs]
        same = np.linalg.norm(px - before, axis=-1) < SAME_COLOUR
        background = np.zeros(len(px), bool)
        if len(self.background) and len(px):
            nearest = np.min(np.linalg.norm(px[:, None, :] - self.background[None, :, :], axis=-1), axis=1)
            background = ~same & (nearest < BACKGROUND_COLOUR)
        rest_share = 1.0 - hand_share
        k = max(len(px), 1)
        return {"object": rest_share * same.sum() / k, "background": rest_share * background.sum() / k,
                "hand": hand_share, "new": rest_share * (~same & ~background).sum() / k}

    def explain(self, img, raw, hand_mask=None):
        if self.mask is None or not self.mask.any():
            return Explanation("unsure")
        h, w = raw.shape
        hand_mask = hand_mask if hand_mask is not None else np.zeros((h, w), bool)
        # where to look for the object's details: around the mask and the last outline; but with no mask, only ask
        # "is it still in its place?" (a wider search found an identical object right next to it)
        search = _dilate(self.mask | raw, int(self.size * 0.75)) if raw.any() else _dilate(self.mask, 15)
        M, n = self._motion(img, search)
        angle = scale = None
        if M is not None:
            expected = cv2.warpAffine(self.mask.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST) > 0
            old_look = cv2.warpAffine(self.img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            scale = float(math.hypot(M[0, 0], M[1, 0]))
            angle = float(-math.degrees(math.atan2(M[1, 0], M[0, 0])))
        elif raw.any() and (raw & self.mask).sum() >= 0.7 * raw.sum():
            # too few details to trust a motion estimate, and the visible part sits where the object was:
            # assume it stayed put (a half-hidden or flipped object)
            expected, old_look = self.mask, self.img
        elif raw.any() and (raw & _dilate(self.mask, int(self.size * 0.5))).any():
            # too few details: assume the outline moved with the tracker's mask
            (ys, xs), (ys0, xs0) = np.nonzero(raw), np.nonzero(self.mask)
            shift = np.float32([[1, 0, xs.mean() - xs0.mean()], [0, 1, ys.mean() - ys0.mean()]])
            expected = cv2.warpAffine(self.mask.astype(np.uint8), shift, (w, h), flags=cv2.INTER_NEAREST) > 0
            old_look = cv2.warpAffine(self.img, shift, (w, h), borderMode=cv2.BORDER_REPLICATE)
        else:
            expected, old_look = self.mask, self.img
        area = max(int(expected.sum()), 1)
        coverage = (raw & expected).sum() / area
        iou = (raw & expected).sum() / max((raw | expected).sum(), 1)
        missing = expected & ~_dilate(raw, 3)
        content = self._missing_content(img, old_look, missing, hand_mask)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        hand_share, new_share = content.get("hand", 0.0), content.get("new", 0.0)
        blocker = "hand" if hand_share >= 0.3 else "something"
        visible = raw & expected
        old = cv2.cvtColor(old_look, cv2.COLOR_BGR2LAB).astype(np.float32)
        visible_match = float((np.linalg.norm(lab - old, axis=-1)[visible] < 22).mean()) if visible.sum() > 20 else 0.0
        # has anything changed just outside the missing part since the reference view? (the camera is still, so
        # compare the same pixels; outlines of the object then and now are left out)
        around = _dilate(missing, 25) & ~_dilate(expected | raw | self.mask, 6) & ~hand_mask
        ring_changed = 0.0
        if around.sum() > 30:  # smoothed first, so pixel noise doesn't count as change
            now = cv2.cvtColor(cv2.GaussianBlur(img, (7, 7), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
            ref = cv2.cvtColor(cv2.GaussianBlur(self.img, (7, 7), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
            ring_changed = float((np.linalg.norm(now[around] - ref[around], axis=-1) > SAME_COLOUR).mean())
        content["around_changed"] = ring_changed
        blocker_seen = hand_share >= 0.3 or ring_changed >= 0.15 or new_share >= 0.4
        ex = Explanation("unsure", float(coverage), angle, scale, None, n, content, expected)
        # a scale estimate must agree with how much the mask's area changed (area ~ scale^2); a few matched
        # details on a mask that spread onto something else can otherwise look like "farther"
        area_ratio = raw.sum() / max(int(self.mask.sum()), 1)
        scale_fits_area = M is not None and 0.6 <= area_ratio / scale ** 2 <= 1.6

        if not raw.any():
            if hand_share >= 0.3:
                ex.label, ex.blocker = "blocked", "hand"
            elif content.get("object", 0) >= 0.5:
                ex.label = "same"  # it is still there; the tracker lost it
            elif new_share >= 0.4 or ring_changed >= 0.15:
                ex.label, ex.blocker = "blocked", "something"
            elif content.get("background", 0) >= 0.5:
                ex.label = "gone"
            return ex
        if coverage >= 0.85 and visible_match >= 0.7 and (M is None or (abs(angle) < MIN_TURN_DEG
                                                                         and SCALE_FARTHER < scale < SCALE_CLOSER)):
            ex.label = "same"  # looks like it did a moment ago (it differs from the registration, e.g. lighting)
        elif scale_fits_area and coverage >= 0.6 and scale >= SCALE_CLOSER:
            ex.label = "closer"
        elif scale_fits_area and coverage >= 0.6 and scale <= SCALE_FARTHER:
            ex.label = "farther"
        elif coverage < 0.85 and blocker_seen and visible_match >= 0.4:
            ex.label, ex.blocker = "blocked", blocker
        elif M is not None and coverage >= 0.6 and abs(angle) >= MIN_TURN_DEG:
            ex.label = "turned"
        elif not blocker_seen and ((iou >= 0.4 and coverage >= 0.75)
                                   or (coverage >= 0.3 and (raw & expected).sum() >= 0.85 * raw.sum())):
            # same place, nothing blocking, but it looks different (another angle or side): either the outline is
            # mostly intact, or what is visible lies inside where the object was (mid-flip, standing on an edge)
            ex.label = "turned"
        elif iou < 0.2 and coverage < 0.2:
            ex.label = "something_else"  # the mask is away from where the object should be
            if content.get("object", 0) >= 0.5:
                ex.blocker = None
                ex.missing["still_there"] = 1.0  # ...and the object is still in its place
        return ex
