"""Where is the object? Visible / covered / held by a hand / out of view, plus the hand-off rule
(validated on drawn scenes in experiment E2b: 335/336 frames, 0 on a lookalike, 8/8 final answers).

States per frame:
  visible      the tracker's mask passes the appearance check (compass.appearance.Gate)
  unsure       the tracker sees something object-sized but it doesn't look like the object (e.g. turned over);
               shown as a grey outline, not claimed as "your object"
  covered      the mask vanished without growing first (something on top)
  held         the mask grew onto something bigger: a hand took the object. The tracker then follows the hand
               (EdgeTAM drifts onto it by itself), and the hand stands in for the hidden object.
  out_of_view  the object, or the hand holding it, left the picture at an edge
  unknown      the hand was lost mid-picture for too long

Hand-off (held -> visible), on held frames: object-looking things (DINOv2 similarity peaks, each outlined by
SAM) where the hand is or was in its last 6 frames, excluding lookalikes already seen elsewhere (at
registration or when the hand took the object), confirmed by the appearance check -> restart the tracker there.

Re-detection (covered / out_of_view / unknown): the same search over the whole picture. Needed live: at ~4 fps
the tracker often loses an object that a hand grabs (instead of drifting onto the hand), so the state is "covered";
without this, an object put down elsewhere was never found again (measured on a live replay). Lookalikes visible
elsewhere when the object disappeared are ignored. With no hand path as evidence, a find is only trusted when no
lookalikes are known; otherwise it is reported as "maybe here" (a live replay of a table of letter blocks accepted
a different blue block at similarity 0.88, about as similar as the real block after it was turned).

Limits: the hand is followed via the tracker's own drift, not a hand detector; lookalikes are assumed to stay
put while the object is held.
"""
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
import timm
import torch

from compass.appearance import Gate
from compass.models import image_predictor

HOLDER_MEMORY, HOLDER_REACH, LOST_LIMIT, BORDER = 6, 40, 15, 12


class Finder:
    """Object-looking things via DINOv2 (ViT-S/14) patch similarity to the registered object's patches."""

    GRID = 32  # 448 px / 14 px patches
    MAX_PEAKS = 10  # when searching, outline at most the 10 strongest peaks (each costs a SAM decode)
    MAX_PEAKS_RECORD = 40  # when recording which lookalikes are where, be more complete

    def __init__(self, device):
        self.device = device
        self.model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0,
                                       img_size=448).eval().to(device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    @torch.no_grad()
    def tokens(self, img_bgr):
        rgb = cv2.cvtColor(cv2.resize(img_bgr, (448, 448), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = (torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).float() / 255 - self.mean) / self.std
        t = self.model.forward_features(x[None])[0, self.model.num_prefix_tokens:]
        t = t / t.norm(dim=-1, keepdim=True)
        return t.view(self.GRID, self.GRID, -1)

    def register(self, img0, mask0):
        """Prototype = mean patch token inside the object; a peak must reach 60% of the way from the
        background's typical similarity to the object's."""
        tok = self.tokens(img0)
        self._cached = (img0, tok)
        inside = cv2.resize(mask0.astype(np.float32), (self.GRID, self.GRID), interpolation=cv2.INTER_AREA) > 0.3
        outside = ~(cv2.dilate(inside.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
        proto = tok[torch.from_numpy(inside).to(self.device)].mean(0)
        self.proto = proto / proto.norm()
        sim = (tok @ self.proto).cpu().numpy()
        bg, obj = np.median(sim[outside]), np.median(sim[inside])
        self.thr = float(bg + 0.6 * (obj - bg))
        self.ref_area = int(mask0.sum())

    def candidates(self, img, segmenter, max_peaks=None):
        """Masks of separate object-looking things: one SAM mask per local peak of the similarity map."""
        tok = self._cached[1] if getattr(self, "_cached", (None,))[0] is img else self.tokens(img)
        sim = (tok @ self.proto).cpu().numpy()
        peaks = (sim == cv2.dilate(sim, np.ones((3, 3), np.uint8))) & (sim > self.thr)
        h, w = img.shape[:2]
        masks = []
        for gy, gx in sorted(zip(*np.nonzero(peaks)), key=lambda p: -sim[p])[:max_peaks or self.MAX_PEAKS]:
            x, y = (gx + 0.5) * w / self.GRID, (gy + 0.5) * h / self.GRID
            if any(m[int(y), int(x)] for m in masks):
                continue  # this peak lies on an object already outlined
            m = segmenter.segment_point(img, (x, y))
            if 0.25 * self.ref_area <= m.sum() <= 3 * self.ref_area:
                masks.append(m)
        return masks


class Segmenter:
    """SAM (EdgeTAM's image model) outlining the object under a point."""

    def __init__(self, device):
        self.pred = image_predictor(device)
        self.last_image = None

    def segment_point(self, img, xy):
        if self.last_image is not img:
            self.pred.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            self.last_image = img
        masks, _, _ = self.pred.predict(point_coords=np.array([xy], dtype=np.float32), point_labels=np.array([1]),
                                        multimask_output=False)
        return masks[0] > 0


def box_of(mask):
    ys, xs = np.nonzero(mask)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def mask_iou(a, b):
    union = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / union if union else 0.0


def exit_side(mask):
    """Which picture edge a mask was nearest to when it disappeared (None if not at an edge)."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    cx, cy = xs.mean(), ys.mean()
    touching = {"left": xs.min() <= BORDER, "right": xs.max() >= w - 1 - BORDER,
                "top": ys.min() <= BORDER, "bottom": ys.max() >= h - 1 - BORDER}
    dist = {"left": cx, "right": w - 1 - cx, "top": cy, "bottom": h - 1 - cy}
    sides = [s for s in dist if touching[s] and dist[s] < 100]
    return min(sides, key=dist.get) if sides else None


@dataclass
class Step:
    t: int
    mask: np.ndarray  # what to show as "your object" (empty if it can't be seen)
    raw: np.ndarray  # the tracker's own mask
    score: float  # the tracker's own visibility logit
    state: str  # visible | unsure | covered | held | out_of_view | unknown
    detail: str | None = None  # exit side for out_of_view
    holder: np.ndarray | None = None  # the hand's mask while held
    events: list = field(default_factory=list)  # (t, text) that happened on this frame
    debug: dict = field(default_factory=dict)  # why: check readings, hand-off search results
    maybe: np.ndarray | None = None  # possible sighting while lost, not trusted (similar objects around)

    def describe(self, obj="object"):
        if self.maybe is not None:
            return "lost - maybe here (grey), but similar objects are around"
        if self.state == "visible":
            return f"your {obj} is visible"
        if self.state == "out_of_view":
            return f"carried out of view to the {self.detail}"
        return {"covered": "covered - last seen here", "held": "in a hand - following the hand",
                "unknown": "lost track of the hand",
                "unsure": "not sure - something here looks different (turned?)"}[self.state]


class ObjectState:
    """Runs the tracker + appearance check + holder states for one registered object, frame by frame.

    tracker: compass.tracker.StreamingTracker (start / track / reinit).
    finder_every: when the object is held or lost, search for it only every N-th frame (the search costs ~200 ms
    on an M2; 1 = every frame)."""

    def __init__(self, tracker, appearance, finder, segmenter, finder_every=1):
        self.trk, self.app, self.finder, self.seg = tracker, appearance, finder, segmenter
        self.finder_every = finder_every
        self.events = []

    def start(self, img, box):
        raw, score = self.trk.start(img, box)
        self.gate = Gate(self.app, img, raw)
        self.finder.register(img, raw)
        self.seg.last_image = None
        self.lookalikes0 = [m for m in self.finder.candidates(img, self.seg, Finder.MAX_PEAKS_RECORD)
                            if not np.logical_and(m, raw).any()]
        self.state, self.detail, self.last_visible = "visible", None, raw
        self.holders, self.known_others, self.lost_for, self.held_frames = deque(maxlen=HOLDER_MEMORY), list(self.lookalikes0), 0, 0
        self.lost_frames, self.maybe_at = 0, None
        self.t = 0
        self.events = []
        return Step(0, raw, raw, score, "visible")

    def update(self, img):
        self.t += 1
        t = self.t
        raw, score = self.trk.track(img)
        out, holder, new, maybe = np.zeros_like(raw), None, [], None
        if raw.any():
            show, _, info = self.gate.check(img, raw)
            kind = info["kind"]
        else:
            show, kind, info = False, "empty", {}
        debug = {"check": {k: info[k] for k in ("kind", "sim", "area") if k in info},
                 "typical_area": float(np.median(self.gate.areas))}

        if self.state != "held":
            if show:
                if self.state not in ("visible", "unsure"):
                    new.append("your object is visible again")
                self.state, self.detail, out, self.last_visible = "visible", None, raw, raw
            elif kind == "grown":
                self.state, self.detail, holder = "held", None, raw
                self.holders.clear()
                self.holders.append(raw)
                self.lost_for, self.held_frames = 0, 0
                self.known_others = list(self.lookalikes0) + [
                    m for m in self.finder.candidates(img, self.seg, Finder.MAX_PEAKS_RECORD) if not np.logical_and(m, raw).any()]
                new.append(f"taken by a hand ({len(self.known_others)} similar object(s) seen elsewhere, ignored)")
            elif kind == "empty" and self.state in ("visible", "unsure"):
                self.lost_frames, self.maybe_at = 0, None
                near = cv2.dilate(self.last_visible.astype(np.uint8), np.ones((2 * HOLDER_REACH + 1,) * 2, np.uint8)) > 0
                self.known_others = list(self.lookalikes0) + [  # lookalikes visible elsewhere right now
                    m for m in self.finder.candidates(img, self.seg, Finder.MAX_PEAKS_RECORD) if not np.logical_and(m, near).any()]
                side = exit_side(self.last_visible)
                if side:
                    self.state, self.detail = "out_of_view", side
                    new.append(f"left the picture to the {side}")
                else:
                    self.state = "covered"
                    new.append("covered (disappeared without being picked up)")
            elif kind in ("normal", "partial"):  # object-sized, but doesn't look like the object
                self.state = "unsure"
            if not show and not new and self.state in ("covered", "out_of_view", "unknown"):
                self.lost_frames += 1
                searched = self.lost_frames % self.finder_every == 0
                found = self._search(img, debug) if searched else None
                debug["re_detect_searched"] = searched
                cx, cy = box_of(found).reshape(2, 2).mean(0) if found is not None else (0, 0)
                if found is not None and not self.known_others:
                    out = self.trk.reinit(box_of(found))
                    self.state, self.detail, self.last_visible = "visible", None, out
                    new.append(f"found your object again at ({cx:.0f}, {cy:.0f})")
                elif found is not None:  # similar objects around and no hand path: don't claim it
                    maybe = found
                    if self.maybe_at is None or abs(self.maybe_at[0] - cx) + abs(self.maybe_at[1] - cy) > 40:
                        new.append(f"maybe your object at ({cx:.0f}, {cy:.0f}) - similar objects around, not sure")
                    self.maybe_at = (cx, cy)
        else:
            self.held_frames += 1
            if show:  # the tracker found its way back to the object itself
                self.state, out, self.last_visible = "visible", raw, raw
                new.append("your object is visible again (tracker)")
            else:
                if raw.any():
                    holder = raw
                    self.holders.append(raw)
                    self.lost_for = 0
                else:
                    self.lost_for += 1
                searched = self.held_frames % self.finder_every == 0
                found = self._search(img, debug, near=self._reach()) if searched else None
                debug["hand_off_searched"] = searched
                if found is not None:
                    out = self.trk.reinit(box_of(found))  # restart the tracker on the object
                    self.state, self.last_visible = "visible", out
                    cx, cy = box_of(found).reshape(2, 2).mean(0)
                    new.append(f"your object seen again where the hand is or was, at ({cx:.0f}, {cy:.0f})")
                elif not raw.any():
                    side = exit_side(self.holders[-1])
                    if side:
                        self.state, self.detail = "out_of_view", side
                        new.append(f"carried out of view to the {side}")
                    elif self.lost_for > LOST_LIMIT:
                        self.state = "unknown"
                        new.append("lost track of the hand")
        events = [(t, e) for e in new]
        self.events += events
        if self.state not in ("covered", "out_of_view", "unknown"):
            self.maybe_at = None
        return Step(t, out, raw, score, self.state, self.detail, holder, events, debug, maybe)

    def _reach(self):
        """Where the hand is or was in its last frames (+40 px)."""
        return cv2.dilate(np.logical_or.reduce(list(self.holders)).astype(np.uint8),
                          np.ones((2 * HOLDER_REACH + 1,) * 2, np.uint8)) > 0

    def _search(self, img, debug=None, near=None):
        """Object-looking candidate that is not a known lookalike, (if `near`) overlaps it, and passes the check."""
        log = []
        for m in self.finder.candidates(img, self.seg):
            ys, xs = np.nonzero(m)
            entry = {"at": [int(xs.mean()), int(ys.mean())], "area": int(m.sum())}
            if any(mask_iou(m, k) > 0.3 for k in self.known_others):
                entry["rejected"] = "known lookalike"
            elif near is not None and not np.logical_and(m, near).any():
                entry["rejected"] = "not near the hand"
            else:
                entry["sim"] = round(self.gate.similarity(img, m)[0], 3)
                if self.gate.confirm(img, m):
                    log.append(entry)
                    if debug is not None:
                        debug["search_candidates"] = log
                    return m
                entry["rejected"] = "does not look like the object"
            log.append(entry)
        if debug is not None:
            debug["search_candidates"] = log
        return None
