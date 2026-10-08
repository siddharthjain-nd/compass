"""Where is the object? Visible / covered / held by a hand / out of view, plus the hand-off rule
(validated on drawn scenes in experiment E2b: 335/336 frames, 0 on a lookalike, 8/8 final answers).

States per frame:
  visible      the tracker's mask passes the appearance check (compass.appearance.Gate), or the explainer
               (compass.explain) says why it looks different: turned, moved closer/farther, partly hidden
               behind a hand / something (then only the visible part is shown). A note says which.
  unsure       the tracker sees something object-sized that doesn't look like the object and isn't explained
               (e.g. the tracker may have jumped to something else); grey outline, not claimed as "your object"
  covered      the mask vanished without growing first (something on top)
  held         a hand took the object: with the hand detector (compass.hands), the object disappeared under or
               inside a detected hand, or its mask spread onto one; that hand is then followed. Without hand
               detection, the mask spreading onto something bigger is used and the tracker's drift is followed.
  out_of_view  the object, or the hand holding it, left the picture at an edge
  unknown      the hand was lost mid-picture for too long

Hand-off (held -> visible), on held frames: object-looking things (DINOv2 similarity peaks, each outlined by
SAM) where the hand is or was in its last 6 frames, excluding lookalikes already seen elsewhere (at
registration or when the hand took the object), confirmed by the appearance check -> restart the tracker there.
Lookalikes are remembered as spots (peak positions), not outlines: outlining up to 40 of them with SAM took
0.7-1 s, at start-up and again at the moment a hand grabbed the object.

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
from compass.explain import Explainer
from compass.models import image_predictor

HOLDER_MEMORY, HOLDER_REACH, LOST_LIMIT, BORDER = 6, 40, 15, 12
LOOKALIKE_SIM = 0.75  # a spot counts as a lookalike only if its box crop looks this much like the object's
DETAIL_MIN, CLUTTER = 12, 5  # details needed to decide by detail matching; lookalike spots that make a scene "cluttered"
IN_HAND_SIM = 0.45  # in a detected hand, show the visible part if it looks at least this much like the object
IN_HAND_VISIBLE = 0.3  # less than this share visible + a hand touching it = "in your hand", not "visible"


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

    def spots(self, img, exclude=None):
        """Where object-looking things are: one (x, y) per similarity peak, without outlining them (cheap; used to
        remember where lookalikes lie). Peaks inside `exclude` are skipped."""
        tok = self._cached[1] if getattr(self, "_cached", (None,))[0] is img else self.tokens(img)
        sim = (tok @ self.proto).cpu().numpy()
        peaks = (sim == cv2.dilate(sim, np.ones((3, 3), np.uint8))) & (sim > self.thr)
        h, w = img.shape[:2]
        pts = [((gx + 0.5) * w / self.GRID, (gy + 0.5) * h / self.GRID) for gy, gx in zip(*np.nonzero(peaks))]
        return [(x, y) for x, y in pts if exclude is None or not exclude[int(y), int(x)]]

    def candidates(self, img, segmenter, max_peaks=None):
        """Masks of separate object-looking things: one SAM mask per local peak of the similarity map."""
        tok = self._cached[1] if getattr(self, "_cached", (None,))[0] is img else self.tokens(img)
        sim = (tok @ self.proto).cpu().numpy()
        peaks = (sim == cv2.dilate(sim, np.ones((3, 3), np.uint8))) & (sim > self.thr)
        h, w = img.shape[:2]
        order = sorted(zip(*np.nonzero(peaks)), key=lambda p: -sim[p])[:max_peaks or self.MAX_PEAKS]
        points = [((gx + 0.5) * w / self.GRID, (gy + 0.5) * h / self.GRID) for gy, gx in order]
        masks = []
        for (x, y), m in zip(points, segmenter.segment_points(img, points)):
            if any(k[int(y), int(x)] or mask_iou(k, m) > 0.5 for k in masks):
                continue  # this peak lies on an object already outlined (or outlines the same object again)
            if 0.25 * self.ref_area <= m.sum() <= 3 * self.ref_area:
                masks.append(m)
        return masks


class Segmenter:
    """SAM (EdgeTAM's image model) outlining the objects under points, in one batch, at half resolution (the
    outlines are only used to find candidates; full-size outlines made 40 candidates take ~0.75 s)."""

    WIDTH = 640

    def __init__(self, device):
        self.pred = image_predictor(device)
        self.last_image = None

    def segment_points(self, img, points):
        if not points:
            return []
        h, w = img.shape[:2]
        k = min(1.0, self.WIDTH / w)
        if self.last_image is not img:
            small = cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else img
            self.pred.set_image(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            self.last_image = img
        pts = np.array(points, np.float32)[:, None, :] * k
        masks, _, _ = self.pred.predict(point_coords=pts, point_labels=np.ones((len(points), 1)), multimask_output=False)
        masks = masks.reshape(len(points), *masks.shape[-2:]) > 0
        return [cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool) if k < 1 else m
                for m in masks]

    def segment_point(self, img, xy):
        return self.segment_points(img, [xy])[0]


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
    detail: str | None = None  # exit side for out_of_view (as the camera sees it)
    holder: np.ndarray | None = None  # the hand's mask while held
    events: list = field(default_factory=list)  # (t, text) that happened on this frame
    debug: dict = field(default_factory=dict)  # why: check readings, explanation, search results
    maybe: np.ndarray | None = None  # possible sighting while lost, not trusted (similar objects around)
    note: str | None = None  # e.g. "turned ~40 deg", "partly hidden by your hand (45% visible)"
    hands: list = field(default_factory=list)  # hands detected on this frame (compass.hands.Hand)

    def describe(self, obj="object"):
        if self.maybe is not None and self.state == "held":
            return f"{self.note or 'in a hand'} - maybe put down here (grey), similar objects around"
        if self.maybe is not None:
            return "lost - maybe here (grey), but similar objects are around"
        if self.state == "visible":
            return f"your {obj} is visible" + (f" - {self.note}" if self.note else "")
        if self.state == "out_of_view":
            return f"carried out of view to the {self.detail}"
        default = {"covered": "covered - last seen here", "held": "in a hand - following the hand",
                   "unknown": "lost track", "unsure": "not sure - something here looks different"}[self.state]
        return self.note or default


def _hand_name(hand):
    return f"your {hand.side} hand" if hand is not None and hand.side in ("left", "right") else "a hand"


class ObjectState:
    """Runs the tracker + appearance check + explainer + holder states for one registered object, frame by frame.

    tracker: compass.tracker.StreamingTracker (start / track / reinit).
    hands: compass.hands.HandDetector, or None (no hand detection: the drawn test scenes have no real hands).
    finder_every: when the object is held or lost, search for it only every N-th frame (the search costs ~200 ms
    on an M2; 1 = every frame)."""

    def __init__(self, tracker, appearance, finder, segmenter, finder_every=1, hands=None):
        self.trk, self.app, self.finder, self.seg = tracker, appearance, finder, segmenter
        self.finder_every = finder_every
        self.hand_detector = hands
        self.events = []

    def start(self, img, box, t_ms=None):
        raw, score = self.trk.start(img, box)
        self.gate = Gate(self.app, img, raw)
        self.explainer = Explainer()
        self.explainer.remember(img, raw)
        self.finder.register(img, raw)
        self.seg.last_image = None
        ys, xs = np.nonzero(raw)
        self.obj_box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        self.ref_box = self.app.embed_boxes(img, [self.obj_box])[0]
        self.lookalikes0 = self._lookalike_spots(img, _dilate(raw, HOLDER_REACH))  # (x, y) of similar things
        self.state, self.detail, self.last_visible = "visible", None, raw
        self.holders, self.known_others, self.lost_for, self.held_frames = deque(maxlen=HOLDER_MEMORY), list(self.lookalikes0), 0, 0
        self.lost_frames, self.maybe_at, self.holder_hand, self.note = 0, None, None, None
        self.t = 0
        self.events = []
        hands = self.hand_detector.detect(img, t_ms) if self.hand_detector else []
        return Step(0, raw, raw, score, "visible", hands=hands)

    # ------------------------------------------------------------------ per frame

    def update(self, img, t_ms=None):
        self.t += 1
        raw, score = self.trk.track(img)
        hands = self.hand_detector.detect(img, t_ms) if self.hand_detector else []
        hand_union = np.logical_or.reduce([h.mask for h in hands]) if hands else np.zeros_like(raw)
        if raw.any():
            show, _, info = self.gate.check(img, raw)
            kind = info["kind"]
        else:
            show, kind, info = False, "empty", {}
        f = {"raw": raw, "show": show, "kind": kind, "hands": hands, "hand_union": hand_union, "img": img,
             "out": np.zeros_like(raw), "holder": None, "maybe": None, "new": [], "note": None,
             "debug": {"check": {k: info[k] for k in ("kind", "sim", "area") if k in info},
                       "typical_area": float(np.median(self.gate.areas)), "hands": len(hands)}}
        if self.state == "held":
            self._update_held(f)
        else:
            self._update_free(f)
        events = [(self.t, e) for e in f["new"]]
        self.events += events
        if self.state not in ("covered", "out_of_view", "unknown", "held"):
            self.maybe_at = None
        return Step(self.t, f["out"], raw, score, self.state, self.detail, f["holder"], events, f["debug"], f["maybe"],
                    f["note"], hands)

    def _visible(self, f, mask, note=None, learn=False, remember="complete"):
        """Show `mask` as the object. The explainer's reference view is refreshed only from complete views (size
        within 90-115% of normal), unless remember=True (a confirmed new look: turned, moved closer); a view taken
        while something was already sliding over the object made that blocker part of the "background"."""
        self.state, self.detail, self.last_visible = "visible", None, mask
        f["out"], f["note"] = mask, note
        if learn:
            self.gate.learn(f["img"], mask)
        ratio = mask.sum() / max(float(np.median(self.gate.areas)), 1.0)
        if remember is True or (remember == "complete" and 0.9 <= ratio <= 1.15):
            self.explainer.remember(f["img"], mask)

    def _take(self, f, holder, hand):
        """The object was taken: start following its holder (a detected hand, or the tracker's drift)."""
        self.state, self.detail, f["holder"], self.holder_hand = "held", None, holder, hand
        self.take_img = f["img"]  # the picture when the hand took it: anything already lying there was not put down
        self.holders.clear()
        self.holders.append(holder)
        self.lost_for, self.held_frames = 0, 0
        self.known_others = list(self.lookalikes0) + self._lookalike_spots(
            f["img"], _dilate(holder | self.last_visible, HOLDER_REACH), f["debug"])
        f["note"] = f"in {_hand_name(hand)} - following it"
        f["new"].append(f"taken by {_hand_name(hand)} ({len(self.known_others)} similar-looking spot(s) elsewhere, ignored)")

    def _update_free(self, f):
        raw, show, kind, img = f["raw"], f["show"], f["kind"], f["img"]
        on_lookalike = any(raw[int(y), int(x)] for x, y in self.known_others)
        jumped = self.state == "visible" and self.last_visible.any() and mask_iou(raw, self.last_visible) < 0.2
        if show and on_lookalike and (jumped or self.state not in ("visible", "unsure")):
            # the tracker jumped (or came back from "lost") onto a spot where a lookalike was seen: don't claim it.
            # Identical objects pass the look check, so only where they were seen tells them apart.
            show = False
            f["debug"]["tracker_on_known_lookalike"] = True
            if self.state in ("visible", "unsure"):
                self.state = "unsure"
                f["note"] = "not sure - this may be the other, similar object"
                f["new"].append("the tracker moved onto a similar object - not claiming it")
            else:
                f["maybe"], f["note"] = raw, self.note
            return
        hand = self._touching_hand(raw, f["hands"])
        typical = float(np.median(self.gate.areas))
        if show and hand is not None and raw.sum() < IN_HAND_VISIBLE * typical:
            self._in_hand(f, hand)  # only a small part shows, next to a hand: it is in the hand
            return
        if show and raw.sum() < 0.85 * typical:
            # passes the look check but is smaller than usual: say why (partly hidden?), and keep the last full
            # view as the reference (a half view saved as the reference made later covers undetectable)
            ex = self.explainer.explain(img, raw, f["hand_union"])
            f["debug"]["explain"] = ex.summary()
            note = None
            if ex.label == "blocked" or hand is not None:
                who = "your hand" if ex.blocker == "hand" or hand is not None else "something"
                note = f"partly hidden by {who} ({raw.sum() / max(typical, 1):.0%} visible)"
            if self.state not in ("visible", "unsure"):
                f["new"].append("your object is visible again")
            self._visible(f, raw, note)
            return
        if show:
            if self.state not in ("visible", "unsure"):
                f["new"].append("your object is visible again")
            self._visible(f, raw)
            return
        ex = self.explainer.explain(img, raw, f["hand_union"])
        f["debug"]["explain"] = ex.summary()
        if raw.any():
            touching_hands = [h for h in f["hands"] if (raw & _dilate(h.mask, 5)).sum() >= 0.25 * raw.sum()]
            small = raw.sum() < IN_HAND_VISIBLE * typical
            if hand is not None and kind != "grown" and (small or ex.label in ("unsure", "something_else", "blocked")):
                # part of it shows, touching a detected hand, and nothing else explains the look: it is in the hand
                # (from live sessions: 6-35% visible between the fingers, look similarity 0.3-0.7, which the
                # explainer alone left as "not sure" in 17-65% of frames)
                self._in_hand(f, hand)
            elif ex.label in ("closer", "farther"):
                self._visible(f, raw, f"moved {ex.label}", learn=True, remember=True)
            elif kind == "grown" and (touching_hands or not self.hand_detector):
                hand = max(touching_hands, key=lambda h: (h.mask & raw).sum()) if touching_hands else None
                self._take(f, hand.mask if hand else raw, hand)
            elif kind == "grown":
                self._take(f, raw, None)  # spread onto something bigger that is not a detected hand
            elif ex.label == "blocked":
                who = "your hand" if ex.blocker == "hand" or touching_hands else "something"
                self._visible(f, raw, f"partly hidden by {who} ({ex.visible_fraction:.0%} visible)", remember=False)
            elif ex.label == "turned":
                big = ex.angle is not None and abs(ex.angle) >= 12
                turn = f"turned ~{abs(ex.angle):.0f} deg" if big else "turned / other side up"
                self._visible(f, raw, turn, learn=True, remember=True)
            elif ex.label == "same":
                self._visible(f, raw, learn=True)
            elif ex.label == "something_else" and ex.missing.get("still_there") and not self._on_lookalike(ex.expected):
                f["out"] = self.trk.reinit(box_of(ex.expected))  # the tracker jumped; the object is still here
                self._visible(f, f["out"])
                f["new"].append("the tracker had jumped to something else - back on your object")
            else:
                self.state = "unsure"
                f["note"] = "not sure - may be something else" if ex.label == "something_else" else None
            return
        # the mask is empty
        if self.state in ("visible", "unsure"):
            self.lost_frames, self.maybe_at = 0, None
            over = [h for h in f["hands"] if (h.mask & self.last_visible).sum() >= 0.3 * self.last_visible.sum()]
            if over or (ex.label == "blocked" and ex.blocker == "hand" and f["hands"]):
                hand = max(over or f["hands"], key=lambda h: (h.mask & self.last_visible).sum())
                self._take(f, hand.mask, hand)
                f["note"] = f"in or under {_hand_name(hand)} - following it"
                return
            near = _dilate(self.last_visible, HOLDER_REACH)
            self.known_others = list(self.lookalikes0) + self._lookalike_spots(img, near, f["debug"])  # lookalikes now
            side = exit_side(self.last_visible)
            if side:
                self.state, self.detail = "out_of_view", side
                f["new"].append(f"left the picture to the {side}")
            elif ex.label == "same" and not self._on_lookalike(ex.expected):  # still there; the tracker lost it
                f["out"] = self.trk.reinit(box_of(ex.expected))
                self._visible(f, f["out"])
                f["new"].append("the tracker lost it, but it is still here")
                return
            elif ex.label == "gone":
                self.state, self.note = "unknown", "no longer where it was (moved without being seen) - searching"
                f["new"].append("no longer where it was")
            else:
                who = "something" if not f["hands"] else "something (not a hand)"
                self.state, self.note = "covered", f"covered by {who} - last seen here"
                f["new"].append("covered (disappeared without being picked up)")
        f["note"] = self.note if self.state in ("covered", "unknown") else f["note"]
        if not f["new"] and self.state in ("covered", "out_of_view", "unknown"):
            self._redetect(f)

    def _redetect(self, f):
        self.lost_frames += 1
        searched = self.lost_frames % self.finder_every == 0
        found, sure = self._search(f["img"], f["debug"]) if searched else (None, False)
        f["debug"]["re_detect_searched"] = searched
        if found is None:
            return
        cx, cy = box_of(found).reshape(2, 2).mean(0)
        if sure and (not self.known_others or mask_iou(found, self.last_visible) > 0.3):
            f["out"] = self.trk.reinit(box_of(found))
            self._visible(f, f["out"])
            f["new"].append(f"found your object again at ({cx:.0f}, {cy:.0f})")
        else:  # similar objects around and no hand path: don't claim it
            f["maybe"] = found
            if self.maybe_at is None or abs(self.maybe_at[0] - cx) + abs(self.maybe_at[1] - cy) > 40:
                f["new"].append(f"maybe your object at ({cx:.0f}, {cy:.0f}) - similar objects around, not sure")
            self.maybe_at = (cx, cy)

    def _update_held(self, f):
        raw, show, kind, img, hands = f["raw"], f["show"], f["kind"], f["img"], f["hands"]
        self.held_frames += 1
        if show and not (self.hand_detector and self.holder_hand is not None
                         and (raw & self.holders[-1]).sum() >= 0.5 * raw.sum()):
            self._visible(f, raw)
            f["new"].append("your object is visible again (tracker)")
            return
        # follow the holder: the same hand (nearest to where it was), else the tracker's drift
        hand = None
        if self.hand_detector and hands:
            prev = np.argwhere(self.holders[-1]).mean(0)[::-1]
            hand = min(hands, key=lambda h: np.linalg.norm(h.centre - prev))
            if np.linalg.norm(hand.centre - prev) > 3 * max(np.linalg.norm(hand.points[0] - hand.points[9]), 20):
                hand = None  # the nearest hand is too far away to be the same one
        if hand is not None:
            holder = hand.mask
            self.holder_hand = hand
        elif not self.hand_detector and raw.any():
            holder = raw
        else:
            holder = None
        if holder is not None:
            f["holder"] = holder
            self.holders.append(holder)
            self.lost_for = 0
        else:
            self.lost_for += 1
        name = _hand_name(self.holder_hand)
        f["note"] = f"in {name} - following it"
        # partly visible inside the hand: show it
        if hand is not None and raw.any() and kind != "grown" and (raw & _dilate(hand.mask, 10)).any():
            if self.gate.similarity(img, raw)[0] >= IN_HAND_SIM:
                mostly = raw.sum() >= 0.7 * float(np.median(self.gate.areas))
                f["out"], f["note"] = raw, f"in {name}" + ("" if mostly else ", partly visible")
        searched = self.held_frames % self.finder_every == 0
        # while the hand is seen, look where it is or was; once it is lost, look everywhere (but without the hand's
        # path as evidence, a find among lookalikes is only "maybe")
        near = self._reach() if holder is not None else None
        found, sure = self._search(img, f["debug"], near=near) if searched else (None, False)
        f["debug"]["hand_off_searched"] = searched
        if found is not None and near is None and self.known_others and mask_iou(found, self.last_visible) <= 0.3:
            sure = False
        if found is not None and not sure:  # two similar candidates, or no hand path: don't guess
            f["maybe"], found = found, None
            cx, cy = box_of(f["maybe"]).reshape(2, 2).mean(0)
            if self.maybe_at is None or abs(self.maybe_at[0] - cx) + abs(self.maybe_at[1] - cy) > 40:
                f["new"].append(f"maybe put down at ({cx:.0f}, {cy:.0f}) - similar objects around, not sure")
            self.maybe_at = (cx, cy)
        if found is not None and hand is not None and (found & _dilate(hand.mask, 5)).sum() >= 0.3 * found.sum():
            f["out"], f["note"] = found, f"in {name}, partly visible"  # still in the hand, not put down
            found = None
        if found is not None:
            f["out"] = self.trk.reinit(box_of(found))
            self._visible(f, f["out"])
            cx, cy = box_of(found).reshape(2, 2).mean(0)
            f["new"].append(f"put down / let go: your object is at ({cx:.0f}, {cy:.0f})")
        elif holder is None:
            side = exit_side(self.holders[-1])
            if side:
                self.state, self.detail = "out_of_view", side
                f["new"].append(f"carried out of view to the {side}")
            elif self.lost_for > LOST_LIMIT:
                self.state, self.note = "unknown", f"lost track of {name}"
                f["new"].append(f"lost track of {name}")

    def _touching_hand(self, raw, hands):
        """The detected hand that the object's visible part touches (>= 30% of it within 15 px), if any."""
        if not raw.any() or not hands:
            return None
        best = max(hands, key=lambda h: (raw & _dilate(h.mask, 15)).sum())
        return best if (raw & _dilate(best.mask, 15)).sum() >= 0.3 * raw.sum() else None

    def _in_hand(self, f, hand):
        """The object is in this hand: follow the hand; show the visible part if it looks enough like the object."""
        raw = f["raw"]
        if self.state != "held":
            self._take(f, hand.mask, hand)
        sim = self.gate.similarity(f["img"], raw)[0]
        f["debug"]["in_hand_sim"] = round(sim, 3)
        if sim >= IN_HAND_SIM:
            f["out"], f["note"] = raw, f"in {_hand_name(hand)}, partly visible"
        else:
            f["note"] = f"in {_hand_name(hand)} - following it"

    def _lookalike_spots(self, img, exclude, debug=None):
        """Spots of similar-looking things, kept only if a box crop there looks like the object (from live sessions:
        unchecked spots counted 9-10 'lookalikes' in a front-camera scene, so every re-find became 'maybe')."""
        pts = self.finder.spots(img, exclude=exclude)
        if not pts:
            return []
        h, w = img.shape[:2]
        bw, bh = self.obj_box[2] - self.obj_box[0], self.obj_box[3] - self.obj_box[1]
        boxes = [(max(0, int(x - bw / 2)), max(0, int(y - bh / 2)), min(w, int(x + bw / 2) + 1), min(h, int(y + bh / 2) + 1))
                 for x, y in pts]
        sims = (self.app.embed_boxes(img, boxes) @ self.ref_box).cpu().numpy()
        if debug is not None:
            debug["spot_sims"] = [round(float(s), 2) for s in sorted(sims, reverse=True)[:12]]
        return [p for p, s in zip(pts, sims) if s >= LOOKALIKE_SIM]

    def _on_lookalike(self, mask):
        """Does this mask cover a spot where a similar-looking object was seen?"""
        return any(mask[int(y), int(x)] for x, y in self.known_others)

    def _reach(self):
        """Where the hand is or was in its last frames (+40 px)."""
        return _dilate(np.logical_or.reduce(list(self.holders)), HOLDER_REACH)

    def _already_there(self, img, m):
        """Was this candidate already lying here when the hand took the object? (then the hand didn't put it down)"""
        before = getattr(self, "take_img", None)
        if before is None or before.shape != img.shape:
            return False
        ys, xs = np.nonzero(m)
        if len(ys) > 3000:
            pick = np.random.default_rng(0).choice(len(ys), 3000, replace=False)
            ys, xs = ys[pick], xs[pick]
        a = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2LAB)[ys, xs].astype(np.float32)
        b = cv2.cvtColor(cv2.GaussianBlur(before, (5, 5), 0), cv2.COLOR_BGR2LAB)[ys, xs].astype(np.float32)
        return float((np.linalg.norm(a - b, axis=-1) < 18).mean()) >= 0.6

    def _search(self, img, debug=None, near=None):
        """Find the object among object-looking candidates that are not known lookalikes and (if `near`) are where
        the hand is or was. Returns (mask, sure). The object's last spot wins (a hand resting over it rather than
        carrying it); otherwise the best match, unless the runner-up is about as good (then not sure).
        History: v1 took the first candidate that passed; on a table of letter blocks it picked a red U block
        while the T was still in its place under where the hand had been."""
        log, passing = [], []
        for m in self.finder.candidates(img, self.seg):
            ys, xs = np.nonzero(m)
            entry = {"at": [int(xs.mean()), int(ys.mean())], "area": int(m.sum())}
            log.append(entry)
            if any(m[int(y), int(x)] for x, y in self.known_others):
                entry["rejected"] = "known lookalike"
            elif near is not None and not np.logical_and(m, near).any():
                entry["rejected"] = "not near the hand"
            elif near is not None and self._already_there(img, m):
                # a hand-off can only release something new there (a live replay of letter blocks put the hand next
                # to another block that had been lying there all along, and it was taken for the released object)
                entry["rejected"] = "already lying there before the hand took it"
            else:
                entry["sim"] = round(self.gate.similarity(img, m)[0], 3)
                if self.gate.confirm(img, m):
                    entry["details"] = self.explainer.detail_matches(img, m)
                    passing.append((entry["sim"], m, entry))
                else:
                    entry["rejected"] = "does not look like the object"
        if debug is not None:
            debug["search_candidates"] = log
        if not passing:
            return None, False
        at_spot = [p for p in passing if mask_iou(p[1], self.last_visible) > 0.3]
        if at_spot:
            best = max(at_spot, key=lambda p: p[0])
            best[2]["accepted"] = "at its last spot"
            return best[1], True
        # identity by small details first: the candidate whose details match the remembered view, clearly
        by_details = sorted(passing, key=lambda p: -p[2]["details"])
        top = by_details[0]
        others = [p for p in by_details[1:] if mask_iou(p[1], top[1]) <= 0.5]
        if top[2]["details"] >= DETAIL_MIN and (not others or others[0][2]["details"] <= 0.6 * top[2]["details"]):
            top[2]["accepted"] = f"details match ({top[2]['details']})"
            return top[1], True
        # otherwise by overall look; with many similar things around, look alone is not enough to be sure
        passing.sort(key=lambda p: -p[0])
        rivals = [p for p in passing[1:] if mask_iou(p[1], passing[0][1]) <= 0.5]  # not the same object again
        sure = (not rivals or passing[0][0] - rivals[0][0] >= 0.03) and len(self.known_others) < CLUTTER
        passing[0][2]["accepted"] = "best match" if sure else "best match, but not sure (similar things around)"
        return passing[0][1], sure


def _dilate(mask, r):
    return cv2.dilate(mask.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
