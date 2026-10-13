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
  in_pocket    (with compass.body) the object vanished while the holding hand was at the person's clothes, then
               the hand moved away from there or left the picture (or stayed there for ~2.5 s): "in your pocket".
               Which pocket is guessed from which hand put it there.
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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import cv2
import numpy as np
import timm
import torch

from compass.appearance import SIM_PARTIAL, Gate
from compass.explain import Explainer
from compass.models import image_predictor

HOLDER_MEMORY, HOLDER_REACH, LOST_LIMIT, BORDER = 6, 40, 15, 12
LOOKALIKE_SIM = 0.75  # a spot counts as a lookalike only if its box crop looks this much like the object's
PRECHECK_SIM, PRECHECK_KEEP = 0.6, 3  # search: outline only the 3 best spots whose box crop looks >= 0.6 like the object
DETAIL_MIN, CLUTTER = 12, 5  # details needed to decide by detail matching; lookalike spots that make a scene "cluttered"
IN_HAND_SIM = 0.45  # in a detected hand, show the visible part if it looks at least this much like the object
IN_HAND_VISIBLE = 0.3  # less than this share visible + a hand touching it = "in your hand", not "visible"
HAND_TOUCH = 0.15  # share of the visible part within 15 px of a detected hand that counts as "touching" (real clips:
# a hand holding the wallet from below/the side touched 13-30% of it)
MOSTLY_HIDDEN = 0.25  # less than this share visible, with something new in front = "covered", not "visible"
POCKET_CLOTHES = 0.35  # share of clothes around the holding hand for "the hand is at the clothes"
LOST = ("unknown", "out_of_view")  # states where we don't know where the object is
LOST_CHECK_MS = 1000  # while lost, the whole picture is checked (tracker + search) once per second of video
STILL_DIFF, STILL_SHARE, STILL_MAX_MS = 25, 0.01, 2000  # skip the tracker while a visible object rests: fewer than 1%
# of the (blurred, quarter-size) pixels around it differ by more than 25 grey levels from the last tracked frame and no
# hand is near; the tracker still runs at least every 2 s of video
POCKET_FRAMES, POCKET_STAY = 2, 8  # hidden-at-clothes frames needed; ...or this many if the hand just stays there


class Finder:
    """Object-looking things via DINOv2 (ViT-S/14) patch similarity to the registered object's patches."""

    GRID = 32  # 448 px / 14 px patches
    MAX_PEAKS = 10  # when searching, outline at most the 10 strongest peaks (each costs a SAM decode)
    MAX_PEAKS_RECORD = 40  # when recording which lookalikes are where, be more complete
    one_at_a_time = True  # outline peaks one by one, skipping those inside earlier outlines (round 4; same masks)

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

    def candidates(self, img, segmenter, max_peaks=None, pick=None, window=None, worth=None):
        """Masks of separate object-looking things: one SAM mask per local peak of the similarity map.
        `pick`: optional function choosing which peak points to outline (each outline costs a SAM decode, ~17 ms).
        `worth(x, y)`: optional cheap test of a peak before outlining it (round 4).
        Peaks are outlined one at a time, strongest first, and a peak inside an outline already made is not outlined
        again (same answer as outlining all and dropping duplicates afterwards; on the E0 clips 1281 of 3046 in-hand
        outlines were such duplicates)."""
        tok = self._cached[1] if getattr(self, "_cached", (None,))[0] is img else self.tokens(img)
        sim = (tok @ self.proto).cpu().numpy()
        peaks = (sim == cv2.dilate(sim, np.ones((3, 3), np.uint8))) & (sim > self.thr)
        h, w = img.shape[:2]
        order = sorted(zip(*np.nonzero(peaks)), key=lambda p: -sim[p])[:max_peaks or self.MAX_PEAKS]
        points = [((gx + 0.5) * w / self.GRID, (gy + 0.5) * h / self.GRID) for gy, gx in order]
        if pick is not None:
            points = pick(points)
        masks = []
        if not self.one_at_a_time:  # round-3 behaviour (for comparisons): outline all, then drop duplicates
            for (x, y), m in zip(points, segmenter.segment_points(img, points, window=window)):
                if any(k[int(y), int(x)] or mask_iou(k, m) > 0.5 for k in masks):
                    continue
                if 0.25 * self.ref_area <= m.sum() <= 3 * self.ref_area:
                    masks.append(m)
            return masks
        for x, y in points:
            if any(k[int(y), int(x)] for k in masks):
                continue  # this peak lies on an object already outlined
            if worth is not None and not worth(x, y):
                continue
            m = segmenter.segment_points(img, [(x, y)], window=window)[0]
            if any(mask_iou(k, m) > 0.5 for k in masks):
                continue  # it outlines the same object again
            if 0.25 * self.ref_area <= m.sum() <= 3 * self.ref_area:
                masks.append(m)
        return masks


class Segmenter:
    """SAM (EdgeTAM's image model) outlining the objects under points, in one batch.

    When the tracker has already encoded this frame (`share`), its image features are reused: the image model has
    the same weights, so encoding the frame again (~90 ms on an M2) would give the same features. The outlines are
    then made at full resolution. Otherwise the frame is encoded at half resolution (the outlines are only used to
    find candidates; full-size outlines made 40 candidates take ~0.75 s)."""

    WIDTH = 640

    def __init__(self, device, image_size=1024):
        self.pred = image_predictor(device, image_size)  # same size as the tracker: it reuses the tracker's features
        self.last_image = None
        self.k = 1.0
        self._shared = (None, None)

    def share(self, img, backbone_out):
        """Image features the tracker computed for this frame (EdgeTAM backbone output), to reuse."""
        self._shared = (img, backbone_out)

    def _use_features(self, backbone_out, hw):
        """Like SAM2ImagePredictor.set_image, from already computed backbone features."""
        p, m = self.pred, self.pred.model
        p.reset_predictor()
        _, feats, _, _ = m._prepare_backbone_features(backbone_out)
        if m.directly_add_no_mem_embed:
            feats[-1] = feats[-1] + m.no_mem_embed
        feats = [f.permute(1, 2, 0).view(1, -1, *s) for f, s in zip(feats[::-1], p._bb_feat_sizes[::-1])][::-1]
        p._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        p._is_image_set, p._orig_hw = True, [hw]

    @torch.inference_mode()
    def segment_points(self, img, points, window=None):
        """One outline per point. `window` (grid cells, on shared full-resolution features): outline each point in a
        window x window square of EdgeTAM's 64x64 feature grid around it instead of the whole picture (round 3b)."""
        if not points:
            return []
        h, w = img.shape[:2]
        if window and self._shared[0] is img and self._shared[1] is not None:
            if self.last_image is not img:
                self._use_features(self._shared[1], (h, w))
                self.k, self.last_image = 1.0, img
            return [self._segment_in_window(pt, window, (h, w)) for pt in points]
        if self.last_image is not img:
            if self._shared[0] is img and self._shared[1] is not None:
                self._use_features(self._shared[1], (h, w))
                self.k = 1.0
            else:
                self.k = min(1.0, self.WIDTH / w)
                small = cv2.resize(img, (round(w * self.k), round(h * self.k)), interpolation=cv2.INTER_AREA) \
                    if self.k < 1 else img
                self.pred.set_image(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            self.last_image = img
        k = self.k
        pts = np.array(points, np.float32)[:, None, :] * k
        masks, _, _ = self.pred.predict(point_coords=pts, point_labels=np.ones((len(points), 1)), multimask_output=False)
        masks = masks.reshape(len(points), *masks.shape[-2:]) > 0
        return [cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool) if k < 1 else m
                for m in masks]

    def _segment_in_window(self, pt, s, hw):
        """Outline the object under `pt` from the s x s grid cells around it: the mask decoder sees the window's
        features with their true positions (sliced, as in compass.window); outside the window the outline is empty."""
        p, (h, w), g = self.pred, hw, self.pred.model.image_size // 16
        gx = min(max(round(pt[0] * g / w - s / 2), 0), g - s)
        gy = min(max(round(pt[1] * g / h - s / 2), 0), g - s)
        win = (slice(None), slice(None), slice(gy, gy + s), slice(gx, gx + s))
        hr0, hr1 = p._features["high_res_feats"]
        coords = torch.tensor([[[float(pt[0]) * p.model.image_size / w, float(pt[1]) * p.model.image_size / h]]],
                              dtype=torch.float32, device=hr0.device)
        labels = torch.ones(1, 1, dtype=torch.int, device=hr0.device)
        sparse, dense = p.model.sam_prompt_encoder(points=(coords, labels), boxes=None, masks=None)
        low, _, _, _ = p.model.sam_mask_decoder(
            image_embeddings=p._features["image_embed"][win].contiguous(),
            image_pe=p.model.sam_prompt_encoder.get_dense_pe()[win].contiguous(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense[win].contiguous(),
            multimask_output=False, repeat_image=False,
            high_res_features=[hr0[:, :, 4 * gy:4 * (gy + s), 4 * gx:4 * (gx + s)].contiguous(),
                               hr1[:, :, 2 * gy:2 * (gy + s), 2 * gx:2 * (gx + s)].contiguous()])
        x0, x1, y0, y1 = round(gx * w / g), round((gx + s) * w / g), round(gy * h / g), round((gy + s) * h / g)
        logits = cv2.resize(low[0, 0].float().cpu().numpy(), (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        mask = np.zeros((h, w), bool)
        mask[y0:y1, x0:x1] = logits > 0
        return mask

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
    state: str  # visible | unsure | covered | held | out_of_view | unknown | in_pocket
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
                   "in_pocket": "in your pocket",
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

    def __init__(self, tracker, appearance, finder, segmenter, finder_every=1, hands=None, clothes=None,
                 skip_when_still=True, precheck=False, window_outlines=True, spot_tests=True):
        self.trk, self.app, self.finder, self.seg = tracker, appearance, finder, segmenter
        # search: quick look check of each spot before outlining it (round 2, step 2b). Off: on a quiet Mac it cost
        # about as much as it saved (look checks +75 ms, outlines -80 ms) and 4 frames of E0 accuracy
        self.precheck = precheck
        self.window_outlines = window_outlines  # search: outline each spot in a window around it (round 3b)
        self.spot_tests = spot_tests  # in-hand search: cheap tests before outlining a spot (round 4)
        self.skip_when_still = skip_when_still  # reuse the last result while the object rests and nothing changes
        self.finder_every = finder_every
        self.hand_detector = hands
        # hands run on the CPU (MediaPipe) while the tracker runs on the GPU: in parallel, on one worker thread
        self._hand_pool = ThreadPoolExecutor(max_workers=1) if hands else None
        self.clothes = clothes  # compass.body.ClothesSegmenter, or None (no pocket detection)
        self.events = []

    def _hands_async(self, img, t_ms):
        return self._hand_pool.submit(self.hand_detector.detect, img, t_ms) if self.hand_detector else None

    def start(self, img, box, t_ms=None):
        hands_job = self._hands_async(img, t_ms)
        raw, score = self.trk.start(img, box)
        self.seg.share(img, self.trk.features())
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
        self.pocket_evidence, self.pocket_spot, self.pocket_hand = 0, None, None
        self.t = 0
        self.events = []
        self._anchor, self.skipped, self._lost_check_t, self._last_maybe = None, 0, None, None
        hands = hands_job.result() if hands_job else []
        self._set_anchor(img, raw, raw, raw, score, None, t_ms)
        return Step(0, raw, raw, score, "visible", hands=hands)

    # ------------------------------------------------------------------ skip work / look only where it can be

    @staticmethod
    def _small_grey(patch):
        h, w = patch.shape[:2]
        small = cv2.resize(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), (max(w // 4, 1), max(h // 4, 1)),
                           interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(small, (5, 5), 0)

    def _set_anchor(self, img, region, out, raw, score, note, t_ms):
        """Remember how the area around `region` looked on this tracked frame, to skip later frames while it stays
        the same (or forget it)."""
        if t_ms is None or region is None or not region.any():
            self._anchor = None
            return
        h, w = img.shape[:2]
        x0, y0, x1, y1 = box_of(region)
        box = (max(int(x0) - HOLDER_REACH, 0), max(int(y0) - HOLDER_REACH, 0),
               min(int(x1) + HOLDER_REACH, w), min(int(y1) + HOLDER_REACH, h))
        self._anchor = {"state": self.state, "box": box, "patch": self._small_grey(img[box[1]:box[3], box[0]:box[2]]),
                        "t": t_ms, "out": out, "raw": raw, "score": score, "note": note}

    def _nothing_changed(self, img, t_ms):
        a = self._anchor
        if a is None or a["state"] != self.state or t_ms is None or not 0 <= t_ms - a["t"] <= STILL_MAX_MS \
                or img.shape[:2] != a["out"].shape:
            return False
        x0, y0, x1, y1 = a["box"]
        changed = cv2.absdiff(self._small_grey(img[y0:y1, x0:x1]), a["patch"]) > STILL_DIFF
        return float(changed.mean()) < STILL_SHARE

    def _hand_near_anchor(self, hands):
        x0, y0, x1, y1 = self._anchor["box"]
        return any(h.mask[y0:y1, x0:x1].any() for h in hands)

    def _skip_reason(self, img, t_ms, hands_job):
        """Why this frame needs no tracking at all (or None), and the hands if they had to be awaited to decide.
          visible / covered  nothing changed around the object (or its last spot) and no hand is near
          in pocket          no hand in view (only a hand can take it out)
          lost               the whole picture is checked only every LOST_CHECK_MS"""
        if not self.skip_when_still or t_ms is None:
            return None, None
        if self.state in ("visible", "covered") and self._nothing_changed(img, t_ms):
            hands = hands_job.result() if hands_job else []
            if self._hand_near_anchor(hands):
                return None, hands
            return ("resting, nothing changed" if self.state == "visible" else "still covered, nothing changed"), hands
        if self.state == "in_pocket" and self.hand_detector:
            hands = hands_job.result()
            return (None if hands else "in the pocket, no hand in view"), hands
        if self.state in LOST and self._lost_check_t is not None and 0 <= t_ms - self._lost_check_t < LOST_CHECK_MS:
            return "lost, next whole-picture check later", None
        return None, None

    def _focus(self, hands):
        """Where the tracker looks: a box (it tracks in a window around it), "full" (whole picture), or None (around
        its own last mask)."""
        if self.state == "covered" and self.last_visible.any():
            return box_of(self.last_visible)  # under the cover, at its last spot
        if self.state == "held" and self.holders and self.holders[-1].any():
            return box_of(self.holders[-1])  # with the hand that holds it
        if self.state == "in_pocket":
            return box_of(np.logical_or.reduce([h.mask for h in hands])) if hands else "full"
        if self.state in LOST:
            return "full"
        return None

    def _skipped_step(self, reason, hands):
        self.skipped += 1
        a = self._anchor if self.state in ("visible", "covered") else None
        empty = np.zeros_like(self.last_visible)
        out, raw = (a["out"], a["raw"]) if a else (empty, empty)
        note = a["note"] if a else self.note
        maybe = self._last_maybe if self.state in LOST and self.maybe_at is not None else None
        return Step(self.t, out, raw, a["score"] if a else 0.0, self.state, self.detail, None, [],
                    {"skipped": reason, "hands": len(hands)}, maybe, note, hands)

    # ------------------------------------------------------------------ per frame

    def update(self, img, t_ms=None):
        self.t += 1
        hands_job = self._hands_async(img, t_ms)
        reason, hands = self._skip_reason(img, t_ms, hands_job)
        if reason:
            return self._skipped_step(reason, hands if hands is not None else (hands_job.result() if hands_job else []))
        was_lost = self.state in LOST
        raw, score = self.trk.track(img, focus=self._focus(hands or []))
        self.seg.share(img, self.trk.features())
        if hands is None:
            hands = hands_job.result() if hands_job else []
        hand_union = np.logical_or.reduce([h.mask for h in hands]) if hands else np.zeros_like(raw)
        if raw.any():
            show, _, info = self.gate.check(img, raw)
            kind = info["kind"]
        else:
            show, kind, info = False, "empty", {}
        f = {"raw": raw, "show": show, "kind": kind, "hands": hands, "hand_union": hand_union, "img": img, "t_ms": t_ms,
             "out": np.zeros_like(raw), "holder": None, "maybe": None, "new": [], "note": None,
             "debug": {"check": {k: info[k] for k in ("kind", "sim", "area") if k in info},
                       "typical_area": float(np.median(self.gate.areas)), "hands": len(hands)}}
        if self.state == "held":
            self._update_held(f)
        else:
            self._update_free(f)
        events = [(self.t, e) for e in f["new"]]
        self.events += events
        if self.state not in ("covered", "out_of_view", "unknown", "held", "in_pocket"):
            self.maybe_at = None
        if f["maybe"] is not None:
            self._last_maybe = f["maybe"]
        if self.state in LOST:  # a whole-picture check ran (the frame it got lost doesn't count: check right away)
            self._lost_check_t = t_ms if was_lost else None
        else:
            self._lost_check_t = None
        if events or f["maybe"] is not None:
            self._anchor = None  # something happened: track the next frame
        elif self.state == "visible" and f["out"].any():
            self._set_anchor(img, f["out"], f["out"], raw, score, f["note"], t_ms)
        elif self.state == "covered":
            self._set_anchor(img, self.last_visible, np.zeros_like(raw), np.zeros_like(raw), score, f["note"], t_ms)
        else:
            self._anchor = None
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
        # where the hand and the object were at that moment: put down near there, it can look as it did then
        self.take_region = _dilate(holder | self.last_visible, 25)
        self.pocket_evidence, self.pocket_spot, self.pocket_hand = 0, None, None
        self.holders.clear()
        self.holders.append(holder)
        self.lost_for, self.held_frames = 0, 0
        self.known_others = list(self.lookalikes0) + self._lookalike_spots(
            f["img"], _dilate(holder | self.last_visible, HOLDER_REACH), f["debug"])
        f["note"] = f"in {_hand_name(hand)} - following it"
        f["new"].append(f"taken by {_hand_name(hand)} ({len(self.known_others)} similar-looking spot(s) elsewhere, ignored)")

    def _update_free(self, f):
        raw, show, kind, img = f["raw"], f["show"], f["kind"], f["img"]
        if self.state == "in_pocket":
            # in the pocket until it is clearly seen again; the tracker can keep a faint mask at the pocket next to
            # the hand, which must not count as "taken by your hand" again. Seen again = passes the look check, or a
            # full-size outline away from the pocket that looks like the object at the partial-view level (real clip
            # 2026-10-12: put back on the table, it scored 0.79-0.81, below the 0.85 of the normal check, for 16 s)
            typical = float(np.median(self.gate.areas))
            sim = (f["debug"].get("check") or {}).get("sim", 0.0)
            away = self.pocket_spot is None or (raw.any() and np.linalg.norm(
                box_of(raw).reshape(2, 2).mean(0) - self.pocket_spot) > np.sqrt(typical))
            seen = show or (kind != "grown" and raw.sum() >= 0.7 * typical and away and sim >= SIM_PARTIAL)
            if seen and raw.sum() >= MOSTLY_HIDDEN * typical:
                self._visible(f, raw)
                f["new"].append("taken out of your pocket")
            else:
                f["note"] = self.note
            return
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
            elif (raw.sum() < MOSTLY_HIDDEN * typical and ex.missing.get("object", 0) < 0.5 and exit_side(raw) is None
                  and (ex.label in ("blocked", "something_else") or ex.missing.get("around_changed", 0) >= 0.5)):
                # (not when the object still shows where it should be - a tracker glitch - nor when the sliver is at
                # the picture's edge - it is leaving the view)
                # only a sliver shows and something new is in front: it is covered (real clip: a sheet of paper over
                # the wallet left 1-21% visible, which was shown as "visible, partly hidden")
                if self.state != "covered":
                    f["new"].append("covered (disappeared without being picked up)")
                self.state = "covered"
                self.note = f"mostly covered by something ({raw.sum() / max(typical, 1):.0%} visible) - last seen here"
                f["note"] = self.note
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
            elif (ex.label == "unsure" and ex.missing.get("around_changed", 0) >= 0.5 and ex.visible_fraction < 0.9
                  and (f["debug"].get("check") or {}).get("sim", 0) >= 0.6):
                # the surroundings clearly changed (something came in front) and what shows still looks like the
                # object; strict colour matching failed (real clip: a paper's shadow over the visible part)
                self._visible(f, raw, f"partly hidden by something ({ex.visible_fraction:.0%} visible)", remember=False)
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
        elif self.state == "covered":
            # it should be under the cover at its last spot, so only that spot is watched (no whole-picture search):
            # the cover lifted and it is still there -> visible; the spot shows plain background -> it was moved
            # without being seen -> lost (from then on the whole picture is checked)
            if ex.label == "same" and not self._on_lookalike(ex.expected):
                f["out"] = self.trk.reinit(box_of(ex.expected))
                self._visible(f, f["out"])
                f["new"].append("uncovered: your object is still here")
                return
            if ex.label == "gone":
                self.state, self.note = "unknown", "not under the cover any more (moved without being seen) - searching"
                f["new"].append("no longer where it was covered - searching")
        f["note"] = self.note if self.state in ("covered", "unknown") else f["note"]
        if self.state == "in_pocket":
            f["note"] = self.note
        if not f["new"] and self.state in LOST:
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
        found, sure = self._search(img, f["debug"], near=near, hands=hands) if searched else (None, False)
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
        elif self._pocket(f, hand):
            pass
        elif holder is None:
            side = exit_side(self.holders[-1])
            if side:
                self.state, self.detail = "out_of_view", side
                f["new"].append(f"carried out of view to the {side}")
            elif self.lost_for > LOST_LIMIT:
                self.state, self.note = "unknown", f"lost track of {name}"
                f["new"].append(f"lost track of {name}")

    def _pocket(self, f, hand):
        """Did the object go into a pocket? Evidence: it is hidden while the holding hand is at the person's clothes
        (two frames); confirmed when the hand then moves away from there or leaves the picture, or stays ~2.5 s.
        (Real clip: while carrying the wallet the hand was always next to the shorts, 40-65% clothes around it, so
        "hand at clothes" alone means nothing; the wallet vanishing there does.)"""
        if self.clothes is None:
            return False
        hidden = not f["out"].any()
        last = self.holder_hand
        if hidden and hand is not None:
            near = _dilate(hand.mask, 25) & ~hand.mask
            share = float(self.clothes.clothes(f["img"], f["t_ms"])[near].mean()) if near.any() else 0.0
            f["debug"]["clothes_at_hand"] = round(share, 2)
            if share >= POCKET_CLOTHES:
                self.pocket_evidence += 1
                self.pocket_spot, self.pocket_hand = hand.centre.copy(), hand
        elif (hidden and last is not None and 1 <= self.lost_for <= POCKET_STAY and exit_side(last.mask) is None
              and f["raw"].sum() < MOSTLY_HIDDEN * float(np.median(self.gate.areas))):
            # the holding hand vanished together with the object (even the tracker sees < 25% of it), at the clothes
            # and not at the picture's edge: it went into the pocket (real clip 2026-10-12: the hand stayed inside the
            # pocket for ~10 s, undetected; tracker 2-11% of the object). Not when the tracker still sees the object:
            # a hand holding it close to the camera, in front of a shirt, went undetected for 10 s too (tracker ~100%)
            near = _dilate(last.mask, 25) & ~last.mask
            share = float(self.clothes.clothes(f["img"], f["t_ms"])[near].mean()) if near.any() else 0.0
            f["debug"]["clothes_at_last_hand"] = round(share, 2)
            if share >= POCKET_CLOTHES:
                self.pocket_evidence += 1
                self.pocket_spot, self.pocket_hand = last.centre.copy(), last
        elif not hidden:
            self.pocket_evidence = 0
        if self.pocket_evidence < POCKET_FRAMES:
            return False
        size = max(np.linalg.norm(self.pocket_hand.points[0] - self.pocket_hand.points[9]), 20)
        moved_away = hand is not None and np.linalg.norm(hand.centre - self.pocket_spot) > 1.2 * size
        gone = hand is None and self.lost_for >= 2
        if not (moved_away or gone or self.pocket_evidence >= POCKET_STAY):
            f["note"] = f"in {_hand_name(hand or self.pocket_hand)}, at your pocket"
            return True
        side = self.pocket_hand.side
        pocket = f"your {side} pocket" if side in ("left", "right") else "your pocket"
        self.state, self.detail = "in_pocket", side
        self.note = f"in {pocket} (put there by {_hand_name(self.pocket_hand)})"
        f["note"] = self.note
        f["new"].append(f"put into {pocket}")
        return True

    def _touching_hand(self, raw, hands):
        """The detected hand that the object's visible part touches (>= 30% of it within 15 px), if any."""
        if not raw.any() or not hands:
            return None
        best = max(hands, key=lambda h: (raw & _dilate(h.mask, 15)).sum())
        return best if (raw & _dilate(best.mask, 15)).sum() >= HAND_TOUCH * raw.sum() else None

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

    def _spot_tests(self, img, near, hands):
        """Cheap tests of an in-hand search spot before outlining it (round 4). On the E0 clips the in-hand search
        outlined 3046 spots, 11 of them useful (accepted or passing the checks); each test below dropped hundreds and
        none of the 11. A spot is not outlined if it lies on a detected hand, more than one object size from where the
        hand is or was, within 25 px of a known lookalike, or if its surroundings look as they did when the hand took
        the object (the "already lying there" check, done before outlining instead of after) - except where the hand
        took it."""
        h, w = img.shape[:2]
        on_hand = np.logical_or.reduce([hd.mask for hd in hands]) if hands else None
        dist = cv2.distanceTransform((~near).astype(np.uint8), cv2.DIST_L2, 3)
        size = float(np.sqrt(np.median(self.gate.areas)))
        before = getattr(self, "take_img", None)
        take_region = getattr(self, "take_region", None)
        if take_region is not None and take_region.shape != (h, w):
            take_region = None

        def worth(x, y):
            xi, yi = min(int(x), w - 1), min(int(y), h - 1)
            if on_hand is not None and on_hand[yi, xi]:
                return False
            if dist[yi, xi] > size:
                return False
            if any(abs(x - ox) + abs(y - oy) < 25 for ox, oy in self.known_others):
                return False
            # (not where the hand took it: put back there, or uncovered without ever leaving, the area can look as it
            # did when taken; drawn test scenes: releases there were noticed one picture late)
            if take_region is not None and take_region[yi, xi]:
                return True
            return not _looks_unchanged(img, before, xi, yi)

        return worth

    def _outline_window(self, img):
        """Grid cells (of 64) around each search spot to outline it in: twice the object's registered size, at least
        16; None (whole picture) if that would be most of the picture or windowed outlines are off."""
        if not self.window_outlines:
            return None
        h, w = img.shape[:2]
        bw, bh = self.obj_box[2] - self.obj_box[0], self.obj_box[3] - self.obj_box[1]
        g = self.seg.pred.model.image_size // 16  # feature grid: 64 at 1024 px
        s = 4 * int(np.ceil(max(16, 2 * max(bw * g / w, bh * g / h)) / 4))
        return s if s <= 0.75 * g else None

    def _precheck(self, img, points, debug=None):
        """Which search spots are worth outlining: a quick look check of an object-sized box around each (one batch,
        ~20 ms), keeping the PRECHECK_KEEP best that reach PRECHECK_SIM, plus the spot nearest the object's last spot
        (that spot wins anyway if the object is still there). Outlining all 10 cost ~17 ms each."""
        if len(points) <= PRECHECK_KEEP:
            return points
        h, w = img.shape[:2]
        bw, bh = self.obj_box[2] - self.obj_box[0], self.obj_box[3] - self.obj_box[1]
        boxes = [(max(0, int(x - bw / 2)), max(0, int(y - bh / 2)), min(w, int(x + bw / 2) + 1), min(h, int(y + bh / 2) + 1))
                 for x, y in points]
        sims = (self.app.embed_boxes(img, boxes) @ self.ref_box).cpu().numpy()
        order = sorted(range(len(points)), key=lambda i: -sims[i])
        keep = [i for i in order if sims[i] >= PRECHECK_SIM][:PRECHECK_KEEP]
        if self.last_visible.any():
            cx, cy = box_of(self.last_visible).reshape(2, 2).mean(0)
            nearest = min(range(len(points)), key=lambda i: (points[i][0] - cx) ** 2 + (points[i][1] - cy) ** 2)
            if nearest not in keep and self.last_visible[min(int(points[nearest][1]), h - 1), min(int(points[nearest][0]), w - 1)]:
                keep.append(nearest)
        if debug is not None:
            debug["precheck_sims"] = [round(float(sims[i]), 2) for i in order]
            debug["precheck_kept"] = len(keep)
        return [points[i] for i in sorted(keep, key=lambda i: -sims[i])]

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

    def _search(self, img, debug=None, near=None, hands=None):
        """Find the object among object-looking candidates that are not known lookalikes and (if `near`) are where
        the hand is or was. Returns (mask, sure). The object's last spot wins (a hand resting over it rather than
        carrying it); otherwise the best match, unless the runner-up is about as good (then not sure).
        History: v1 took the first candidate that passed; on a table of letter blocks it picked a red U block
        while the T was still in its place under where the hand had been."""
        log, passing, left = [], [], []
        pick = (lambda pts: self._precheck(img, pts, debug)) if self.precheck else None
        worth = self._spot_tests(img, near, hands) if near is not None and self.spot_tests else None
        for m in self.finder.candidates(img, self.seg, pick=pick, window=self._outline_window(img), worth=worth):
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
                left.append((m, entry))
        for (m, entry), sim in zip(left, self.gate.similarities(img, [m for m, _ in left])):  # one batch
            entry["sim"] = round(sim, 3)
            if self.gate.confirm(img, m, sim=sim):
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


def _looks_unchanged(img, before, x, y, r=24):
    """Does the area around (x, y) look as it did in `before`? (smoothed Lab, as ObjectState._already_there)"""
    if before is None or before.shape != img.shape:
        return False
    h, w = img.shape[:2]
    y0, y1, x0, x1 = max(y - r, 0), min(y + r + 1, h), max(x - r, 0), min(x + r + 1, w)
    a = cv2.cvtColor(cv2.GaussianBlur(img[y0:y1, x0:x1], (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(cv2.GaussianBlur(before[y0:y1, x0:x1], (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    return float((np.linalg.norm(a - b, axis=-1) < 18).mean()) >= 0.6


def _dilate(mask, r):
    if r > 100:  # big squares: a chessboard distance map gives exactly the same result in one pass (~4 ms at any size;
        # cv2.dilate is faster up to ~100 px, 16 ms at 300 px)
        return cv2.distanceTransform((~mask.astype(bool)).astype(np.uint8), cv2.DIST_C, 3) <= r
    return cv2.dilate(mask.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
