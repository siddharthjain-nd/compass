"""Hands: where they are in the picture, as outlines, using MediaPipe's hand landmarker (Google, Apache-2.0).

MediaPipe finds up to 2 hands and 21 points on each (wrist, knuckles, finger joints, tips). A hand's outline is
built from those points: the filled convex hull (palm + finger area) plus the finger bones drawn with a width
proportional to the hand's size, so fingers wrapped around an object are covered.

Used by compass.holder to tell "the object went into a hand" from "something covered it", to follow the hand that
holds the object, and to name the blocker ("hidden by your hand").

Left/right: MediaPipe's documentation says labels assume a mirrored (selfie) image. Checked on real video with
MediaPipe 1.1.0 and un-mirrored frames, its labels already match the person's own hands: in EdgeTAM's coffee video
the hand holding the mug wears a watch (left) and the pouring hand is labelled right. So no swap is applied to
camera frames; swap only if frames are mirrored before detection.
"""
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from compass.models import ROOT

MODEL = ROOT / "third_party" / "models" / "hand_landmarker.task"
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
         (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


@dataclass
class Hand:
    mask: np.ndarray  # bool HxW outline of the hand
    points: np.ndarray  # 21 x 2 pixel coordinates
    side: str  # "left" / "right": the person's own hand (un-mirrored camera frames)
    score: float

    @property
    def centre(self):
        return self.points.mean(0)


def hand_mask(points, shape):
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    hull = cv2.convexHull(points.astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 1)
    size = np.linalg.norm(points[0] - points[9])  # wrist to middle-finger knuckle
    width = max(int(size * 0.35), 3)
    for a, b in BONES:
        cv2.line(mask, tuple(points[a].astype(int)), tuple(points[b].astype(int)), 1, width)
    return mask.astype(bool)


class HandDetector:
    def __init__(self, max_hands=2, min_confidence=0.5, swap_sides=False):
        if not MODEL.exists():
            raise FileNotFoundError(f"{MODEL} is missing: run  bash scripts/setup_third_party.sh")
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL)), running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands, min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence, min_tracking_confidence=min_confidence)
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.swap_sides = swap_sides
        self._t_ms = 0

    def detect(self, img_bgr, t_ms=None):
        """Hands in this frame. `t_ms` must increase from call to call (MediaPipe tracks hands over time);
        by default it advances 33 ms per call."""
        self._t_ms = max(self._t_ms + 1, int(t_ms) if t_ms is not None else self._t_ms + 33)
        h, w = img_bgr.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        result = self.landmarker.detect_for_video(image, self._t_ms)
        hands = []
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], np.float32)
            label = handedness[0].category_name.lower()
            if self.swap_sides:
                label = {"left": "right", "right": "left"}.get(label, label)
            hands.append(Hand(hand_mask(pts, (h, w)), pts, label, float(handedness[0].score)))
        return hands

    def close(self):
        self.landmarker.close()
