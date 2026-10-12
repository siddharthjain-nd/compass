"""StreamingTracker: EdgeTAM fed one frame at a time, as a live camera needs.

EdgeTAM's video predictor is built for finished videos: it loads every frame up front and keeps every
frame's image (~12 MB each) and outputs, which would fill 16 GB of RAM in about 40 s of 30 fps video.
This wrapper uses the same predictor code paths, but:
  - frames are added as they arrive (the frame count is set huge, the frame store is filled live);
  - only the current frame's image is kept;
  - only outputs EdgeTAM can still read are kept: the last 16 frames (6 are used as memory, 16 as object
    pointers) and the first + latest conditioning frames.
So memory stays flat however long it runs, and results match running the offline predictor frame by frame.
"""
import cv2
import numpy as np
import torch

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class StreamingTracker:
    KEEP_RECENT = 16  # EdgeTAM reads up to 16 past frames (object pointers); memory itself uses 6
    KEEP_COND = 4  # conditioning frames kept: the first one + the latest 3 re-initialisations

    def __init__(self, predictor, window=False):
        """window: track only around the object (compass.window, E5a): the memory step and mask decoder run on a
        window around the last mask instead of the whole frame."""
        self.p = predictor
        self.state = None
        self.t = -1
        self._mean = self._std = None
        self.window = None
        if window:
            from compass.window import Window
            self.window = Window(predictor)
        self._last = None

    # ------------------------------------------------------------ public API

    def start(self, img_bgr, box):
        """Begin tracking the object inside `box` (x0, y0, x1, y1) on this frame. Returns (mask, score)."""
        h, w = img_bgr.shape[:2]
        self.state = self._new_state(h, w)
        self.t = 0
        self._push(img_bgr)
        if self.window:
            self.window.cell = None
        with torch.inference_mode():
            _, _, m = self.p.add_new_points_or_box(self.state, frame_idx=0, obj_id=1, box=np.asarray(box, np.float32))
        self._last = self._mask(m)
        return self._last, self.score()

    def track(self, img_bgr, focus=None):
        """Track the object into the next frame. Returns (mask, score).
        focus (with a window): None = around the last mask; (x0, y0, x1, y1) = around this box (where the object
        should be: its last spot when covered, the hand holding it); "full" = the whole frame."""
        self.t += 1
        self._push(img_bgr)
        if self.window:
            if isinstance(focus, str):
                self.window.cell = None
            elif focus is not None:
                self.window.set_from_box(focus, img_bgr.shape[:2])
            else:
                self.window.set_from_mask(self._last)
        with torch.inference_mode():
            _, _, m = next(iter(self.p.propagate_in_video(self.state, start_frame_idx=self.t, max_frame_num_to_track=0)))
        self._prune()
        self._last = self._mask(m)
        return self._last, self.score()

    def reinit(self, box):
        """Re-initialise the object on the current frame from a box: segment it with no memory (as on the first
        frame), make it a conditioning frame, and forget recent memory (which carried the wrong object).
        A plain prompt on an already-tracked frame gets overridden by that memory."""
        s, k = self.state, self.t
        s["frames_already_tracked"].pop(k, None)
        for d in (s["output_dict"], s["output_dict_per_obj"][0]):
            for f in [f for f in d["non_cond_frame_outputs"] if f <= k]:
                d["non_cond_frame_outputs"].pop(f)
        if self.window:
            self.window.cell = None
        with torch.inference_mode():
            _, _, m = self.p.add_new_points_or_box(s, frame_idx=k, obj_id=1, box=np.asarray(box, np.float32))
        self._last = self._mask(m)
        return self._last

    def score(self):
        """EdgeTAM's own "object is visible" logit for the current frame (> 0 = visible)."""
        s, k = self.state, self.t
        for d in (s["temp_output_dict_per_obj"].get(0, {}), s["output_dict"]):
            for key in ("cond_frame_outputs", "non_cond_frame_outputs"):
                out = d.get(key, {}).get(k)
                if out is not None:
                    return float(out["object_score_logits"].flatten()[0])
        return 0.0

    def stored_frames(self):
        """How many frames' outputs are held in memory (should stay small)."""
        od = self.state["output_dict"]
        return len(od["cond_frame_outputs"]) + len(od["non_cond_frame_outputs"])

    # ------------------------------------------------------------ internals

    def _new_state(self, h, w):
        dev = self.p.device
        return {
            "images": {}, "num_frames": 10 ** 9, "offload_video_to_cpu": True, "offload_state_to_cpu": False,
            "video_height": h, "video_width": w, "device": dev, "storage_device": dev,
            "point_inputs_per_obj": {}, "mask_inputs_per_obj": {}, "cached_features": {}, "constants": {},
            "obj_id_to_idx": {}, "obj_idx_to_id": {}, "obj_ids": [],
            "output_dict": {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
            "output_dict_per_obj": {}, "temp_output_dict_per_obj": {},
            "consolidated_frame_inds": {"cond_frame_outputs": set(), "non_cond_frame_outputs": set()},
            "tracking_has_started": False, "frames_already_tracked": {},
        }

    def _push(self, img_bgr):
        """Preprocess like EdgeTAM's video loader: RGB, resize to 1024x1024, /255, ImageNet norm.
        The loader resizes with PIL (bicubic, 15 ms for a 720p frame); OpenCV's bicubic takes 1.7 ms and differs by
        0.3 grey levels on average. The normalisation runs on the GPU (it took ~10 ms in float64 on the CPU)."""
        size, dev = self.p.image_size, self.p.device
        rgb = cv2.cvtColor(cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
        if self._mean is None:
            self._mean, self._std = MEAN.to(dev), STD.to(dev)
        x = torch.from_numpy(rgb).to(dev).permute(2, 0, 1).float().div_(255).sub_(self._mean).div_(self._std)
        self.state["images"] = {self.t: x}  # only the current frame's image is ever needed

    def features(self):
        """The image features EdgeTAM computed for the current frame (to reuse them instead of encoding again)."""
        return self.state["cached_features"].get(self.t, (None, None))[1]

    def _prune(self):
        s, k = self.state, self.t
        for d in [s["output_dict"]] + list(s["output_dict_per_obj"].values()):
            for f in [f for f in d["non_cond_frame_outputs"] if f < k - self.KEEP_RECENT]:
                d["non_cond_frame_outputs"].pop(f)
            cond = sorted(d["cond_frame_outputs"])
            for f in cond[1:-(self.KEEP_COND - 1)] if len(cond) > self.KEEP_COND else []:
                d["cond_frame_outputs"].pop(f)
        for f in [f for f in s["frames_already_tracked"] if f < k - self.KEEP_RECENT]:
            s["frames_already_tracked"].pop(f)
        # EdgeTAM also indexes conditioning frames; drop pruned ones there too (otherwise, after more than
        # KEEP_COND re-initialisations, its consistency check fails: "assert frame_idx in cond_frame_outputs")
        s["consolidated_frame_inds"]["cond_frame_outputs"] &= set(s["output_dict"]["cond_frame_outputs"])
        for per_obj in (s["point_inputs_per_obj"], s["mask_inputs_per_obj"]):
            for inputs in per_obj.values():
                kept = set(s["output_dict"]["cond_frame_outputs"])
                for f in [f for f in inputs if f not in kept]:
                    inputs.pop(f)

    @staticmethod
    def _mask(m):
        return (m[0, 0] > 0).cpu().numpy()
