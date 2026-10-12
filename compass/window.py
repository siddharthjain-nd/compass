"""Track only around the object: EdgeTAM's memory step and mask decoder run on a square window of its 64x64
feature grid instead of the whole frame (E5a, RESEARCH.md §9).

Why: on an M2, the memory step costs ~64 ms per object, almost all of it spent on the whole-frame query grid
(4,096 cells attending to each other and to the memory), while the object covers a small part of it. A 24x24
window measured 9 ms (experiments/e5_speed_probe.py).

How (on frames tracked from memory only; prompted frames and re-initialisations use the whole frame):
  memory step   only the window's cells are queries. Self-attention uses rotary positions, which depend only on
                the distance between cells, so a window behaves like the same cells inside the full grid. The
                cross-attention to the memory gets each cell's true position (sliced from the full 64x64 table).
  mask decoder  runs on the window's features, with the true positional encodings and the matching high-
                resolution features; the mask outside the window is "no object" (as EdgeTAM writes for absent
                objects).
Everything else (image encoder, memory encoder, memory bank) is unchanged.

The window is a square around the last mask, or around a box the caller gives (where the object should be: its last
spot while covered, the hand holding it), with `margin` of its size on each side (at least `min_cells`); or the whole
frame when there is nothing to centre on or the window would cover most of the grid anyway.
"""
import math

import torch
import torch.nn.functional as F

GRID = 64  # EdgeTAM's feature grid (1024 px input / 16)


class Window:
    def __init__(self, predictor, margin=0.5, min_cells=16, max_share=0.75):
        from sam2.modeling.sam2_base import NO_OBJ_SCORE

        self.p, self.margin, self.min_cells, self.max_cells = predictor, margin, min_cells, int(max_share * GRID)
        self.no_obj = NO_OBJ_SCORE
        self.cell = None  # (x0, y0, size) in grid cells, or None: whole frame
        self.enabled = True
        self._armed = False  # the memory step ran on the window this step, so the decoder must too
        ma = predictor.memory_attention
        self._ma_forward = ma.forward
        self._cross = [layer.cross_attn_image for layer in ma.layers]
        self._full_q = [c.freqs_cis_q for c in self._cross]
        self._sliced = {}
        ma.forward = self._memory_attention
        self._heads = predictor._forward_sam_heads
        predictor._forward_sam_heads = self._sam_heads
        self.windowed = self.full = 0  # frames tracked each way (for reports)

    # ------------------------------------------------------------------ choosing the window

    def set_from_mask(self, mask):
        """Window around this (video-resolution, bool) mask, or the whole frame."""
        self.cell = None
        if not self.enabled or mask is None or not mask.any():
            return
        ys, xs = mask.nonzero()
        self.set_from_box((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1), mask.shape)

    def set_from_box(self, box, hw):
        """Window around this (x0, y0, x1, y1) box in video pixels (frame size hw), or the whole frame."""
        self.cell = None
        if not self.enabled:
            return
        h, w = hw
        x0, x1 = box[0] * GRID / w, box[2] * GRID / w
        y0, y1 = box[1] * GRID / h, box[3] * GRID / h
        side = max(x1 - x0, y1 - y0)
        size = max(self.min_cells, math.ceil(side * (1 + 2 * self.margin)))
        size = 4 * math.ceil(size / 4)  # few distinct sizes
        if size > self.max_cells:
            return
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        gx = min(max(round(cx - size / 2), 0), GRID - size)
        gy = min(max(round(cy - size / 2), 0), GRID - size)
        self.cell = (gx, gy, size)

    def _index(self, device):
        gx, gy, s = self.cell
        ys = torch.arange(gy, gy + s, device=device)
        xs = torch.arange(gx, gx + s, device=device)
        return (ys[:, None] * GRID + xs[None, :]).reshape(-1)

    # ------------------------------------------------------------------ memory step

    def _memory_attention(self, curr, memory, curr_pos=None, memory_pos=None, num_obj_ptr_tokens=0,
                          num_spatial_mem=-1):
        if self.cell is None:
            self.full += 1
            return self._ma_forward(curr, memory, curr_pos, memory_pos, num_obj_ptr_tokens, num_spatial_mem)
        c, cp = (curr[0], curr_pos[0]) if isinstance(curr, list) else (curr, curr_pos)
        idx = self._index(c.device)
        key = self.cell
        if key not in self._sliced:
            self._sliced = {key: [full.to(c.device)[idx] for full in self._full_q]}  # true positions of the cells
        for cross, q in zip(self._cross, self._sliced[key]):
            cross.freqs_cis_q = q
        try:
            out_w = self._ma_forward(c[idx], memory, cp[idx], memory_pos, num_obj_ptr_tokens, num_spatial_mem)
        finally:
            for cross, full in zip(self._cross, self._full_q):
                cross.freqs_cis_q = full
        out = c.new_zeros((GRID * GRID,) + tuple(out_w.shape[1:]))  # outside the window: never read (see decoder)
        out[idx] = out_w
        self._armed = True
        self.windowed += 1
        return out

    # ------------------------------------------------------------------ mask decoder

    def _sam_heads(self, backbone_features, point_inputs=None, mask_inputs=None, high_res_features=None,
                   multimask_output=False):
        if not self._armed or point_inputs is not None or mask_inputs is not None:
            return self._heads(backbone_features, point_inputs, mask_inputs, high_res_features, multimask_output)
        self._armed = False
        p = self.p
        gx, gy, s = self.cell
        B, dev = backbone_features.size(0), backbone_features.device
        feats = backbone_features[:, :, gy:gy + s, gx:gx + s].contiguous()
        hr = None
        if high_res_features is not None:
            hr = [high_res_features[0][:, :, 4 * gy:4 * (gy + s), 4 * gx:4 * (gx + s)].contiguous(),
                  high_res_features[1][:, :, 2 * gy:2 * (gy + s), 2 * gx:2 * (gx + s)].contiguous()]
        coords = torch.zeros(B, 1, 2, device=dev)
        labels = -torch.ones(B, 1, dtype=torch.int32, device=dev)
        sparse, dense = p.sam_prompt_encoder(points=(coords, labels), boxes=None, masks=None)
        dense = dense[:, :, gy:gy + s, gx:gx + s].contiguous()
        pe = p.sam_prompt_encoder.get_dense_pe()[:, :, gy:gy + s, gx:gx + s].contiguous()
        low_w, ious, tokens, obj_logits = p.sam_mask_decoder(
            image_embeddings=feats, image_pe=pe, sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            multimask_output=multimask_output, repeat_image=False, high_res_features=hr)
        if p.pred_obj_scores:
            is_obj = obj_logits > 0
            low_w = torch.where(is_obj[:, None, None], low_w, self.no_obj)
        low = torch.full((B, low_w.shape[1], 4 * GRID, 4 * GRID), self.no_obj, device=dev)
        low[:, :, 4 * gy:4 * (gy + s), 4 * gx:4 * (gx + s)] = low_w.float()
        high = F.interpolate(low, size=(p.image_size, p.image_size), mode="bilinear", align_corners=False)
        token = tokens[:, 0]
        if multimask_output:
            best = torch.argmax(ious, dim=-1)
            rows = torch.arange(B, device=dev)
            low_best, high_best = low[rows, best].unsqueeze(1), high[rows, best].unsqueeze(1)
            if tokens.size(1) > 1:
                token = tokens[rows, best]
        else:
            low_best, high_best = low, high
        obj_ptr = p.obj_ptr_proj(token)
        if p.pred_obj_scores:
            lam = obj_logits.sigmoid() if p.soft_no_obj_ptr else is_obj.float()
            if p.fixed_no_obj_ptr:
                obj_ptr = lam * obj_ptr
            obj_ptr = obj_ptr + (1 - lam) * p.no_obj_ptr
        return low, high, ious, low_best, high_best, obj_ptr, obj_logits
