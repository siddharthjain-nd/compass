# Compass: research review and next direction

_2026-10-05. A companion to `PLAN.md`. It covers what limits us now, what the field offers as of late 2026, and which experiments come next._

**How sure each claim is:**
- **✔** I checked it here: a measurement on our M2, a look at the code, or the primary source.
- **◐** A research agent reported it from papers or READMEs, and I have not checked it myself. Treat ◐ numbers as leads, not facts.

---

## 1. The thesis in one paragraph

Today we ask one model (EdgeTAM) to do four jobs at once:
1. follow the object frame to frame,
2. decide what to remember about it,
3. decide *which* of several lookalikes it is,
4. know where it is when it can't be seen.

It does job 1 well. Its memory gets corrupted (job 2), it handles identity only through continuity (job 3), and it doesn't attempt job 4.

The 2025–26 literature points the same way. The biggest gains come from:
- **memory hygiene:** SAM 3, DAM4SAM and SENTRY all control *what gets written* to memory;
- **explicit state:** on the identical-cups shell game, VLMs score at chance, while a model that writes out object trajectories first scores over 90% (VET-Bench ✔).

So the next level is not a bigger model. It is **separating the four jobs and giving each the right tool**:
- a fast continuity tracker,
- a gated memory,
- an identity verifier,
- an explicit world-state graph that a VLM only puts into words.

---

## 2. Diagnosis: what limits us now (our own evidence)

| # | Limitation | Evidence | Status |
|---|---|---|---|
| D1 | **Identity of lookalikes depends on frame rate.** With identical objects, identity comes only from continuity. | Real cups clip: 3 cups keep their labels at 30 fps; one is lost at 15 fps; labels silently swap at 5 fps | ✔ |
| D2 | **Memory gets corrupted by holders.** When a hand grabs the object, the mask moves onto the hand, and the presence score stays above 0. | Synthetic `taken_*`: 22 of 45 frames on the hand; the score never drops while on the hand | ✔ |
| D3 | **Compute is set by memory work per object.** | M2, EdgeTAM + attention patch: 1 object 195 ms/frame (encoder 83, memory attention 59); 4 objects 480 ms on a 40-frame run (memory attention 230), and ~1 s/frame over the full 300-frame clip. The encoder is shared, and EdgeTAM already batches objects, so batching won't help further. | ✔ |
| D4 | **No instance identity yet.** Nothing yet checks "is this *my* wallet". Global embeddings alone are known to weaken once lookalikes are present. | PerMIS/PerVA benchmarks | ◐ |
| D5 | **No permanence reasoning.** "Into the bag" has to be inferred, and VLMs can't do it from raw video. | VET-Bench: VLMs at chance on a shell game with identical objects | ✔ |

**The Mac speed gap matters more than anything else.**
- D1 says identity needs about 30 fps.
- D3 says the Mac gives about 5 fps for one object and 1–2 fps for four.
- Every design decision below either raises fps or **stops identity from depending on fps**.

---

## 3. State of the field (late 2026), by job

### 3.1 Trackers: continuity and memory

| Work | What it adds | Relevance to us | |
|---|---|---|---|
| **SAM 3** (Meta, Nov 2025) | Detector + tracker on one backbone. Periodic re-prompting from confident detections, a memory that keeps only frames where the object is confidently present, and a presence head. MOSEv2 60.3 vs 47.9 (SAM 2.1-L). 848M params, CUDA-first, gated checkpoints. | The best-known recipe against memory drift (D2). Too heavy to run live on the M2. | ◐ |
| **SAM 3.1** (Mar 2026) | "Object Multiplex": up to 16 objects in one forward pass, 16→32 fps on an H100 | Directly targets D3, but it's a new architecture, not a patch for EdgeTAM. | ✔ |
| **DAM4SAM** (CVPR 2025) | Distractor-aware memory, as a drop-in memory-management change. DiDi Q 0.694 vs 0.649. A follow-up reports +4% when applied to EdgeTAM. | The most direct fix for lookalike swaps (D1). The repo is SAM 2.1 only, Apache-2.0; the EdgeTAM port isn't in it. | ✔ repo / ◐ numbers |
| **SENTRY** (ECCV 2026) | "Refine-before-write": checks each memory update for short-term consistency before storing it | Exactly D2: the mask moves while confidence stays high. | ◐ |
| **SAMURAI**, **HiM2SAM** | Kalman / point-track motion priors | Help at 30 fps. Constant-velocity priors break at 5 fps during fast swaps. | ◐ |
| **EdgeTAM** (ours) | About 14M params. 15.7 fps on iPhone 15 Pro Max via Core ML + Neural Engine. | Still the right base for edge devices. | ◐ |
| **TinySAM 2**, **EdgeDAM**, **EfficientSAM3** | Newer efficient variants | No code, a withdrawn paper, and no speed numbers, respectively. Not ready. | ◐ |

### 3.2 Identity: "is this *my* wallet?"

| Work | Key point | |
|---|---|---|
| **DINOv3** (Aug 2025) | Better instance retrieval than DINOv2 (reported for the 7B model; the gain for the small distilled models is unverified). Custom licence, needs review. DINOv2 is Apache-2.0. | ◐ |
| **Local matching beats global features on near-identical instances** | Animal re-ID: LightGlue/LoFTR beat trained global descriptors, and fusing both adds +8.5 pp. Frog photo-ID: 98% vs 60%. | ◐ |
| **IDOW** (CVPR 2025) | Fine-tune the embedding with **lookalike distractors as negatives** | ◐ |
| **SAM 3 exemplars** | An exemplar defines a *concept*: a box on one dog finds all dogs. Use it for candidates, never for identity. | ◐ |
| **Personalised VLMs** (R2P, RAP) | About 90% at best on "is this mine?", which is not enough to be the identity check | ◐ |

**Licence traps for commercial use:**
- **Non-commercial:** CoTracker3 (CC-BY-NC), MASt3R (CC BY-NC-SA), SAM2Long (CC-BY-NC), Magic Leap SuperPoint.
- **Use ALIKED/DISK + LightGlue** (Apache-2.0) instead.

### 3.3 Permanence: "where is it when hidden?"

- **TCOW** (CVPR 2023, MIT): the right *output design*, with masks for the target, its occluder and its container. It runs offline at 320×240 and is weak on real video. ◐
- **Diffusion amodal completion** (Diffusion-VAS, TABE): handles full occlusion, but runs offline on GPUs with 24–32 GB+. Useful only for labelling. ◐
- **VLMs and world models fail at containment:**
  - VET-Bench (identical cups): VLMs at chance. ✔
  - IntPhys 2: Gemini near chance. ◐
  - WROP and MemoBench (2026): world models lose hidden state. ◐
- **What works is explicit trajectories.** Molmo2 with "spatiotemporal grounded chain-of-thought" scores over 90% on VET-Bench. ✔

### 3.4 Language layer (VLM)

- **On-device candidates:** Qwen3-VL 2B/4B (Apache-2.0), Gemma 4 E2B/E4B, Qwen3.5 small. ◐
  - **Mac:** Qwen3-VL-4B 4-bit via mlx-vlm. Expect several seconds per event on an M2 (extrapolated).
  - **Jetson:** TensorRT Edge-LLM numbers exist. Orin Nano, 2B INT4: 32 tok/s, 1 s to first token. ◐
- **Avoid:** FastVLM (research-only licence) and Moondream 3 (BSL licence) as the main VLM.
- **What gets shipped:** the vision model writes a text/event log, an LLM answers questions over it, and it abstains when unsure. For example, Lando et al. 2025 match specialised SOTA on QAEgo4D with 3.6 kB/min of text memory. ◐
- **NVIDIA VSS "alert verification":** CV raises a candidate event, and the VLM confirms or rejects it. Our event-triggered design is the same idea. ◐

### 3.5 Hardware for the tracker

| Platform | Best evidence | |
|---|---|---|
| **Mac (M2)** | PyTorch MPS: our numbers. EdgeTAM's own Core ML export covers **image segmentation only, no video memory path**. A community full-video Core ML port claims about 10 fps (unverified). MLX gives about 1.3× over MPS. | ✔ export scope / ◐ rest |
| **Qualcomm** (QCS8550 / IQ-9075) | Qualcomm AI Hub ships EdgeTAM: about 17 ms per frame at int8 for encoder + memory encoder + decoder. **Memory attention runs on the CPU** (their code, checked 2026-10-09, §9.2). | ◐ numbers / ✔ CPU |
| **NVIDIA Jetson** | SAM2-base with TensorRT: about 6 fps on Orin Nano Super, 7 fps on Orin NX. A community SAM2.1 build on Thor: 26.5 fps for 1 object. **No published EdgeTAM TensorRT number.** We would have to measure. | ◐ |
| **Hailo / Rockchip** | Image-only SAM variants; the compilers choke on transformers. Detector use only. | ◐ |
| **Cloud GPU** | EdgeTAM at about 150 fps on an A100. But continuous 720p upload is 2–5 Mbps (≈1–2 GB/hour per camera), and it is a privacy problem for in-home video. | ◐ |

---

## 4. Proposed direction: four changes and the reasons for each

### 4.1 Fast/slow tracking: make identity independent of the heavy model's fps (fixes D1, D3)

> **Tested in E1 (2026-10-06): not supported on the cups clip.** See §8.1.

**Idea:**
- A **cheap tracker runs at 30 fps** on every frame. Optical flow or sparse point tracks inside each mask, plus a Kalman filter on the box. CPU cost is a few ms.
- **EdgeTAM runs at 5–10 fps** as the *corrector*, re-anchoring each mask on keyframes.
- **Identity is assigned at 30 fps** by matching EdgeTAM's keyframe masks to the flow-propagated masks (Hungarian matching on IoU).

**Why it should work:**
- D1 shows identity is lost *between* keyframes, when an object moves further than its own width.
- Flow at 30 fps never takes those big jumps.
- The research reports agree that motion priors help only when the frame rate is high enough. This design supplies that frame rate cheaply.

**Test:**
- On the cups clip with EdgeTAM at stride 6 plus 30 fps flow, count identity swaps.
- Compare with pure stride 1 (0 swaps) and pure stride 6 (several swaps).

### 4.2 Gated memory: only write what you trust (fixes D2)

> **Tested in E2 (2026-10-06):** the output check is adopted; memory gating is postponed until E4. See §8.2.

**Idea (SAM 3 / SENTRY spirit, simplified):** before a frame's mask enters EdgeTAM's memory bank, it must pass three checks:
1. area and shape are consistent with recent frames;
2. appearance matches the registered reference (DINO similarity above a threshold);
3. the mask does not substantially overlap a hand or holder mask.

A frame that fails is *used for output* but *not remembered*, and the object state switches to "held/occluded".

**Why:** drift onto the hand happens because hand frames get written to memory and then become "the object".

**Test:** on synthetic `taken_*`, the holder frames should drop from 22 to near 0. Then repeat on real hand-grab clips.

### 4.3 Two-stage identity verifier: global shortlist, local proof (fixes D4)

**Idea:**
- **Stage 1:** DINOv2 features pooled over the mask (try DINOv3 if its licence is OK) shortlist the candidates.
- **Stage 2:** ALIKED + LightGlue inlier count against registration close-ups (wear, stitching, logo) decides.
- **Calibrate** the fused score so that "70% yours" really means 70%.
- **At registration,** the user can add **known lookalikes as hard negatives**.

**Why:** fine-grained re-ID research consistently finds local geometric matching decisive for near-identical instances.

**Test:** two real, similar wallets, 50 crops each, varied angles and lighting. Metric: verification accuracy / ROC-AUC for DINOv2 vs DINOv3 vs +LightGlue.

### 4.4 Explicit world-state graph: permanence by bookkeeping, not by a VLM (fixes D5)

**Idea:** keep a graph of `object → holder/container → location`, updated by **contact and disappearance events**. When the object's track ends:
- **Classify how it ended,** using the TCOW-style trio target / occluder / container:
  - covered in place,
  - carried by a holder (the object's last point tracks move with the hand),
  - put in a container,
  - left the frame.
- **Find holders** with MediaPipe (hands) and **containers** with an open-vocabulary detector run only on events (SAM 3 text prompts or YOLO-World: "bag", "drawer", "box").
- **Follow the holder** as a stand-in for the hidden object. Re-verify identity when the object reappears.
- **The VLM gets the graph as text plus key frames** and only phrases or confirms it. It never infers the state itself.

**Why:** VET-Bench shows explicit trajectories take accuracy from chance to over 90%. Our tracker already produces those trajectories.

---

## 5. Experiment plan, in order

Each experiment is small (1–3 days), has one hypothesis, and is measured on the same clips.

| # | Experiment | Hypothesis | Metric | Needs |
|---|---|---|---|---|
| **E0** | **Benchmark set.** Cups clip (have it) plus ~10 real clips: 2 similar wallets, hand grab, bag, drawer, cloth. Hand-labelled identity at keyframes. | — | ID swaps, holder-drift frames, final-location accuracy, fps | Camera + your objects |
| **E1** ✔ done | **Fast/slow tracking** (§4.1) | Stride-6 EdgeTAM + 30 fps flow matches stride-1 identity at about ⅕ of the compute | swaps on cups, ms/frame | **Rejected** on the cups clip (§8.1) |
| **E2** ✔ done | **Gated memory** (§4.2) | Removes drift onto the hand | holder frames on `taken_*` + real grabs | **Output check adopted; memory gating postponed** (§8.2) |
| **E3** | **DAM-style distractor memory on EdgeTAM** (port from DAM4SAM, Apache-2.0) | Fewer swaps at 15 fps without extra cost | swaps at stride 2 and 6 | DAM4SAM code |
| **E4** | **Identity verifier** (§4.3) | Local matching separates two similar wallets where global features don't | ROC-AUC | 2 similar wallets |
| **E5** | **Speed for many objects** (§9): window memory attention, sleeping objects, encoder size, helpers at their own pace (E5a–e, §9.6) | Cost flat in N; ≥ 10 fps with 1 active object on the M2 | ms/frame vs N; E0 score; cups identity | nothing new |
| **E6** | **Event VLM.** Qwen3-VL-4B (mlx-vlm), given the tracker state as text vs given raw frames only | Tracker state turns wrong guesses into right ones on bag and hand events | accuracy on E0 events, seconds per event | ~4 GB download |
| **E7** | **SAM 3 / 3.1 reference run** (cloud GPU or offline) | An upper bound to compare against: how far is EdgeTAM + our fixes from the strongest tracker? | same metrics | GPU access, gated checkpoints |
| **E8** | **Edge bake-off.** Same pipeline and clips on Jetson (TensorRT) and/or Qualcomm (AI Hub EdgeTAM) | Meets the fps target for 1–4 objects | fps, identity metrics vs Mac | hardware access |
| **E9** | **Explain the change** (PLAN.md Phase 1b): blocked / turned / closer-farther / something else, from an expected outline + 5 cheap signals | Separable without a hand detector; never "turned" when blocked | label accuracy per frame on drawn scenes + blocks/mug replays + user clips; fps | nothing new |

**Order and why** (revised 2026-10-07 after the first live tests):
0. **E9 first:** the live tests showed users can't tell *why* the object is "not sure" (blocked vs turned vs moved closer). This is cheap and needs no new hardware or models. Then the hand detector.
1. **E1 and E2:** done (§8).
2. **E5, speed, now comes first.** E1 showed that nothing cheap replaces EdgeTAM frames during fast swaps of identical objects. About 15 fps of EdgeTAM is needed, and the M2 manages about 5 fps for one object.
3. **E0:** real clips. Every remaining hypothesis was tested only on one hard clip or on drawings.
4. **E4:** the identity check. It also unblocks memory gating (E2's variant C), which needs a way to confirm what it re-finds. The ambiguous cases left by E2b (both lookalikes handled at once) also need it.
5. ~~**Follow the hand on real video**~~: done (hand detector, E0 clips 99%). Next: **speed (E5)** and **"where is my wallet?" memory** (Phase 4).
6. **E3, E6**, then **E7 and E8** for the edge move.

---

## 6. Edge vs cloud: recommended split

| Runs where | What | Why |
|---|---|---|
| **On the device** (Mac now; Jetson/Qualcomm later), always on | capture, motion gating, fast/slow tracking, gated memory, identity verifier, world-state graph | Needs 30 fps and low latency. Video never leaves home. |
| **On the device**, on events | small VLM (Qwen3-VL 2B/4B) to describe or confirm | A few seconds per event is fine |
| **Cloud**, rarely | key frames + world-state for ambiguous events; free-form questions over long history | Bigger models help with *language*. They don't solve permanence, because frontier VLMs are at chance on the shell game too. |

**Not recommended:** streaming continuous video to the cloud.
- It costs 1–2 GB/hour per camera.
- It sends in-home video off-site.
- It doesn't fix D1/D5: identity and permanence still need the tracker's state.

**Hardware choice:** decide later, from what works on the Mac (the user's call, 2026-10-09). Known so far (§9.2): Jetson's TensorRT is the best-proven way to run this kind of tracker fast.
- **Qualcomm** (AI Hub EdgeTAM) has the best published encoder numbers, but its memory attention runs on the CPU.
- **Jetson** is the most flexible to develop on (PyTorch → TensorRT, same code).
- **Hailo and Rockchip** fit only as detector accelerators.

---

## 7. What would change this plan

- ~~**If E1 already fixes the cups swaps,** fast/slow tracking becomes the core architecture.~~ It didn't (§8.1), so DAM (E3) and raw speed (E5) stay on the plan.
- **If E5 finds no path to ≥ 10 fps on the M2,** the Mac stays a development box, and we move to edge hardware earlier.
- **If the window (E5a) loses accuracy,** try the published versions first (RS³-Prune, Efficient-SAM2); a short fine-tune with windows is the fallback.
- **If many objects must move at once** (not a home), joint memory with ID slots (AOT / SAM 3.1 Multiplex style) becomes worth training.
- **If a released efficient SAM 3 variant shows real edge numbers** (EfficientSAM3, or whatever follows SAM 3.1), re-evaluate the tracker choice. Its detector-based re-anchoring addresses D2 natively.

---

## 8. Results

### 8.1 E1: fast/slow tracking (real cups clip)

**Setup:**
- **Clip:** 3 identical red cups + a ball, 300 frames at 30 fps.
- **Reference:** EdgeTAM on every frame. I checked it by eye at 9 moments, and the ball is revealed under CUP 1, the cup that covered it.
- **Scoring:** a cup is correct on a frame if its mask overlaps the reference mask of the same cup (IoU ≥ 0.5), over 300 frames × 3 cups.
- **Code:** `experiments/e1_fast_slow.py`. Results page: `experiments/output/e1/index.html`.

**E1a: flow corrections at a fixed EdgeTAM rate** (all frames correct):

| EdgeTAM rate | alone | + flow, warp | + flow, shift | + perfect prior |
|---|---|---|---|---|
| 5 fps | 40% | 47% | 24% | **91%** |
| 10 fps | 61% | 67% | 25% | **95%** |
| 15 fps | **90%** | 71% | 26% | 98% |

**E1b: the same EdgeTAM compute, placed differently** (all frames correct):

| Budget | evenly spaced | where the motion is | every 3rd + fast moments |
|---|---|---|---|
| 33% of frames | 61% | 67% | (same as even) |
| 50% of frames | **90%** | 74% | 87% |

**What we learned:**
1. **The correction mechanism works.** Re-initialising a cup from a correct box takes 5 fps EdgeTAM from 40% to 91%.
   - A plain re-prompt doesn't work: EdgeTAM's memory overrides the box, so the cup's confused recent memory has to be cleared.
2. **Cheap optical flow is not a reliable enough prior.**
   - It carries cups well across most gaps: median IoU 0.9, and it fails in only 10 of 140 six-frame gaps.
   - But those failures are the fast, blurred swaps, which decide identity.
   - A correction from a wrong prior is worse than none.
3. **No frame-scheduling trick beat evenly spaced frames at 50% budget.**
   - This clip moves almost all the time (median 7.5 px/frame at ¼ resolution).
   - **A failed first attempt:** skipping frames inside one tracker session left gaps in EdgeTAM's frame-indexed memory (56–60%). It must only ever see the frames it processes.
4. **Compute on the M2** (4 objects, one tracker per object sharing the encoder): 15 fps EdgeTAM costs about 8 s of compute per 1 s of video. Real time needs under 1 s.
   - **Conclusion:** speed (E5), not cleverness, is the lever for fast identical-object shuffles.

**Limits:**
- One deliberately hard clip.
- The reference is EdgeTAM itself, verified by eye only at sampled moments.
- Motion gating may still save compute in **calm** scenes, which this clip doesn't have. Retest it on E0 clips.

### 8.2 E2: the "does it still look like my object?" check (synthetic scenes)

**Setup:**
- **The check** compares the tracker's mask with the object at frame 0, using size plus DINOv2 ViT-S/14 appearance. It uses no ground truth.
- **The rule:**
  - grown more than 1.5× → reject;
  - partial view (under 0.7× size) → show if similarity ≥ 0.75, never remember;
  - normal size → show and remember if similarity ≥ 0.85.
- **Code:** `experiments/e2_gated_memory.py`. Results page: `experiments/output/e2/index.html`.

| Scenario | A: tracker alone | **B: check on the output** | C: + check also gates memory |
|---|---|---|---|
| occlusion (key partly covered) | 30/30 | **30/30** | 25/30 |
| placed nearby / touching / hand covers | 40 / 40 / 40 | **40 / 40 / 40** | 40 / 40 / 40 |
| taken, part sticking out | 23/45 | 30/45 | 38/45, **5 frames on the other wallet** |
| taken, hidden in fist | 23/45 | **45/45** | 44/45, 1 frame on the other wallet |
| moved, other wallet present | 15/48 | **30/48** | 30/48, **18 frames on the other wallet** |
| moved, no other wallet (control) | 46/48 | 47/48 | 48/48 |

**What we learned:**
1. **Adopt B.**
   - It removes the "highlighting the hand" mistakes (taken, hidden: 23 → 45/45).
   - It causes no regressions and never picks the other wallet.
   - Its misses are honest: "can't see it" rather than a wrong object.
2. **C (memory gating) does what §4.2 predicted, and also something it didn't.**
   - **As predicted:** the tracker itself follows the hand far less (hand frames 22 → 7–9).
   - **Not predicted:** with a clean memory, the tracker searches again, and with a lookalike present it locks onto the **wrong** wallet.
   - Being confidently wrong is the worst failure for this product, so C waits for E4: a re-found object must pass the identity check before it is accepted.
3. **"Moved" with a lookalike is unsolved by A, B and C.** Appearance can't tell which wallet the hand carried. This needs holder tracking (§4.4, Phase 3).
4. **The rule needed one fix.** v1 used a single similarity threshold and wrongly rejected partial views (30 → 16 on occlusion). The partial-view threshold (0.75) was set after seeing these values, so real clips must confirm it.

**Limits:** drawn scenes only; thresholds not validated on real video yet.

### 8.3 E2b: follow the hand that took the object (synthetic scenes)

**Setup:**
- **Variant D** = B + a state per frame: visible / covered / **held** / out of view / unknown.
- **How "held" starts:** the tracker's mask suddenly grows onto something bigger. That bigger thing is the hand, and we follow it, using EdgeTAM's own drift onto it.
- **Hand-off rule:** while held, look for object-looking things where the hand is or just was. Ignore lookalikes already seen elsewhere, at registration or when the hand took the object. Confirm with the E2 appearance check, then restart the tracker on the find.
- **Finding object-looking things:** peaks of DINOv2 patch similarity, each outlined by SAM. No training.
- **Code:** `experiments/e2b_follow_holder.py`. Results page: `experiments/output/e2b/index.html`.

| Scenario | A | B | **D** | D's final answer |
|---|---|---|---|---|
| occlusion | 30/30 | 30/30 | **30/30** | key visible ✔ |
| placed nearby / touching / hand covers | 40 / 40 / 40 | 40 / 40 / 40 | **40 / 40 / 40** | wallet visible ✔ |
| taken, part sticking out | 23/45 | 30/45 | **45/45** | carried out of view to the left ✔ |
| taken, hidden in fist | 23/45 | 45/45 | **45/45** | carried out of view to the left ✔ |
| moved, other wallet present | 15/48 | 30/48 | **48/48** | wallet at its new spot (400, 120) ✔ |
| moved, no other wallet | 46/48 | 47/48 | 47/48 | wallet at its new spot ✔ |
| **All 8** | 257/336 (76%) | 302/336 (90%) | **335/336 (99.7%)** | **8/8** |

D puts **0 frames on the other wallet**.

**What we learned:**
1. **The hand path is what tells identical objects apart.** "Moved" with a lookalike went 15 → 48/48. The wallet that appears where the hand put something down is yours, and the identical one lying elsewhere is ignored.
2. **D can now say where the object went**, which A and B cannot: "carried out of view to the left", "at its new spot".
3. **v1 failed instructively.** One thresholded region per object-looking area merged two wallets 70 px apart, so the other wallet wasn't recorded as "already elsewhere" and was picked (25 wrong frames). Outlining each peak separately with SAM fixed it.

**Limits (important):**
- Drawn scenes with rectangle hands, and only 8 of them.
- The hand is followed by the tracker's drift. Real video needs a hand detector (e.g. MediaPipe), and real hands occlude, rotate and pass objects between each other.
- The rule assumes lookalikes stay put while the object is held. If both wallets are handled at once, or the hand puts yours right next to the other one, it must fall back to the identity check (E4) or say "ambiguous".
- Thresholds (peak height, 40 px reach, 6-frame memory) were set by design, and adjusted once after v1. Not tuned on real data.

---

### 8.4 E9: explain the change + hand detector (2026-10-07)

**Built:**
- `compass/explain.py`: blocked / turned / closer-farther / something else.
- `compass/hands.py`: MediaPipe hand landmarker.
- Both are wired into `compass/holder.py` and the live app, together with a mirrored front-camera view.

**Results:**

| Test | Result |
|---|---|
| Controlled edits of a real block image (14 cases) | 14/14; angles and scales measured within 1° / 0.01× |
| E9 drawn scenes (6 scenes, 264 scored frames) | **263/264, 0 critical** (never "turned" while blocked, never "blocked" while turned, never "taken by a hand" when moved closer, never claims the identical twin) |
| E2b drawn scenes (regression) | 335/336, 8/8 final answers |
| Blocks replay (real) | "taken by your right hand", then "put down at (806, 331)" ✔ |
| Mug replay (real) | all "visible" ✔ |
| Cups replay (real, identical, fast) | still mostly the wrong cup at live speed (11/31 right) ✘, a known limit (E1) |
| Hands | ~24 ms/frame; hands found in 75–100% of frames |

**What the iterations taught us** (each was found by tracing frame-by-frame logs):
1. **Explain *partial* views that still pass the look check, too.** A half-covered object often passes, and otherwise was never explained.
2. **Refresh the reference view only from complete views.** A view taken while a book was already sliding in made the book part of the "background".
3. **The best blocker signal is a change *just outside* the missing part.** A blocker reaches beyond the object; turning or flipping leaves the surroundings alone. Compare smoothed images, so pixel noise doesn't count.
4. **When the object is gone, only ask "is it still in its exact place?"** Searching nearby found the identical twin next to it.
5. **Identical objects pass any look check.** Only *where* they were seen tells them apart, so a jump onto a known lookalike spot is never claimed.
6. **A scale estimate must agree with the area change**, or a few matched details on a mask that spread onto a blocker read as "farther".
7. **In a hand-off, prefer the object's last spot and compare all candidates.** Taking the first that passed picked a wrong block.
8. **Remember lookalikes as spots, not outlines.** Outlining 40 with SAM cost 0.7–1 s at start-up and at the moment of a grab.

**Limits:**
- The thresholds were set on drawn scenes, and the left/right hand labels were checked on one video.
- **Not yet run on the real Mac camera.**

### 8.5 E0: the real-camera test + pocket (2026-10-09)

**Setup:**
- Six clips from the user's MacBook camera, replayed at 3 fps (the same frames every time), scored against labels.
- **Score: 86% → 99% (615/624) after fixes; 0 frames on the other wallet.**
- **The pocket feature** (clothes segmentation + hand path + "vanished at the clothes, then the hand moved away") takes the pocket clip from 60% to 100%.

**What the real clips taught that drawings and other people's videos did not:**
1. **Hands hold objects from below or the side.** The hand outline touched only 13–30% of the object, so "touching" needs a low threshold.
2. **Held objects look 20–45% bigger and less similar (0.70–0.80).** The angle changes too, and the leather's details are too faint to estimate the turn.
3. **A sheet of paper leaves a sliver visible**, and a shadow changes the colours of the visible part.
4. **The face-based pose detector fails when the camera sees only torso and legs.** Per-pixel clothes segmentation works.
5. **Long sessions need more tracker restarts.** That exposed a pruning bug in the streaming tracker.

**Limits:**
- One person, one room, one object type.
- The rules were tuned on these clips.
- Some labels are approximate (1-second resolution; one segment relabelled after the first run, documented in the script).

---

## 9. Speed for many objects (research + measurements, 2026-10-09)

**The question:**
- How do we track **many** objects (5–20 per camera) at **≥ 15 fps**?
- It has to work on edge hardware that is weaker than the M2.
- Nobody builds this system for one object, so cost must not grow with every object added.

**Method:**
- Four research passes ran in parallel: joint multi-object trackers, detect-and-match trackers, skipping repeated work, and edge runtimes.
- I checked the claims that the plan rests on myself (✔).
- I added one timing probe on our M2 (`experiments/e5_speed_probe.py`).

### 9.1 Where our time goes (✔ measured on the M2)

| Step | When it runs | ms |
|---|---|---|
| Image encoder (1024 px input) | once per frame, shared by all objects | **88** (768 px: 51; 512 px: 24) |
| Hands (MediaPipe, CPU) | once per frame | 24 |
| Clothes (pocket check) | only while an object is held | 120 |
| Finder search | only when lost / at a hand-off | ~200 |
| **Memory attention** | **every frame, per object** | **64** |
| Memory encoder + mask decoder | every frame, per object | ~21 + ~15 |
| Appearance check (DINOv2) | every frame, per object | ~19 |

**Per object per frame: about 120 ms. Today: 88 + 120 × N ms per frame.**

| Objects | Model time per frame | fps (model only) |
|---|---|---|
| 1 | ~210 ms | ~5 (live: 2.5) |
| 5 | ~690 ms | ~1.5 |
| 10 | ~1.3 s | < 1 |

**What the probe found inside memory attention:**
1. **It grows straight with the number of objects, even batched:** 1 object 64 ms, 2 objects 116, 4 objects 224, 8 objects 467.
2. **The cost comes from the queries, not from the memory.** Every one of the 64×64 = 4,096 grid cells of the *whole* frame attends to all other cells and to the memory.
   - Fewer memory frames barely helps: 7 → 4 frames is 64 → 54 ms.
   - Attending only to a window around the object cuts it hard:

     | Window (cells) | ms |
     |---|---|
     | 64×64 (whole frame) | 64 |
     | 32×32 | 14.6 |
     | **24×24** | **8.8** |
     | 16×16 | 5.6 |

   - **8 objects with 24×24 windows: 44 ms, against 467 ms.**
   - EdgeTAM already compressed the memory side (512 tokens per memory frame), so the whole-frame query grid is what is left.
   - **This is timing only.** Whether tracking stays accurate with a window is untested (§9.6, E5a).
3. **Half precision is slower on the Mac GPU.** Memory attention: 69 vs 64 ms. Encoder: 119 vs 88 ms. Don't use it on MPS.

### 9.2 What the field offers

**A. Trackers that handle many objects in one pass.** Only two families do this.
- **AOT / DeAOT** (NeurIPS 2021/2022, BSD-3):
  - Each object gets an "ID" vector, and all objects are tracked together.
  - The README says it handles up to 10 objects "as efficiently as processing a single object" ✔.
  - DeAOT-T: 63.5 fps on a V100, DAVIS17 80.5 ◐. That accuracy is well below EdgeTAM's 87.7.
- **SAM 3.1 Object Multiplex** (Meta, Mar 2026) ✔ (SAM 3 paper, Appendix H):
  - Objects go into buckets of 16. Memory encoding, memory attention and mask decoding run **once per bucket**, and per-object embeddings keep identities apart.
  - +7% time at 1 object; 5.2× faster than SAM 3 at 128 objects (H100).
  - But it has 848M parameters, so it is not an edge model. It also needs training, so it can't be bolted onto EdgeTAM.

**B. Everything else repeats the memory step per object.**
- That covers SAM 2, EdgeTAM, EfficientTAM, Cutie, XMem and the SAM 2-based multi-object trackers (SAM2MOT, Det-SAM2).
- The SAM 3 paper says this plainly about its own tracker: "compute and memory costs that grow linearly with the number of objects" ✔.
- Cutie-small: 45.5 fps at 1 object, 29.9 fps at 5 (V100) ◐.
- Cutie's own limitation section: it "often fails when highly similar objects move in close proximity" ◐.

**C. Cheaper memory step, without retraining.**
- **RS³-Prune** (BMVC 2026) ✔:
  - "read-sparse": only queries near the object take part;
  - "store-sparse": only the object's region enters memory;
  - result: up to +38.8% fps, accuracy competitive.
- **Efficient-SAM2** (ICLR 2026) ◐: 1.7–1.8× faster memory attention, −1.0 J&F, about 1 hour of training.
- **EfficientTAM** pooling of the memory ◐: 2.2× on iPhone, −0.5 J&F.
- Our probe suggests read-sparse gains much more on EdgeTAM (7×) than the published 1.4×. The published numbers are on SAM 2, where the memory side is still large.

**D. Do heavy work only when something changes.**
- **MARLIN** (SenSys 2019, mobile AR) ◐: runs the network only for new objects or big changes, with a cheap tracker in between. Up to 73% less energy than running the network on every frame.
- **Reducto** (SIGCOMM 2020) ◐: cheap frame differences filter 51–97% of frames.
- **HiM2SAM** ◐: runs its expensive step only on low-confidence frames. 3.7 ms on average against 166 ms on every frame.
- **DeltaCNN** (CVPR 2022) ◐: recomputes only changed pixels, 3–7× with a static camera. CUDA-only kernels, so not for the Mac or NPUs.

**E. Detect-and-match (a detector every frame, then link boxes over time).**
- Speed:
  - YOLO26n: 4–5 ms on Jetson Orin NX / Nano with TensorRT (◐ vendor numbers).
  - OC-SORT linking: ~1 ms.
  - Cost is almost flat in N.
- **YOLOE** (ICCV 2025) ◐ can be prompted with a box on a reference photo. But it finds "a wallet", not "my wallet": still class-level.
- What it gives up: masks; objects inside a hand (box merges with the hand); instance identity.
- It is built on Ultralytics, which is **AGPL-3.0** ◐. Check before any product use.

**F. Fast box trackers (one per object, searching a crop).**
- LiteTrack: 102 fps on Orin NX ◐. AsymTrack: ~80 fps ◐.
- Cost is still per object, and they have no way to re-find an object after a full occlusion. Not for us.

**G. Edge runtimes.**

| Platform | What we found | |
|---|---|---|
| **Jetson + TensorRT** | SAM 2.1 (TinyViT encoder) on Thor: 3.8 fps in PyTorch → **26.5 fps** with TensorRT FP16 (6.9×), masks ≥ 0.99 identical. Batching 2–4 objects was *slower* per object. | ✔ repo |
| **Qualcomm AI Hub EdgeTAM** | Encoder 3–11 ms int8 on the NPU ◐, but **memory attention runs on the CPU** ("Memory attention always runs on CPU via sam2"). | ✔ code |
| **Apple Core ML** | Community ports of the video path run on CPU + GPU because parts don't compile for the Neural Engine. int8 speedups need A17 Pro / M4; on the M2, quantisation mostly saves memory. | ◐ |
| **Quantisation accuracy** | SAM2-tiny on DAVIS (CAR-SAM): 89.1 (full precision) → 87.6 (6-bit weights and activations) → 85.0 (4-bit). Small models hold up better. No published 8-bit video-tracking accuracy for EdgeTAM. | ◐ |

**The memory step is the weak point on every edge platform:**
- It is the part that grows per object.
- No vendor runs it on an NPU today.

### 9.3 What this means for us

1. **The slope matters, not the runtime.** A faster runtime makes each object cheaper, but cost still grows with every object. We have to change what runs per object.
2. **Most of the per-object work is spent far from the object.** The memory step processes all 4,096 cells of the frame for a wallet that covers maybe 5–10% of it.
3. **Most objects are at rest most of the time.** Resting objects can skip work.
   - ~~A person has two hands, so at most ~2 objects move at once.~~ **Wrong** (user, 2026-10-09). Several people can be in view and move many objects at once.
   - So the cost of each *moving* object must also be small. Skipping resting objects alone is not enough.

**Design rule:** do shared work once per frame. Do per-object work **only for objects that are active**, and **only around the object**.

### 9.4 Proposed changes, ranked by gain for effort

| # | Change | What it does | Expected gain | Risk |
|---|---|---|---|---|
| 1 | **Resting objects sleep** | An object with no hand near it and no pixel change inside its box runs no tracker and no check. It keeps its last mask, with a DINOv2 check every ~2 s. It wakes on a pixel change in its box (frame difference, < 1 ms), a hand within reach, or a failed check. | Per-object cost × **moving** objects instead of × all objects. An idle scene costs ~hands only. | A change we can't see. Mitigated: an arriving occluder is itself a pixel change. |
| 2 | **Look only around the object** | Memory attention uses a fixed window (e.g. 24×24 cells, ~2–3× the object) centred on the last mask, with the cells' true positions. Fall back to the whole frame when the object is lost or searched for. | 64 → 9 ms per object (✔ timing). Fixed shapes also help TensorRT / Neural Engine / NPU compilers. | Accuracy untested. A fast throw may leave the window (→ whole frame on low score). |
| 3 | **Cheaper image encoder** | 768 px input (51 ms) or 512 px (24 ms) instead of 1024 (88 ms). Skip the encoder entirely when every object sleeps. | 1.7–3.7× on the shared part | SAM 2 paper: −1.8 to −4.0 J&F at 512 px ◐. Small objects suffer. |
| 4 | **Helpers at their own pace** | Hands in a parallel thread at 10–15 Hz. Appearance check every 3rd frame and on events. Clothes and finder only on events (already). | ~40 ms off each frame | Small |
| 5 | **Window the rest of the per-object path** | Apply the same window to the memory encoder and mask decoder (RS³ "store-sparse"). After #2 these (~36 ms) are the largest per-object cost. | Est. 2–3× on that ~36 ms | Accuracy untested |
| 6 | **Edge runtime** | Jetson + TensorRT FP16. Separate engines for encoder, windowed memory attention, decoder and memory encoder, at fixed shapes. | ~5–7× over PyTorch (Thor measurement) ◐ | Shapes must be fixed; we design for that in #2 |

**Later, only if needed:**
- **Joint memory for many *active* objects** (AOT / Multiplex-style ID slots inside EdgeTAM).
  - Needs training on SA-V.
  - Pays off only when many objects move at once (a warehouse, not a home).
- **A small detector as the re-finder.** It would replace the 200 ms finder search when an object is lost, with our appearance and detail checks still deciding "is it mine". Check the licence first.

**Not recommended:**
- Box trackers per object (cost per object, can't re-find).
- Detect-and-match as the main tracker (no masks, fails on held objects, class-level only).
- DeltaCNN (CUDA-only).
- Token merging (EdgeTAM's encoder is a CNN; DINOv2-S is already small).
- Half precision on the Mac GPU (measured slower).

### 9.5 Budget (model time per frame on the M2; k = active objects; N = all objects)

| Setup | Formula (ms) | k = 1 | k = 2 | Depends on N? |
|---|---|---|---|---|
| Today | 88 + 120 × **N** | ~210 (N = 1) | — | **yes** |
| + #1 sleep + #2 window | 88 + 64 × k | ~150 (6.6 fps) | ~215 | no |
| + #3 768 px + #4 helpers | 51 + 51 × k | ~100 (**~10 fps**) | ~155 (~6.5 fps) | no |
| + #5 window the rest (est.) | 51 + ~30 × k | ~80 (~12 fps) | ~110 (~9 fps) | no |

**What the budget says:**
- **The big win is not a number in this table: cost stops depending on N.** Ten resting objects cost about the same as one.
- **The Mac (PyTorch) gets to about 10 fps.** That is enough to develop on, but below the 15 fps that identical objects need.
- **The 15 fps target needs the TensorRT step (#6).** Its 5–7× is measured on SAM 2 on a different device (◐), so it has to be measured on a Jetson (E8).
- **For the later hardware choice:** Qualcomm's memory step runs on the CPU, and Apple's doesn't compile for the Neural Engine in public ports. Jetson + TensorRT is the best-proven route. The user decides later.

### 9.6 Experiment plan (replaces the old E5 line)

| # | Test | Pass if | Data |
|---|---|---|---|
| **E5a** | Window memory attention (#2) in `StreamingTracker`, true cell positions, whole-frame fallback | E0 score ≥ 98% (now 99%); mask IoU vs whole-frame ≥ 0.9; cups identity at 15 fps no worse | E0 clips, drawn scenes, cups |
| **E5b** | Sleeping objects (#1), with several registered objects | ms/frame flat as N grows 1 → 10; no missed pick-up | new clip with 3–5 objects + blocks replay |
| **E5c** | Encoder input 1024 / 768 / 512 (#3) | E0 ≥ 98% at the chosen size | E0 clips, cups |
| **E5d** | Helpers at their own pace (#4) + live multi-object view | ≥ 10 fps live with 1 active object, flat in N | live camera |
| **E5e** | Window the memory encoder + decoder (#5) | same as E5a | same as E5a |
| **E8** | Jetson + TensorRT, fixed shapes | ≥ 15 fps with 2 active + 10 resting | same clips on the device |

**Order:** superseded by §9.8. The user's priority is one object faster first.

### 9.7 How sure each part is

- **✔ Checked here:**
  - our timings (§9.1);
  - Multiplex design and speed (SAM 3 paper App. H);
  - the AOT README claim;
  - Qualcomm's memory step on CPU (their code);
  - the Thor TensorRT numbers and batching result (repo);
  - the RS³-Prune abstract.
- **◐ Reported, not checked:** all other fps and accuracy numbers above.
  - Edge numbers come from different devices and are not directly comparable.
  - The EfficientTAM paper contradicts itself on iPhone fps.
  - EdgeDAM, a close match to our hybrid, was withdrawn, so it is not used.
- **Estimates:** the budget rows after "today" and the TensorRT factor for our pipeline.

### 9.8 One object, faster first (user's priorities, 2026-10-09)

**The user's corrections:**
1. **Goal (clarified 2026-10-11):** multiple objects eventually, but not necessarily all tracked at the same moment. First make **one object fast**. Then add objects one after another, or side by side, whichever works.
2. Cost may grow per object, but it must stay modest. It must not explode.
3. Edge hardware is **not decided**.
4. Many objects can move at once (several people).

**Full system, one object, measured** on the six E0 clips (665 frames, every part timed; `experiments/e5_time_breakdown.py` → `experiments/output/e5/time_breakdown.json`).

| ms per frame | wallet visible (257 frames) | wallet in hand (330 frames) |
|---|---|---|
| Tracker (EdgeTAM), incl. 29 ms preparing the picture on the CPU | 253 | 289 |
| Hands | 36 | 38 |
| Appearance check ("is this outline still my wallet?") | 21 | 52 |
| Explainer (blocked / turned / closer) | 69 | 9 |
| Finder search ("where in the picture could it be?") | 15 (runs on 4% of frames) | **336** (every frame: scan 76 + outlines 260) |
| Clothes (pocket) | 0 | 56 (runs on 27% of frames, ~200 ms each time) |
| Rest of the logic | 10 | 33 |
| **Total** | **404 ms → 2.5 fps** | **813 ms → 1.2 fps** |

- The tracker alone, with nothing else running, takes 218 ms (4.6 fps).
- In the full system it takes 253–289 ms, because the other models share the GPU and the tracker restarts while the wallet is held.

**Fixes for one object, by gain (estimates until measured):**

| Fix | Saves | Risk |
|---|---|---|
| Prepare the picture on the GPU, not in CPU float64 | ~25 ms | none |
| Hands in parallel with the tracker (CPU vs GPU) | ~36 ms | none |
| In hand: search every 3rd frame (flag exists), or a fast detector + appearance check instead of 10 outlines | ~220–300 ms in hand | put-down noticed up to 2 frames later |
| Clothes at most once per second while the wallet is hidden in the hand | ~50 ms in hand | pocket noticed up to 1 s later |
| Window memory attention (§9.1) | ~55 ms | accuracy untested (E5a) |
| 768 px picture instead of 1024 | ~37 ms | small objects (E5c) |
| Appearance check / explainer only when the look changes, or on a smaller crop | ~20–50 ms | small |
| Core ML for the tracker (Apple's own route; EdgeTAM runs 16 fps on an iPhone 15 Pro Max with it ◐) | unknown, maybe 2–3× on the tracker | memory step may not run on the Neural Engine |

**Expected result:**
- With PyTorch on the Mac: visible 404 → ~200 ms (~5 fps), in hand 813 → ~300 ms (~3 fps).
- Beyond that the tracker dominates. 15 fps needs a faster way to run it: Core ML on the Mac, or TensorRT on a Jetson.

**New order:**
1. The no-risk fixes: picture on the GPU, hands in parallel, search and clothes less often.
2. E5a window, E5c picture size.
3. Core ML test.
4. Sleeping objects (E5b) when multiple objects come.

**Results of steps 1–3** (2026-10-11; full system, one object, 665 frames of the E0 clips):

| | Before | After |
|---|---|---|
| Wallet visible | 404 ms (2.5 fps) | **217–224 ms (4.5–4.6 fps)** |
| Wallet in hand | 813 ms (1.2 fps) | **452–478 ms (2.1–2.2 fps)** |
| All frames | 609 ms (1.6 fps) | **351–366 ms (2.7–2.85 fps)** |
| Live replay of clip 2 (`compass.live`) | ~2.5 fps | **3.9 fps** |
| E0 score | 615/624 | **616/624** (no frames on the other wallet; pocket clip 100%) |

Ranges are two runs of the final code. Other apps were open, so timings vary ±15%. Step-by-step log: `PROGRESS.md`.

**What each change did:**

| Change | Effect | Accuracy |
|---|---|---|
| Picture resize with OpenCV instead of PIL, normalise on the GPU | prep 29 → 7 ms | 0.3 grey levels different. Cost 3 borderline E0 frames (wallet turned in hand, look-match right at the threshold) |
| Hands in parallel with the tracker | ~30 ms hidden behind the tracker | none |
| Search outlines reuse the tracker's image features (no second encoding; full-resolution outlines) | in-hand outlines 260 → ~130 ms | +2 frames (sharper outlines) |
| Clothes result reused for up to 1 s | in hand 56 → ~12 ms | none (pocket clip 100%) |
| Search candidates checked in one batch | appearance in hand 52 → ~40 ms | none |
| **Skip while nothing changes** (`ObjectState.skip_when_still`) | skipped 26–31% of visible frames on E0; a skipped frame costs ~18 ms instead of ~300 | +3 frames |
| **Window** (`compass/window.py`): memory step + mask decoder on a square around the last mask (box + half its size per side) | tracker 229 → 164 ms (memory step 62 → 22 ms) | tracker-only masks vs whole frame: mean IoU 0.989, 0.6% of frames < 0.8. E0 −1 frame |

**Tried and dropped:**
- **Tighter window** (quarter-box margin): IoU 0.984, and 7% of frames below 0.8 on the fast pocket move, for little extra speed.
- **Outlining only search peaks near the hand:** a fixed 80 px margin lost a close-up wallet (clip 4, −9 frames). A size-based margin covered almost everything, so it saved nothing.

**Why our window gain is smaller than the probe's 64 → 9 ms:** the wallet is big in these clips (~19 of 64 cells across), so its window is 36–48 cells. Close-ups use the whole frame. Small objects will gain more.

**What is left** (in hand ~500 ms):

| Part | ms |
|---|---|
| Tracker | ~200 (image encoder 88, memory encoder 23, memory step ~22, decoder ~15) |
| Search outlines | ~130 (17 ms per candidate, 10 candidates) |
| Search scan | ~65 |
| Appearance check | ~40 |

**Next:**
- outline candidates in small windows, or the fast detector (§9.2 E);
- 768 px input (E5c);
- Core ML test.

**Multi-object design: watch cheaply, track accurately on wake** (the user's idea, 2026-10-11; same as #1 + #2 in §9.4):

| Object state | What runs | Cost |
|---|---|---|
| **Resting** | A pixel-change check inside its box (+ the shared hand detector), and an appearance check every ~2 s | ~1 ms for all resting objects together |
| **Woken** (pixels in its box changed, or a hand came near) | The accurate tracker, only in a window around the object, until it rests again (still ~1 s and looks right) | ~120 ms per moving object today, ~40–50 target |
| **Lost / carried away** | Finder search, later a fast detector + appearance check | event only |

- **Example:** 10 objects, 2 moving.
  - Tracking all of them today: 88 + 10 × 120 ≈ 1.3 s per frame.
  - With waking: 88 + 2 × 120 ≈ 330 ms.
  - With waking and windows: about 190 ms.
- **A YOLO-type detector is not needed for "did something move in this box".** Comparing the box's pixels with the last frame does that almost for free. The detector's job is re-finding an object that left its box.
- **Risks:**
  - Camera moved or light changed: everything wakes, so fall back to round-robin.
  - Two identical objects moving together: both must be tracked at full rate, so cost peaks exactly then.
  - An object slid while covered: the cover arriving is itself a change, so the object wakes.
- **Research using the same pattern:**
  - MARLIN (SenSys 2019): heavy model only on change, up to 73% less energy ◐.
  - Reducto (SIGCOMM 2020): 51–97% of frames skipped ◐.
  - HiM2SAM: expensive step only when unsure ◐.

## 10. Sources

**Checked here (✔):**
- VET-Bench, "Can Vision-Language Models Solve the Shell Game?" (Mar 2026): https://arxiv.org/abs/2603.08436
- SAM 3 / SAM 3.1 Object Multiplex: https://ai.meta.com/blog/segment-anything-model-3/
- DAM4SAM (Apache-2.0, SAM 2.1): https://github.com/jovanavidenovic/DAM4SAM
- EdgeTAM Core ML export scope (image path only): `third_party/EdgeTAM/coreml/README.md`

**Reported by research, not checked by me (◐):**
- SAM 3: https://arxiv.org/abs/2511.16719, https://github.com/facebookresearch/sam3
- Memory designs on SAM 3: https://arxiv.org/abs/2512.22624
- SENTRY: https://hamadya.github.io/SENTRY/page/
- DAM on EdgeTAM / IJCV: https://arxiv.org/abs/2509.13864
- SAMURAI: https://github.com/yangchris11/samurai
- HiM2SAM: https://github.com/LouisFinner/HiM2SAM
- TCOW: https://github.com/basilevh/tcow
- TABE: https://github.com/finlay-hudson/TABE
- MOSEv2: https://arxiv.org/abs/2508.05630
- DINOv3: https://github.com/facebookresearch/dinov3
- IDOW: https://arxiv.org/abs/2503.00359
- WildFusion: https://arxiv.org/abs/2408.12934
- R2P: https://arxiv.org/abs/2503.18623
- RELOCATE: https://arxiv.org/abs/2412.01826
- Text memory (Lando et al.): https://arxiv.org/abs/2506.16450
- IntPhys 2: https://arxiv.org/abs/2506.09849
- EdgeTAM paper: https://arxiv.org/html/2501.07256
- Qualcomm AI Hub EdgeTAM: https://huggingface.co/qualcomm/EdgeTAM
- EdgeTAM-CoreML-Video: https://huggingface.co/AhmadYarAI/EdgeTAM-CoreML-Video
- Jetson Super mode: https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules
- Efficient-SAM2-TensorRT: https://github.com/thedannyliu/Efficient-SAM2-TensorRT
- TensorRT Edge-LLM benchmarks: https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/performance/performance-benchmarks.html
- NVIDIA VSS: https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization
- SAM 2 paper (resolution ablation): https://arxiv.org/pdf/2408.00714

**Speed research, §9 (2026-10-09):**
- ✔ SAM 3 paper, Appendix H "Object Multiplex": https://arxiv.org/html/2511.16719v2
- ✔ SAM 3.1 release notes: https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md
- ✔ AOT / DeAOT (BSD-3): https://github.com/yoxu515/aot-benchmark · ◐ papers https://arxiv.org/abs/2106.02638, https://arxiv.org/abs/2210.09782
- ✔ Qualcomm AI Hub EdgeTAM app (memory attention on CPU): https://github.com/quic/ai-hub-models/blob/main/src/qai_hub_models/models/edgetam/app.py
- ✔ Efficient-SAM2-TensorRT (Jetson Thor): https://github.com/thedannyliu/Efficient-SAM2-TensorRT
- ✔ RS³-Prune (BMVC 2026): https://arxiv.org/abs/2608.22526
- ◐ Cutie: https://arxiv.org/abs/2310.12982 · XMem: https://github.com/hkchengrex/XMem
- ◐ EfficientTAM: https://arxiv.org/abs/2411.18933 · Efficient-SAM2: https://arxiv.org/html/2602.08224
- ◐ EfficientSAM3: https://github.com/SimonZeng7108/efficientsam3
- ◐ LiAM-SAM (SAM 2 multi-object cost): https://arxiv.org/abs/2609.28078
- ◐ MARLIN: https://par.nsf.gov/servlets/purl/10201676 · Reducto: https://www.cs.princeton.edu/~ravian/publications/reducto_sigcomm20.pdf
- ◐ HiM2SAM: https://arxiv.org/html/2507.07603v1 · DeltaCNN: https://arxiv.org/abs/2203.03996
- ◐ YOLOE: https://arxiv.org/abs/2503.07465 · Ultralytics Jetson numbers: https://docs.ultralytics.com/guides/nvidia-jetson/ · OC-SORT: https://arxiv.org/abs/2203.14360
- ◐ LiteTrack: https://arxiv.org/abs/2309.09249 · AsymTrack: https://arxiv.org/abs/2503.00516
- ◐ Core ML quantisation: https://apple.github.io/coremltools/docs-guides/source/opt-quantization-perf.html
- ◐ CAR-SAM (quantised SAM 2): https://www.alphaxiv.org/abs/2605.16901 · Q-SAM2: https://arxiv.org/abs/2506.09782
- ◐ SAM 2 resolution / memory ablations: https://arxiv.org/abs/2408.00714
