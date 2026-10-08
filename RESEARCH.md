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
| **Qualcomm** (QCS8550 / IQ-9075) | Qualcomm AI Hub ships EdgeTAM: about 17 ms per frame at int8 for encoder + memory encoder + decoder. The best-evidenced edge number. It's unclear whether memory attention fully runs on the NPU. | ◐ |
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
| **E5** | **Speed on Mac.** Core ML full video path (community port) vs MPS vs lower input resolution | ≥ 10 fps for 1 object on the M2 | ms/frame and accuracy drop | coremltools |
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
5. **Follow the hand on real video:** E2b worked on drawings; add a hand detector and test on E0 clips.
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

**Hardware choice:** decide with the E8 bake-off, not up front.
- **Qualcomm** (AI Hub EdgeTAM) has the best published numbers.
- **Jetson** is the most flexible to develop on (PyTorch → TensorRT, same code).
- **Hailo and Rockchip** fit only as detector accelerators.

---

## 7. What would change this plan

- ~~**If E1 already fixes the cups swaps,** fast/slow tracking becomes the core architecture.~~ It didn't (§8.1), so DAM (E3) and raw speed (E5) stay on the plan.
- **If E5 finds no path to ≥ 10 fps on the M2,** the Mac stays a development box, and we move to edge hardware earlier.
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

## 9. Sources

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
