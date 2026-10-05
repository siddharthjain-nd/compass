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
| **E1** | **Fast/slow tracking** (§4.1) | Stride-6 EdgeTAM + 30 fps flow matches stride-1 identity at about ⅕ of the compute | swaps on cups, ms/frame | nothing new |
| **E2** | **Gated memory** (§4.2) | Removes drift onto the hand | holder frames on `taken_*` + real grabs | nothing new |
| **E3** | **DAM-style distractor memory on EdgeTAM** (port from DAM4SAM, Apache-2.0) | Fewer swaps at 15 fps without extra cost | swaps at stride 2 and 6 | DAM4SAM code |
| **E4** | **Identity verifier** (§4.3) | Local matching separates two similar wallets where global features don't | ROC-AUC | 2 similar wallets |
| **E5** | **Speed on Mac.** Core ML full video path (community port) vs MPS vs lower input resolution | ≥ 10 fps for 1 object on the M2 | ms/frame and accuracy drop | coremltools |
| **E6** | **Event VLM.** Qwen3-VL-4B (mlx-vlm), given the tracker state as text vs given raw frames only | Tracker state turns wrong guesses into right ones on bag and hand events | accuracy on E0 events, seconds per event | ~4 GB download |
| **E7** | **SAM 3 / 3.1 reference run** (cloud GPU or offline) | An upper bound to compare against: how far is EdgeTAM + our fixes from the strongest tracker? | same metrics | GPU access, gated checkpoints |
| **E8** | **Edge bake-off.** Same pipeline and clips on Jetson (TensorRT) and/or Qualcomm (AI Hub EdgeTAM) | Meets the fps target for 1–4 objects | fps, identity metrics vs Mac | hardware access |

**Order and why:**
1. **E0 first:** nothing can be claimed without real clips.
2. **E1 and E2 next:** they're the cheapest and attack the two failures we have actually measured.
3. **E3 and E4:** they need the E0 clips.
4. **E5 and E6** finish the Mac version.
5. **E7 and E8** inform the edge move.

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

- **If E1 already fixes the cups swaps,** fast/slow tracking becomes the core architecture, and we may not need DAM (E3) at all.
- **If E5 finds no path to ≥ 10 fps on the M2,** the Mac stays a development box, and we move to edge hardware earlier.
- **If a released efficient SAM 3 variant shows real edge numbers** (EfficientSAM3, or whatever follows SAM 3.1), re-evaluate the tracker choice. Its detector-based re-anchoring addresses D2 natively.

---

## 8. Sources

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
