# Compass: object permanence tracker plan

_Last updated: 2026-10-04_

## 1. Goal

A camera system that keeps track of **a specific personal object**, registered from reference photos, and can say where it is even after it goes out of sight:

> "Wallet was picked up by a hand and carried out of frame to the left."
> "Keys went into the right pocket."
> "Mug is behind the laptop."

It must do this **without being trained on concepts like "pocket", "drawer" or "bag"**.

### Requirements

| # | Requirement |
|---|---|
| R1 | **Any object:** keys, wallet, mug, remote, glasses. Whatever the user registers. |
| R2 | **Any occluder or holder:** hands, pockets, bags, books, cloths, drawers, boxes, other people. |
| R3 | **Lookalikes:** tell *my* object apart from similar objects (e.g. someone else's similar wallet). |
| R4 | **Honest uncertainty:** report `UNKNOWN` / `AMBIGUOUS` instead of inventing a location. |
| R5 | **Mac first** (Apple M2, 16 GB), then **edge devices** (Jetson, Pi + Hailo), then **multi-camera / distributed**. |

### Non-goals (for now)

- Training or fine-tuning custom models.
- A cloud-only design. The core must run locally.

---

## 2. Current status

### Done

- [x] Conda env `compass` (Python 3.12, PyTorch 2.14 on MPS, OpenCV, timm). See `environment.yml`.
- [x] SAM 2.1 in `third_party/sam2` (tiny and small checkpoints).
- [x] EdgeTAM in `third_party/EdgeTAM`. It is loaded via `sys.path`, not pip-installed, because both packages are named `sam2`.
  - **Local patch:** `sam2/modeling/perceiver.py:298` changed `.view` → `.reshape`. Without it, tracking more than one object at once crashes. See `git diff` in that folder.
- [x] `experiments/real_video_tracking.py`: tracks a ball and three identical red cups in a **real** video (EdgeTAM's `02_cups.mp4`, a shell game) at a chosen frame rate.
- [x] `scripts/check_env.py`: loads a model, then benchmarks image segmentation and video tracking with an occlusion test.
- [x] `experiments/synthetic_tracking.py`: six synthetic scenarios (occlusion + lookalike wallets) with known ground truth.
  - Renders a video per scenario: input | mask overlay | verdict per frame | mask-area and visibility-score charts.
  - Results page: `experiments/output/index.html`.
- [x] **E1** (fast/slow tracking) and **E2** (the "still looks like my object?" check): see `RESEARCH.md` §8.
  - `experiments/e1_fast_slow.py` → `experiments/output/e1/index.html`
  - `experiments/e2_gated_memory.py` → `experiments/output/e2/index.html`
- [x] **E2b** (follow the hand that took the object): 335/336 frames, 0 on the other wallet, 8/8 final answers on drawn scenes. See `RESEARCH.md` §8.3.
  - `experiments/e2b_follow_holder.py` → `experiments/output/e2b/index.html`

### Findings so far

**Speed on M2 (full resolution, video tracking):**

| Model | Stock | With MPS attention patch |
|---|---|---|
| SAM 2.1 tiny | 1391 ms/frame (0.7 fps) | 819 ms (1.2 fps) |
| **EdgeTAM** | 277 ms (3.6 fps) | **190 ms (5.3 fps)** ← default tracker |

- **The bottleneck is memory attention.** It took 900 of the 1200 ms per frame in SAM 2 tiny.
- **`F.scaled_dot_product_attention` is slow on MPS** for these shapes (4096 queries × ~28k keys): 240 ms per call, against 120 ms for a plain matmul + softmax. `check_env.py` contains `patch_attention_for_mps()`, which swaps it out.
- **fp16 autocast did not help**, because autocast casts the softmax back up to fp32.
- **EdgeTAM gives the same results as SAM 2** on the synthetic occlusion test (mask area 2516→820 vs 2562→822).

**Lookalike tests (EdgeTAM, synthetic, every frame processed):**

| Scenario | Result |
|---|---|
| Similar wallet B placed nearby | Stays on A ✅ |
| B placed touching A | Stays on A, no merge ✅ |
| Hand fully covers A while B is placed | Reports "lost" while covered, then picks up A again (not B) ✅ |
| A picked up and carried off, B stays | Follows A with the hand and never jumps to B ✅ ⚠️ |

Grading uses ground truth: "on A" means at least 70% of the mask lies inside A's true outline. Frame counts:

| Scenario | on A | mask on holder | lost |
|---|---|---|---|
| occlusion (key partly covered) | 30 | – | – |
| placed_nearby | 40 | – | – |
| placed_touching | 40 | – | – |
| hand_covers_a | 33 | – | 7 (while covered) |
| taken_visible | 15 | 22 | 8 |
| taken_hidden | 15 | 22 | 8 |

**No identity switch to the lookalike in any frame.**

**⚠️ Drift onto the holder.** When a hand grabs the object, the mask grows from about 4.6k px to 10–12.7k px: the tracker silently starts tracking *the hand* and never reports "lost". The fix is the mask sanity check in §4.2.

- The tracker's own visibility score stays above 0 the whole time the mask is on the hand. It drops only after the hand leaves the frame.
- **An area check alone is not enough.** As the hand leaves the frame, the mask shrinks back under the 1.6× threshold (around frames 31–36) while it is still on the hand. So the check needs the appearance comparison too.

**Caveat:** these are synthetic shapes, and each test processed every frame.

**Real video test: shell game with 3 identical cups + a ball** (`experiments/output/real_cups_*`):

| Frame rate fed to the tracker | Cups keep their own identity? | Ball |
|---|---|---|
| 30 fps (every frame) | ✅ All three stay consistent. The ball is revealed under CUP 1, the cup that covered it. | ✅ "not visible" while covered, found again at the end |
| 15 fps | ❌ At frame ~258 a fast, blurred move loses CUP 2. Its label jumps onto CUP 3's cup; the left cup is unlabelled from then on. | ✅ |
| 5 fps (≈ Mac live speed) | ❌ Several mix-ups, including a **silent swap**: each cup still has one label, but on the wrong cup by frame 150. | ✅ |

What this means:

- **Frame rate matters a lot for fast-moving lookalikes.** In this clip, every halving of the frame rate made identity worse.
- **Speed:** 4 objects at once ran at about 1 s/frame on the M2 (single object: about 0.19 s/frame). Real-time 30 fps tracking of several objects is not possible on the Mac with the current setup.
- **Silent swaps exist.** "Two labels on one object" is detectable, but a clean swap is not. This needs the identity layer (§4.5: appearance + fine-grained marks + motion plausibility), not the tracker alone.
- **This clip is deliberately hard** (identical objects, fast shuffling). Wallets and keys on a table usually move slower. Phase 1 real clips will tell us how typical scenes behave.

---

## 3. Architecture

```
               ┌──────────── Identity (DINOv2 + local features) ── ~1 Hz + on events
               │                        │
FrameSource ──► Tracker (EdgeTAM) ──► Mask check ──► World memory / ──► Events ──► Reasoner (VLM)
 (latest        every processed        (is it still     state machine      (queue →     only on events
  frame only)   frame                   my object?)          ▲              MQTT later)
                                                             │
                              Scene context (occluders, holders, zones) ── ~1 Hz + on events
```

| Module | Job | Mac implementation | Rate | Edge / distributed swap |
|---|---|---|---|---|
| `FrameSource` | Frames from webcam, iPhone, RTSP or file | OpenCV capture thread, keeps only the latest frame | 30 fps capture | GStreamer / CSI camera |
| `Tracker` | Mask and visibility score per object | EdgeTAM + MPS attention patch, streaming wrapper | Every processed frame (~5 fps) | EdgeTAM on TensorRT / Core ML |
| `Identity` | "Is this my object?" Find it, re-find it, detect lookalikes | DINOv2 similarity map; LightGlue on close-ups | ~1 Hz + on events | ONNX / TensorRT |
| `SceneContext` | Find occluders and holders, name them, named zones | SAM point prompt at the last-seen spot; YOLO-World / OWLv2 labels; MediaPipe hands/pose (optional add-on for people) | ~1 Hz + on events | TensorRT detector |
| `WorldMemory` | `object → holder → location`, confidence, history | Pure Python | On every update | Unchanged |
| `Events` | Typed messages, e.g. `{"obj": "wallet", "event": "carried_away", "holder": "hand", "dir": "left"}` | In-process queue | — | MQTT / ZeroMQ |
| `Reasoner` | Describe or confirm ambiguous events, name unusual holders | Cloud VLM, or a small local VLM via MLX | Only on events | Hub / cloud |

### Design rules

1. **Each model runs at its own rate.** Tracker every frame, identity and context about once a second, VLM only on events.
2. **Every model sits behind an interface.** Moving to edge hardware means swapping implementations, not rewriting the logic.
3. **The core logic (state machine, world memory) is pure Python** with no ML imports.
4. **No Mac-only APIs in the core.** Core ML is allowed only behind the `Tracker` interface.
5. **Record real clips early.** They become the regression suite for every model swap and device move.

---

## 4. Core logic

### 4.1 Object states

| State | Meaning |
|---|---|
| `VISIBLE` | Tracker mask passes the sanity check |
| `OCCLUDED_IN_PLACE` | Something is covering it, and the object is presumably still there |
| `HELD` | Moving with a holder (hand, bag, box...); the holder is tracked as a stand-in |
| `OUT_OF_FRAME` | Left the camera view, with a direction |
| `UNKNOWN` | Disappeared with no explanation (blur, lighting, tracker failure) |

Separate from these, there is an **identity flag**: `CONFIRMED` / `AMBIGUOUS(p)`, used when a lookalike makes the identity uncertain.

### 4.2 Mask sanity check (fixes "drift onto holder")

On every frame, compare the tracker's mask with what the object should look like:

- **Area** compared with a running median of recent areas (e.g. more than 1.6× or less than 0.4× fails).
- **Shape and aspect** change.
- **Appearance:** DINOv2 similarity between the masked crop and the reference photos.

If the check fails, the mask is probably covering a holder or occluder, not the object. Switch to `HELD` / `OCCLUDED_IN_PLACE` and record whatever the mask now covers as the holder.

### 4.3 Disappearance analysis (works for any occluder)

Triggered when the visibility score drops or the mask check fails:

1. **Find the occluder:** segment whatever is now at the last-seen spot, using a point prompt there.
2. **Classify the disappearance:**
   - *Covered in place:* the occluder arrived and stays. → `OCCLUDED_IN_PLACE`
   - *Carried away:* the object was overlapping something that moved off, and the object did not reappear. → `HELD`
   - *Left the frame:* last seen at the image edge, moving outward. → `OUT_OF_FRAME`
   - *Unexplained:* no occluder and not at the edge. → `UNKNOWN` (do not guess)
3. **Track the holder** as a stand-in for the hidden object.
4. **Name the holder** with detector labels or CLIP zero-shot labels. Ask the VLM only if those fail.
5. **Resolve:** if the object reappears, verify identity and set `VISIBLE`. If the holder leaves the frame, the object is `OUT_OF_FRAME` via that holder.

The people add-on (MediaPipe hands + pose) refines a human holder into "left/right hand" and "left/right pocket" (pocket = the zone around the hip landmark). People are just one kind of holder.

### 4.4 World memory

- **Holders can nest:** keys → bag → carried out of frame to the left.
- **Each entry stores:** state, holder chain, location / zone, last-seen time and frame, confidence, and evidence (key frame IDs and a clip snippet).

### 4.5 Identity and lookalikes

Identity signals, strongest first:

1. **Continuity.** The tracker's memory of where the object was and how it moved. This held in every synthetic test.
2. **Multi-track.** Every strong match for the reference gets its **own track**. Lookalikes are labelled, e.g. "lookalike #1, arrived with a hand at 14:02".
3. **Arrival context.** The object that was already there is mine; the newcomer is not.
4. **Fine-grained marks.** Close-ups taken at registration (scratches, logo, wear, stitching, a card sticking out) are compared using local feature matching (LightGlue / LoFTR). This is used when continuity breaks.
5. **Uncertainty.** If it is still ambiguous, report e.g. `"one of two similar wallets was taken; 70% yours"` and ask the user.

When several tracks reappear after a shared occlusion, assign identities with the Hungarian algorithm. The cost combines appearance, fine-grained match and how plausible the motion is.

---

## 5. Roadmap

> **Research update (2026-10-05):** see `RESEARCH.md`. It adds four changes and runs them as experiments E0–E8:
> - fast/slow tracking (identity at 30 fps, EdgeTAM as a 5–10 fps corrector);
> - gated memory writes;
> - a global + local identity verifier;
> - an explicit world-state graph that the VLM only phrases.
>
> The phase items below are tagged with their experiment number.

### Phase 1: Live tracker on the Mac

> **Status (2026-10-06): first live version built.** Run it with `python -m compass.live`.
> - The tracker, the B check and "follow the hand" now live in the `compass/` package, shared with the experiments.
> - **Verified:**
>   - the streaming tracker gives masks identical to the offline tracker (120/120) and holds flat memory over 900 frames (2.0 GB);
>   - E2b reproduced on the shared code: 334/336 frames, 8/8 final answers;
>   - replaying real videos as a live camera: ~4.2 fps processed, 0.26 s lag;
>   - mug held in a hand: stays "visible" and never highlights the hand;
>   - cups shuffle at live speed: right cup only 41% of the time, as E1 predicted for about 5 fps.
> - **Not verified yet:** the real Mac camera. macOS has to grant camera access to your terminal on the first run.
>
> **Fixes after the first live test (2026-10-06).** Reported: the mask disappears when the wallet is turned while the screen still says "visible", and the system keeps "following the hand" after the wallet is put down. Reproduced on a real replay (EdgeTAM's blocks video) with per-frame logs.
> 1. **"Visible" with no mask was a bug:** "object-sized but doesn't look like it" had no state. Added **unsure** (grey outline).
> 2. **Turning the object:** a 30–45° turn drops similarity to 0.79–0.89, right at the 0.85 cutoff.
>    - Fix: the check now compares against the registration view plus 10 rotated/mirrored copies, and learns new angles while tracking.
>    - Turned block: 0.85–0.89 → 0.97–0.99.
> 3. **Calibration bug:** "typical similarity" was learned from the first live frames, which could already show the object half in a hand (0.15). Thresholds are now absolute against the registration.
> 4. **Hand-off delay:**
>    - The search ran every 3rd held frame (~1.2 s apart); now every frame (~0.45 s).
>    - It also compared against the un-turned view only (fixed by 2).
> 5. **Lost objects were never searched for.** At live speed the tracker often *loses* a grabbed object ("covered") instead of sliding onto the hand.
>    - Added whole-picture re-detection while lost: a wallet put down elsewhere is found 0.2 s later.
>    - With lookalikes around, a re-found object is shown only as "maybe here": on the blocks table the search picked a different blue block (similarity 0.88).
> 6. Session logs are now saved automatically to `clips/sessions/`.
>
> Regression after all fixes: drawn scenes unchanged (334/336, 8/8), mug unchanged. **Next:** a real hand detector (covered vs in a hand at live speed, plus hand-path evidence to resolve "maybe here"), then speed (E5).

- [x] Frame sources: Mac camera and a video file replayed as if live (`compass/sources.py`). No iPhone available.
- [x] Streaming EdgeTAM wrapper (`compass/tracker.py`):
  - add an object by box;
  - output a mask and visibility score per frame;
  - bounded memory.
- [x] Live overlay viewer with fps, lag and the current state; `c` records raw clips to `clips/` (`compass/live.py`).
- [ ] Multiple objects at once (v1 tracks one).
- [x] Mask sanity check (§4.2) on the output, E2's variant B (`compass/appearance.py`), plus "follow the hand", E2b (`compass/holder.py`). Both are in the live app.
- [ ] Clip recorder, for building the test suite (§6). **(E0)**
- [x] ~~**Fast/slow tracking** with optical flow~~ **(E1): rejected.** Flow loses cups in fast, blurred swaps. Keep EdgeTAM at ≥ 15 fps instead, which makes speed (E5) the priority.
- [ ] **Gated memory (E2 variant C):** postponed. It makes the tracker re-find objects, but with a lookalike present it can re-find the wrong one. Revisit when the identity check (E4) can confirm re-finds.
- [ ] **Speed (E5), moved up:** Core ML / Neural Engine for the video path, lower input resolution, fewer memory frames. Target ≥ 15 fps for 1 object.

**Exit criteria:**
- Click an object and the mask follows it live at ≥ 4 fps.
- When a hand grabs it, the object is flagged as held/covered instead of the tracker silently following the hand.

### Phase 2: Registration and identity

- [ ] Registration CLI:
  - 3–5 photos of the object;
  - close-ups of distinguishing marks;
  - an optional name.
- [ ] DINOv2 similarity map that finds the object and starts the tracker automatically (no clicking).
- [ ] Re-finding the object when it reappears.
- [ ] Lookalike detection that spawns a separate track (§4.5).
- [ ] Fine-grained verifier: ALIKED + LightGlue on the close-ups (Apache-2.0; avoid non-commercial SuperPoint, MASt3R and CoTracker). Compare DINOv2 vs DINOv3 for the shortlist. **(E4)**
- [ ] Register known lookalikes as hard negatives.

**Exit criteria:**
- A registered object is found without clicking.
- A lookalike gets its own track.
- The correct identity is assigned after reappearance in the lookalike test clips.

### Phase 3: Occluders and holders

- [ ] Occluder segmentation at the last-seen spot.
- [ ] Disappearance classifier (the four types in §4.3).
- [ ] Holder tracking and naming. **Prototype done on drawn scenes (E2b):** held / out-of-view states and the hand-off rule. Next: a real hand detector, and test on E0 clips.
- [ ] Optional people add-on: MediaPipe hands and pose, giving hand and pocket zones.

**Exit criteria:** the correct disappearance type and holder on the occluder test clips.

### Phase 4: World memory and events

- [ ] State machine and nested world memory (§4.1, §4.4).
- [ ] Event log as JSON lines, with key frames and clip snippets.
- [ ] Query CLI: `where wallet` returns the state, holder chain, last-seen time and an evidence frame.

### Phase 5: Reasoner (VLM)

- [ ] On events only: send key frames **plus the world-state graph as text**, get a description or confirmation. Never ask the VLM to infer hidden state, since VLMs are at chance on the shell game (VET-Bench).
- [ ] Use it only for naming unusual holders and resolving ambiguous cases.
- [ ] Mac candidate: Qwen3-VL-4B 4-bit via mlx-vlm; compare "frames only" vs "frames + tracker state". **(E6)**

### Phase 6: Evaluation and speed

- [ ] Scenario test suite running on the recorded clips, reporting the metrics in §6.
- [ ] Core ML for the Neural Engine. Note: EdgeTAM's own export covers the **image path only**, so the video memory path needs the community port or our own split. **(E5)**
- [ ] Tune input resolution and the number of memory frames.
- [ ] Port DAM4SAM's distractor-aware memory to EdgeTAM (identity switches **did** show up on the real cups clip). **(E3)**
- [ ] SAM 3 / 3.1 reference run on a cloud GPU, as an upper bound. **(E7)**
- [ ] Compare against **TCOW** on the container / occluder clips.

### Phase 7: Edge and distributed (later)

- [ ] **Edge bake-off (E8):** the same pipeline and clips on Jetson (TensorRT) and/or Qualcomm (AI Hub ships EdgeTAM, ~17 ms/frame int8, unverified). Pick hardware from the measurements.
- [ ] **Hailo / Rockchip:** detection only; their compilers can't run the tracker's transformer.
- [ ] Cloud only for event clips and long-history questions, not continuous video.
- [ ] **Distributed:**
  - camera nodes do capture, motion gating and short-term tracking;
  - the hub does identity, world memory and the VLM;
  - events travel over MQTT.
- [ ] **Cross-camera handoff:** if the object leaves camera 1 on the left, expect it on the camera that covers that side.

---

## 6. Test scenarios and metrics

Record each scenario as a real clip, with ground-truth events in a small JSON file next to it.

| Category | Scenario | Expected output |
|---|---|---|
| Occluders | In hand → into pocket | `HELD` by hand → `OCCLUDED`, holder = right pocket |
| | Covered by paper or a cloth | `OCCLUDED_IN_PLACE`, holder = cloth |
| | Book or cup slides in front | `OCCLUDED_IN_PLACE` → `VISIBLE` when moved |
| | Put in a drawer or box that closes | `OCCLUDED_IN_PLACE`, holder = drawer |
| | Put in a bag that is carried away | `HELD` by bag → `OUT_OF_FRAME` (direction) |
| | A person walks in front | Brief `OCCLUDED_IN_PLACE` → `VISIBLE` |
| | Leaves the frame | `OUT_OF_FRAME` (direction) |
| Lookalikes | Similar object placed nearby | Mine stays `CONFIRMED`; lookalike gets its own track |
| | Placed touching mine | No merge, no switch |
| | Hand covers mine while placing theirs | Correct identity after uncover |
| | **Mine taken, theirs stays** | Mine `HELD` → `OUT_OF_FRAME`; theirs not mistaken for mine |
| | Theirs taken, mine stays | Mine stays `VISIBLE`, `CONFIRMED` |
| | Both covered, then one taken | Fine-grained check; otherwise `AMBIGUOUS(p)` |
| | Swapped while hidden (shell game) | `AMBIGUOUS` unless distinguishing marks resolve it |
| Negatives | Motion blur, lighting change, camera bump | `UNKNOWN`, never an invented location |
| | Object not in the scene at all | No false detection |

### Metrics

- **Final-location accuracy:** state and holder correct at the end of the clip.
- **Event accuracy:** correct event type, holder and time (± tolerance).
- **Identity accuracy** in the lookalike clips.
- **False-visible rate:** frames where the holder was tracked as if it were the object.
- **Latency** from event to log entry.
- **Throughput** in fps.

---

## 7. Ideas backlog: things that can be added

Priority: **H** = soon after the core works, **M** = valuable later, **L** = nice to have.

### Using it

| Idea | What it adds | Priority |
|---|---|---|
| Natural-language query | "Where did I leave my wallet?" answered with text plus a snapshot or short clip | H |
| Alerts | "Your wallet was picked up by someone else", "You're leaving without your keys" (door zone) | H |
| Named zones | User marks "kitchen counter", "entry shelf" so answers use human names | H |
| Learning from corrections | When the user confirms or corrects an answer, store that view as an extra reference | M |
| Show-and-rotate registration | Hold the object up and rotate it; views are captured automatically | M |
| Timeline view | History of an object with thumbnails and an event list | M |

### Perception

| Idea | What it adds | Priority |
|---|---|---|
| Distractor-aware trackers (DAM4SAM, SAMURAI) | Fewer identity switches with lookalikes | H (if needed) |
| Multi-camera handoff and room map | Follow an object between rooms | M |
| Depth (iPhone LiDAR / stereo) | Tell "behind" apart from "inside", and separate stacked objects | M |
| Re-registration when appearance changes | Handles a fuller wallet or a new keychain without breaking identity | M |
| Linking events to a person | "Taken by the person who arrived at 14:02". Opt-in, privacy-sensitive | L |
| Audio cues | Zip, drawer close or jingle as weak extra evidence | L |

### Efficiency

| Idea | What it adds | Priority |
|---|---|---|
| Motion gating | Skip heavy models while the scene is static; saves power on edge devices | H |
| Core ML / Neural Engine | Faster tracking on Mac and iPhone | M |
| Confidence calibration | So "70% yours" really means 70% | M |

### Integration

| Idea | What it adds | Priority |
|---|---|---|
| Phone notifications | Alerts reach you away from the computer | M |
| Home Assistant / HomeKit | Automations, e.g. a light blinks if you leave without your keys | L |
| Small web dashboard | Live view, world memory, event history | M |
| iPhone app | Phone as the camera and running on-device | L |

### Privacy and safety (build in from the start)

- **On-device by default.** VLM calls send only crops or key frames, never continuous video.
- **Configurable retention** for stored clips, with optional face blurring.
- **Owner-only tracking.** Only objects registered by the owner are tracked.

---

## 8. Risks and open questions

### Risks

| Risk | Mitigation |
|---|---|
| Low fps loses fast motion and swaps lookalikes (seen on the real cups clip: fine at 30 fps, broken at 5–15 fps) | Raise the fps: Core ML / Neural Engine, lower input resolution, only track registered objects, motion gating; Jetson + TensorRT later. Motion-aware tracker (SAMURAI). Identity layer to detect and fix swaps. |
| Tracker drifting onto the holder (observed) | Mask sanity check (§4.2) |
| Synthetic results ≠ real world | Real clip suite (§6) before relying on any number |
| Identical objects swapped while hidden | Close-ups of distinguishing marks; otherwise report `AMBIGUOUS` |
| 16 GB RAM limits local VLMs | Small MLX VLM or a cloud API, only on events |
| Camera placement (the MacBook camera can't see pockets or the table) | iPhone via Continuity Camera, or a USB webcam aimed at the torso / table |
| Privacy of people on camera | §7 privacy defaults |

### Open questions

- Which camera setup do we test with first: desk webcam, iPhone, or a room-view camera?
- Which objects do we register first? Wallet and keys are the suggestion.
- Are cloud VLM calls acceptable, or must everything stay local?
- What is the target edge hardware: Jetson Orin Nano/NX, or Pi 5 + Hailo?

---

## 9. Repo layout and how to run

```
compass/
├── PLAN.md                 ← this file
├── RESEARCH.md             ← literature review + experiment plan (E0–E8)
├── environment.yml         ← conda env spec
├── compass/                ← the package: models, tracker (streaming), appearance (B check),
│                              holder (follow the hand), sources (camera/video), live (app), video_io
├── clips/                  ← clips recorded with the live app (git-ignored)
├── scripts/
│   ├── check_env.py        ← model load + speed/occlusion check
│   └── setup_third_party.sh← re-creates third_party/ (pinned commits + patches + checkpoints)
├── patches/                ← our fixes to third-party code (EdgeTAM multi-object reshape)
├── experiments/            ← synthetic_tracking.py, real_video_tracking.py
│   └── output/             ← generated videos + index.html (git-ignored; re-run the scripts)
├── third_party/            ← git-ignored; created by setup_third_party.sh
│   ├── sam2/               ← SAM 2.1 (pip -e installed, checkpoints/ tiny+small)
│   └── EdgeTAM/            ← EdgeTAM (loaded via sys.path, checkpoints/edgetam.pt)
└── (planned)
    └── tests/
```

```bash
conda env create -f environment.yml && conda activate compass
bash scripts/setup_third_party.sh       # clones SAM 2 + EdgeTAM at pinned commits, applies patches/, downloads checkpoints
pip install -e .                        # installs the compass package (python -m compass.live from any folder)

python -m compass.live                 # LIVE: Mac camera; SPACE, draw a box around the object, ENTER; c = record clip
python -m compass.live --video third_party/EdgeTAM/examples/04_coffee.mp4   # replay a video as if live

python scripts/check_env.py            # EdgeTAM (default) with the MPS attention patch
python scripts/check_env.py tiny       # SAM 2.1 tiny
python experiments/synthetic_tracking.py   # 6 scenarios → experiments/output/*.mp4, index.html
python experiments/real_video_tracking.py --stride 1   # real cups clip at 30 fps (--stride 2 = 15 fps, 6 = 5 fps)
open experiments/output/index.html
```

---

## 10. References

- **SAM 2:** Segment Anything in Images and Videos (Meta). github.com/facebookresearch/sam2
- **EdgeTAM:** On-device Track Anything Model (Meta, CVPR 2025). github.com/facebookresearch/EdgeTAM
- **DINOv2:** self-supervised visual features, used for reference matching.
- **LightGlue / LoFTR:** local feature matching, used for fine-grained identity.
- **YOLO-World / OWLv2 / Grounding DINO:** open-vocabulary detection.
- **MediaPipe Hands / Pose:** the people add-on.
- **DAM4SAM:** A Distractor-Aware Memory for Visual Object Tracking with SAM2 (CVPR 2025).
- **SAMURAI:** SAM 2 with motion-aware memory for zero-shot tracking.
- **TCOW:** Tracking through Containers and Occluders in the Wild (CVPR 2023).
