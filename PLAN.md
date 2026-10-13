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

### Phase 1b: Explain why the object looks different (E9) + hand detector — built 2026-10-07

> **Status:** built and tested on drawn scenes and real replays. **Not yet tested on the real Mac camera.**
> - **Hands** (`compass/hands.py`): MediaPipe hand landmarker, ~24 ms per frame on the CPU.
>   - Hands found in 100% / 95% / 75% of frames in the mug / cups / blocks videos; the 75% is correct, because the hands do leave that picture.
>   - Left/right labels checked on one video (the watch hand is labelled left).
> - **Explainer** (`compass/explain.py`): the expected outline from matching details (ORB + a RANSAC similarity fit gives the angle and scale), plus the content of the missing part and whether the area just outside it changed.
>   - Controlled edits of a real block image: **14/14** (turned 20/40/90°, measured 20.3/39.4/90.3°; ×1.5 and ×0.7, measured ×1.49/×0.70; half-blocked by something / by a hand; flipped; tracker jumped; covered; removed; still there).
> - **Drawn scenes (E9): 263/264 frames correct, 0 critical mistakes.** `experiments/output/e9/index.html`.
>   - **Caution:** the explainer was fixed several times against these same scenes; the regressions and real replays below are the independent check.
> - **Regressions:** E2b drawn scenes 335/336, 8/8 final answers; mug replay all "visible".
> - **Real replays:**
>   - **blocks:** "taken by your right hand", then "put down at (806, 331)", the correct spot. An earlier version picked a wrong block, which is why the search now prefers the object's last spot and compares all candidates.
>   - **cups:** "partly hidden by your hand (71% visible)" while shuffling. Identical cups moving fast are **still mostly mis-tracked** at live speed (right cup 11/31): a speed and identity problem (E5, E4), not an explanation problem.
> - **Speed:**
>   - start-up ~0.4 s (was 1.5–2 s; lookalikes are now remembered as spots, not SAM outlines);
>   - ~250 ms per frame while visible, ~3–3.6 fps in replays (the hand detector + explainer cost ~0.5 fps).

**Fixes from the first live sessions with hands (2026-10-08).** Session logs showed "not sure" in 17–65% of frames while the wallet was held.
- **Cause:** a partly visible wallet between the fingers (6–35% of its size, look 0.3–0.7) never entered "in your hand". That only happened when the mask spread onto the hand or vanished under it.
- **Fixes:**
  - **Wallet touching a detected hand** with an unexplained look, or under 30% visible → **"in your hand"**. The visible part is shown in blue when its look is ≥ 0.45.
  - **Lookalike spots** count only if a box crop there looks like the object (≥ 0.75). Unchecked spots made front-camera re-finds "maybe".
  - **Hand-off:**
    - reject candidates that were already lying there when the hand took the object;
    - decide by **matching small details** (ORB inliers against the last good view): a first version of E4;
    - in a scene with many lookalikes, a look-only match is just "maybe".
- **Blocks stress test** (30+ similar blocks, 3 runs): pickups caught; every confident "put down" was at the correct spot (decided by 56–136 matching details vs ≤ 4 for wrong blocks); wrong spots appeared only as "maybe".
- **Regressions:** E2b 334/336 (8/8 final answers), E9 263/264 (0 critical), mug and cups unchanged.
- **Real camera speed** (from logs): 375–430 ms/frame (~2.3–2.7 fps), slower than replays, so E5 matters.

**Real-camera test (E0) + pocket (2026-10-09).** `experiments/e0_real_clips.py` → `experiments/output/e0/index.html`.
- **Six clips recorded with the live app** (MacBook camera, 15 fps, 4.3 min): hold/turn, put down/pick up, paper cover, closer/farther (with the camera tilted), pocket, two similar wallets.
- **Replay:** every 5th frame (3 fps ≈ live speed), the same every time.
- **Labels:** written from per-second contact sheets.

| Clip | Baseline | After fixes |
|---|---|---|
| hold & turn | 86% | 95% |
| put down / pick up | 84% | 100% |
| paper cover | 71% | 98% |
| closer / farther (+ camera tilt) | 98% | 99% |
| pocket | 60% (no pocket yet) | **100%** ("taken by your right hand" → "put into your right pocket") |
| two similar wallets | 100% | 100% (never on the other wallet) |
| **All** | **86%** | **99% (615/624)** |

**Fixes (each from the clips' decision logs):**
- A hand touching **≥ 15%** of the visible part counts: hands holding from below touched only 13–30%.
- **< 25% visible + something new in front → "covered"** (not when the object still shows in place, nor at the picture edge).
- **Strong change around + decent look → "partly hidden"** (a paper's shadow broke colour matching).
- **Tracker bug:** pruned restart frames stayed in EdgeTAM's index, and after many restarts it crashed. This could also have hit long live sessions.

**Pocket** (`compass/body.py`, MediaPipe selfie segmenter for clothes):
- **Body pose was tried and rejected:** it needs a face, and with the head out of view it put "hips" on the table.
- **Rule:** the object vanishes while the holding hand is at the clothes (2 frames); then the hand moves away or leaves the picture, or stays ~2.5 s → **"in your right pocket"** (side = the hand that put it there).
- **Stays "in your pocket"** until the object clearly passes the look check again.
- **Cost:** ~120 ms, only while the object is hidden in a hand.

**Cautions:**
- These rules were tuned while looking at these same clips. The independent checks still pass (drawn scenes 334/336 and 263/264; blocks and mug replays unchanged; no false pockets).
- **One label was changed after the first run:** pocket clip frames 246–305, "pocket" → "going into the pocket". The hand was still at the pocket there.
- **More clips** (other rooms, lighting, objects) are needed before calling it general.

**Review list (to check together on the real camera):**
1. **Left/right hand labels** on the Mac camera: they were checked on one video only.
2. **"In your hand, partly visible"** uses a looser look threshold (0.6) when a detected hand holds the object: does it show the right thing?
3. **The explainer's thresholds** were set on drawn scenes:
   - surrounding change ≥ 15% → "partly hidden";
   - colour distances 15–18;
   - turned ≥ 12°;
   - closer ≥ 1.25×.
4. **The mirrored view:** the box drawn on the mirrored picture is converted back, and left/right messages are swapped. Check the exit messages on the real camera.
5. **Speed on the real camera** with everything on.


**Why:** "not sure" used to cover several different situations: something is in front of the object, the object was turned, the object moved closer to or farther from the camera, or the tracker jumped to something else. The user can't tell which one is happening, and the right reaction differs for each.

**Target states** (they replace the single "not sure"):

| Situation | Shown as | Message |
|---|---|---|
| **Blocked:** something is in front of part of it | blue on the visible part | "partly hidden behind something (60% visible)" |
| **Turned:** whole object visible, new angle or other side | blue (the new angle is learned) | "visible, turned ~40°" / "visible, other side up" |
| **Closer / farther:** same outline, just bigger or smaller (front camera) | blue | "visible, moved closer" |
| **Something else:** the tracker jumped | grey | "not sure, this may be something else" |

**Evidence it's feasible.** In the blocks replay, "in the hand" and "turned" already separate by size alone:

| Moment | Size vs normal | Look similarity |
|---|---|---|
| Block in the hand (blocked) | 5–7% | 0.17–0.23 |
| Block put down turned | 97–100% | 0.71–0.81 |

One signal isn't enough in general, so the decision combines several.

**How:**
1. **Expected outline:** the last good outline, moved by the motion since then (optical flow).
2. **Signals:**

| # | Signal | Blocked | Turned | Closer/farther | Something else |
|---|---|---|---|---|---|
| 1 | Share of the expected outline still covered by the mask | low | high | high after rescaling | low, the mask is elsewhere |
| 2 | What fills the missing part | **something new** (neither the object's old look nor nearby background) | background / table | — | — |
| 3 | Do the visible pieces match the same pieces from before? (DINOv2 patch tokens, reused from the existing pass) | **yes** | not piece by piece | yes after rescaling | no |
| 4 | Does a rotation explain it? (ORB/SIFT detail matching) | — | **yes**, gives the angle | — | no |
| 5 | Motion inside the outline (optical flow) | something slides in from outside | **swirls around the centre** | expands / contracts | jump, no continuity |

3. **Decision:**
   - **Blocked:** (1) low + (2) new + (3) yes.
   - **Turned:** (1) high and either (4) a rotation, or (5) swirling with no new content.
   - **Closer/farther:** (1) high after rescaling + (5) expanding or contracting. This also stops the size rule from calling "moved closer" "taken by a hand".
   - **Something else:** none of the above.
4. **Turned and closer/farther stay blue**, and the new look is added to the look bank.
5. **Cost:** about 15–20 ms per frame on top of ~240 ms (flow ~5, outline comparison ~2, patch tokens ~0–5, ORB ~5–10).

**Front-camera items** (MacBook camera facing the user):
- [x] **Left/right from the user's point of view.** The picture isn't mirrored, so "out of view to the left" currently means the *camera's* left. Add a mirrored display (`--mirror`, default on for the built-in camera) and phrase sides from the user's view.
- [x] **"Closer/farther"** (above): moving the object toward the camera must not trigger the "grown → taken by a hand" rule. The scale has to agree with the area change.
- [x] **Hand detector** (MediaPipe): "in your right hand", following that hand, and "partly hidden by your hand".

**Tasks:**
- [x] Drawn test scenes with known answers:
  - wallet slowly covered by a book from the side;
  - wallet turned in place 0→90°;
  - wallet flipped over;
  - a hand partly over it;
  - wallet moved toward / away from the camera (scaling);
  - the tracker deliberately switched to the other wallet.
- [x] `compass/explain.py`: the expected outline + the 5 signals + the decision. Plug it into `ObjectState` (replacing the single "unsure"), the live display and the session log.
- [x] Experiment `experiments/e9_explain.py`, with a results page, as before.
- [x] Real replays:
  - blocks video: the block in the hand should be "blocked", and the block put down should be "turned";
  - mug video: lifted toward the camera should be "closer/farther" or "visible".
- [ ] The user's own clips (press `c`): wallet half under paper vs turned vs moved toward the camera, filmed with the front camera.

**Exit criteria:**
- Right label on ≥ 90% of frames in the drawn scenes.
- **Never "turned" when something is blocking.**
- No regressions on E2b (334/336, 8/8) or the mug replay.
- Live speed stays ≥ 3.5 fps.

**Known limits:**
- **Turned onto its edge** (thin side visible) looks a lot like blocked.
- **A side never seen before** gives no detail match, so it relies on motion and continuity.
- **A blocker the same colour as the table** is hard to spot as "something new".

**Done in the same step:** the hand detector (MediaPipe Hands) names the blocker when it's a hand ("partly hidden by your hand", "in your hand") and gives the hand's path for the lookalike cases.

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
- [x] **Speed research for many objects** (2026-10-09): `RESEARCH.md` §9, timing probe `experiments/e5_speed_probe.py`.
  - Per object per frame ~120 ms, of which memory attention is 64 ms, almost all spent on the *whole-frame* query grid.
  - Plan: per-object work only for **active** objects and only **around** the object; shared work once per frame.
- [x] **One object faster, round 1** (2026-10-11, `RESEARCH.md` §9.8):
  - visible 404 → ~220 ms (2.5 → ~4.5 fps); in hand 813 → ~465 ms (1.2 → ~2.1 fps); live replay 3.9 fps;
  - E0 616/624.
  - **Changes:**
    - faster picture prep;
    - hands in parallel;
    - search reuses the tracker's features;
    - clothes reused 1 s;
    - batched checks;
    - skip while nothing changes;
    - window around the object (`compass/window.py`).
  - **Live app flags:** `--no-window`, `--no-skip` to compare.
- [x] **Round 2** (2026-10-12, `PROGRESS.md`):
  - hidden states watched in a window (covered / held / pocket), the whole picture only when lost;
  - two pocket-rule fixes found by a new clip;
  - search pre-check (tried; off by default: no net gain on a quiet Mac);
  - explainer on a crop (same answers, 3.3× faster);
  - 8 clips 794/807;
  - quiet Mac: covered 475 → 169 ms, pocket 188 → 133 ms, visible and in hand unchanged.
- [x] **Round 3** (2026-10-13, `PROGRESS.md`):
  - Core ML image encoder (`compass/coreml_encoder.py`, default in the live app): 93 → 27 ms;
  - windowed search outlines (no gain for the big wallet);
  - visible 191 → 131–161 ms, covered → 94–122, pocket → 93–109; live replay 5.5–6.6 fps;
  - 8 clips 796/807;
  - 3c torch.compile: dropped (slower on MPS);
  - 3d 768 px: +28% speed, −14 frames → available as `--size 768`, default stays 1024.
- [ ] **Round 4:** in-hand search only when something new appears along the hand's path (or a fast detector); torch.compile try; 768 px input if still needed.
- [x] **E5a** window around the object: memory step 62 → 22 ms for the (large) wallet; masks match the whole-frame tracker (mean IoU 0.989).
- [ ] **E5b** multiple objects, "watch cheaply, track on wake": a resting object costs only a pixel-change check in its box; the accurate tracker runs (in a window) only for objects that moved or a hand neared, until they rest again (`RESEARCH.md` §9.8).
- [ ] **E5c** encoder input size 1024 / 768 / 512. **E5d** helpers (hands, appearance check) at their own pace. **E5e** window the memory encoder + decoder too.
- [ ] Core ML for the Neural Engine: lower priority. Community ports of the memory path run on CPU + GPU, not the Neural Engine (◐).
- [ ] Port DAM4SAM's distractor-aware memory to EdgeTAM (identity switches **did** show up on the real cups clip). **(E3)**
- [ ] SAM 3 / 3.1 reference run on a cloud GPU, as an upper bound. **(E7)**
- [ ] Compare against **TCOW** on the container / occluder clips.

### Phase 7: Edge and distributed (later)

- [ ] **Edge bake-off (E8), hardware not decided yet** (user decides later from what works on the Mac). Known: Jetson + TensorRT is the best-proven fast route (fixed-shape engines suit the windowed memory step). Qualcomm's AI Hub EdgeTAM runs memory attention on the CPU (checked).
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
| Motion gating | Skip heavy models while the scene is static; saves power on edge devices. Now planned per object as E5b ("resting objects sleep") | H |
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
├── RESEARCH.md             ← literature review + experiment plan (E0–E9), speed research (§9)
├── PROGRESS.md             ← step-by-step log: what we tried, numbers, decisions
├── progress/data/          ← small per-step result summaries (tracked; experiments/output/ is not)
├── environment.yml         ← conda env spec
├── compass/                ← the package: models, tracker (streaming), appearance (B check), explain (E9),
│                              hands (MediaPipe), body (clothes), holder (states + follow the hand), window (track
│                              only around the object), sources, live (app), video_io
├── clips/                  ← clips recorded with the live app (git-ignored)
├── scripts/
│   ├── check_env.py        ← model load + speed/occlusion check
│   └── setup_third_party.sh← re-creates third_party/ (pinned commits + patches + checkpoints)
├── patches/                ← our fixes to third-party code (EdgeTAM multi-object reshape)
├── experiments/            ← e0_real_clips.py (real-clip score), e5_time_breakdown.py (where time goes),
│                              e5a_window.py, e5_speed_probe.py, progress_snapshot.py, e1/e2/e2b/e9 scripts, ...
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
