# Compass: progress log

One entry per step: what we tried, why, the numbers, and what we decided. Newest at the bottom.
- **The plan** is in `PLAN.md`; the **research and its reasoning** are in `RESEARCH.md`.
- **Data:** each entry links to a small summary file in `progress/data/`, which is kept in git. The full per-frame outputs live in `experiments/output/`, which is *not* in git.
- **How to add an entry:** run the test, then `python experiments/progress_snapshot.py <step-name> --e0 ... --timing ... --note "..."`, then add a section here.

**Words used below:**
- **fps:** pictures processed per second, = 1000 ÷ ms per picture.
- **E0 score:** frames answered correctly on the user's 6 real clips (replayed at 3 pictures per second).
- **"Other wallet" frames:** frames where the system highlights the wrong (lookalike) wallet. This must always be 0.

---

## Where we are now (2026-10-13, after speed round 3)

One registered object, MacBook M2. Tracker image step on the Neural Engine (Core ML), the rest on the GPU (PyTorch). Timings are from quiet-Mac runs (round 3: the quietest paired run).

| | Start (2026-10-09) | Round 1 | Round 2 | **Round 3 (default, 1024 px)** | Round 3, 768 px option |
|---|---|---|---|---|---|
| Wallet visible | 404 ms (2.5 fps) | 217 ms | 191–211 ms | **131–161 ms (6–7.6 fps)** | 108 ms (9.3 fps) |
| Wallet in hand | 813 ms (1.2 fps) | 452–460 ms | 452 ms | **352–445 ms (2.2–2.8 fps)** | 289 ms (3.5 fps) |
| Covered | 462 ms | 475 ms | 145–169 ms | **94–122 ms (8–10.6 fps)** | 77 ms (13 fps) |
| In pocket | 380 ms | 188 ms | 133 ms | **93–109 ms (9–10.8 fps)** | 74 ms (13.6 fps) |
| Live replay, clip 2 | ~2.5 fps | 3.9 fps | 4.4 fps | **5.5–6.6 fps** | 7.6 fps |
| Score, all 8 clips | — | — | 794/807 | **796/807 (98.6%)** | 782/807 (96.9%) |
| Other-wallet frames | 0 | 0 | 0 | 0 | 0 |

(Round-3 ranges: two quiet runs a few hours apart; same code.)

---|---|---|---|---|
| Wallet visible | 404 ms (2.5 fps) | 217 ms | 191–211 ms | **161 ms (6.2 fps)** |
| Wallet in hand | 813 ms (1.2 fps) | 452–460 ms | 452 ms | 445 ms (2.2 fps) |
| Covered | 462 ms | 475 ms | 145–169 ms | **122 ms (8.2 fps)** |
| In pocket | 380 ms | 188 ms | 133 ms | **109 ms (9.2 fps)** |
| Live replay, clip 2 | ~2.5 fps | 3.9 fps | 4.4 fps | **6.6 fps** |
| Score, 6 original clips | 615/624 | 616/624 | 616/624 | 616/624 |
| Score, all 8 clips | — | — | 794/807 | **796/807 (98.6%)** |
| Other-wallet frames | 0 | 0 | 0 | 0 |

---|---|---|---|
| Wallet visible | 404 ms (2.5 fps) | 217 ms (4.6 fps) | 211 ms (4.7 fps) |
| Wallet in hand | 813 ms (1.2 fps) | 452–460 ms (≈2.2 fps) | ≈ same as round 1 |
| Covered | 462 ms | 475 ms | **169 ms (5.9 fps)** |
| In pocket | 380 ms | 188 ms | **133 ms (7.5 fps)** |
| Live replay, clip 2 | ~2.5 fps | 3.9 fps | 4.4 fps |
| Score, 6 original clips | 615/624 | 616/624 | 616/624 |
| Score, all 8 clips (2 new) | — | — | **794/807 (98.4%)** |
| Other-wallet frames | 0 | 0 | 0 |
| Drawn scenes (E2b) | 334/336, 8/8 | 334/336, 8/8 | 334/336, 8/8 |

---|---|---|---|
| Wallet visible | 404 ms (2.5 fps) | 217–224 ms (≈4.5 fps) | see note |
| Wallet in hand | 813 ms (1.2 fps) | 452–478 ms (≈2.1 fps) | −29% vs before round 2, same day |
| Covered | 462 ms | 475 ms | no whole-picture search; skipped 27–29% |
| In pocket | 380 ms | 188–250 ms | tracker only with a hand in view; skipped 35–47% |
| Score, 6 original clips | 615/624 | 616/624 | 612/624 |
| Score, all 8 clips (2 new) | — | — | **790/807 (98%)** |
| Other-wallet frames | 0 | 0 | 0 |
| Drawn scenes (E2b) | 334/336, 8/8 | 334/336, 8/8 | 334/336, 8/8 |

**Note:** round 2 was timed on a heavily loaded Mac (unchanged code ran 1.25–1.8× slower than in the morning). Busy-Mac numbers: visible 264 ms, in hand 489 ms, all 350 ms. Same-day comparison on the 6 original clips: 501 → 369 ms (−26%). A quiet-Mac re-run is needed for a clean comparison with round 1.

---|---|---|
| Wallet visible | 404 ms (2.5 fps) | 217–224 ms (≈4.5 fps) |
| Wallet in hand | 813 ms (1.2 fps) | 452–478 ms (≈2.1–2.2 fps) |
| All frames | 609 ms (1.6 fps) | 351–366 ms (≈2.8 fps) |
| Wallet in pocket | 380 ms (2.6 fps) | 188–250 ms (4–5 fps) |
| Live app, replay of clip 2 | ~2.5 fps | 3.9 fps |
| E0 score | 615/624 | 616/624 |
| Other-wallet frames | 0 | 0 |
| Drawn scenes (E2b) | 334/336, 8/8 final answers | 334/336, 8/8 |

Ranges are two runs of the same code (2026-10-11 and 2026-10-12). Other apps were open, so timings vary about ±15%.

---

## Before the research review

### 2026-10-04/05 · Setup and first measurements
- **What:** conda env, SAM 2.1 and EdgeTAM on the Mac GPU (MPS); `scripts/check_env.py`.
- **Speed, one object, full resolution:**

  | Model | Stock | With our attention patch |
  |---|---|---|
  | SAM 2.1 tiny | 1391 ms (0.7 fps) | 819 ms (1.2 fps) |
  | **EdgeTAM** | 277 ms (3.6 fps) | **190 ms (5.3 fps)** |

  EdgeTAM became the tracker.
- **Found:**
  - The memory step (memory attention) is the bottleneck.
  - PyTorch's built-in attention is 2× slower on the Mac GPU than plain matrix maths, so we patched it.
  - Half precision did not help.
- **Multi-object cost:** 4 objects ≈ 480 ms per picture (≈ 1 s over a long clip). EdgeTAM repeats the memory step for every object.

### 2026-10-05 · Drawn test scenes (lookalikes) and the cups clip
- **Drawn scenes (EdgeTAM alone):** a similar wallet nearby or touching, a hand covering it, the object taken away. The tracker never jumped to the lookalike, **but when a hand grabbed the object it silently tracked the hand** (22 of 45 frames on the hand).
- **Real shell game with 3 identical cups:** identity depends on fps.

  | fps given to the tracker | Cup labels correct |
  |---|---|
  | 30 | 100% |
  | 15 | 90% |
  | 10 | 61% |
  | 5 | 40% (silent swaps) |

  **Conclusion:** identical objects need about 15 fps or more.

## Research review 1 (2026-10-05, `RESEARCH.md` §1–7)

- **Main idea:** split the jobs.
  - fast follow;
  - check before remembering;
  - identity check;
  - explicit state ("in hand", "covered") instead of one model doing everything.
- **Experiments planned:** E0–E9.

### ~2026-10-06 · E1: fast/slow tracking (cheap optical flow between heavy frames): **rejected**
- **Cups clip, all frames correct:**

  | Setup | All frames correct |
  |---|---|
  | EdgeTAM at 5 fps | 40% |
  | + flow | 47% |
  | + a perfect prior (upper bound) | 91% |
  | EdgeTAM at 15 fps alone | **90%** |
  | 15 fps + flow | 71% (worse) |

- **Same compute placed differently:** no scheduling trick beat evenly spaced frames.
- **Learned:**
  - Flow fails exactly on the fast swaps that decide identity.
  - Restarting the tracker from a correct box works, but only if its confused recent memory is cleared.
- **Decision:** raw speed matters more than cleverness.

### ~2026-10-06 · E2: "does it still look like my object?" check: **adopted (variant B)**
- **The check:** size + DINOv2 appearance on the tracker's output. The tracker's memory is left as it is.
- **"Taken, hidden in fist":** 23 → 45/45. No regressions, never the other wallet.
- **Memory gating (variant C):** postponed. With a clean memory the tracker re-found the **wrong** wallet (18 frames).

### ~2026-10-06 · E2b: follow the hand that took the object: **adopted**
- **Drawn scenes:** 257/336 (tracker alone) → 302 (+ check) → **335/336** (+ hand states), 8/8 final answers, 0 other-wallet frames.
- **Learned:** the hand's path tells identical objects apart. "Moved, with a lookalike" went 15 → 48/48.

### 2026-10-06/07 · Live app on the Mac camera (`python -m compass.live`)
- **First live speed:** about 2.3–2.7 fps, and about 1.9 fps in cluttered "held" moments.
- **Fixes from the user's live tests:**

  | Problem | Fix |
  |---|---|
  | "Visible" shown with no outline | New "not sure" state |
  | Turned wallet rejected | Rotated and mirrored reference views |
  | Bad calibration | Fixed thresholds |
  | Late "put down" | Hand-off search every frame, not every 3rd |
  | Lost objects never re-found | Re-detection |
  | False re-find of a lookalike | Say "maybe" instead |
  | Session log not saved | Fixed |

- **Start-up:** 0.7–1 s → 0.37 s, by remembering lookalikes as spots instead of outlines.

### 2026-10-07 · E9: explain the change + MediaPipe hands: **adopted**
- **What:** blocked / turned / closer / farther / something else, plus a real hand detector (~24 ms).
- **Results:**
  - 14/14 controlled cases;
  - drawn scenes 263/264, with 0 critical mistakes;
  - E2b still 335/336.
- **Known limit:** fast identical cups are still wrong at live speed (11/31).

### 2026-10-09 · E0: the user's 6 real clips + pocket: **adopted**
- **Score:** 86% → **99% (615/624)** after fixes. 0 other-wallet frames.
- **Pocket clip:** 60% → **100%** (clothes segmentation + the holding hand's path).
- **Learned:**
  - real hands hold objects from below;
  - held objects look 20–45% bigger;
  - paper leaves slivers and shadows;
  - the face-based body-pose model fails without a face;
  - long sessions exposed a tracker pruning bug, now fixed.
- **Data:** `progress/data/2026-10-09_e0-pocket-baseline.json`

---

## Research review 2: speed (2026-10-09, `RESEARCH.md` §9)

**Question:** how to make it faster for edge devices, and later for several objects.

**Found:**
- **Trackers that do many objects in one pass:** only AOT (≤ 10 objects) and SAM 3.1 Multiplex (16 per pass, but 848M parameters, too big). EdgeTAM, SAM 2, Cutie and XMem repeat the memory step per object.
- **Detectors** (YOLO / YOLOE) are fast and their cost is flat in the number of objects. But they give boxes, are class-level ("a wallet"), and are weak while an object is held.
- **"Run the heavy model only when something changes"** papers:
  - MARLIN: up to 73% less energy;
  - Reducto: 51–97% of frames skipped.
- **Trim the memory step to around the object:** RS³-Prune, Efficient-SAM2 (1.4–1.8×).
- **Edge:**
  - Jetson + TensorRT: 6.9× over PyTorch on a SAM 2 variant.
  - Qualcomm runs the memory step on the CPU.
  - Apple's Neural Engine can't run the memory step in public conversions.
  - 8-bit gives no speed gain on an M2.

**Our probe** (`experiments/e5_speed_probe.py`, memory step for one object):

| Memory step variant | ms |
|---|---|
| Whole frame | 64 |
| Fewer memory frames (7 → 4) | 54 |
| Window 32×32 cells | 14.6 |
| **Window 24×24 cells** | **8.8** |
| 8 objects, whole frame | 467 |
| 8 objects, windows | 44 |
| Half precision | slower |

- **Image step by input size:** 1024 px 88 ms, 768 px 51 ms, 512 px 24 ms.
- **Data:** `progress/data/2026-10-09_speed-probe-memory-attention.json`

**User's direction (2026-10-09 → 10-11):**
- Make one object fast first, then add objects one after another or side by side. Cost may grow per object, but must not explode.
- The hardware decision comes later.
- Don't assume only 1–2 objects move at once.
- The project is open source, so AGPL tools are OK.

### 2026-10-09 · Where the time goes, one object, full system (baseline for the speed work)
`experiments/e5_time_breakdown.py`, 665 frames of the E0 clips, ms per picture:

| Part | Wallet visible | Wallet in hand |
|---|---|---|
| Tracker (EdgeTAM) | 253 | 289 |
| Hands | 36 | 38 |
| Appearance check | 21 | 52 |
| Explainer | 69 | 9 |
| Search (scan + outlines) | 15 | 336 |
| Clothes | 0 | 56 |
| Rest | 10 | 33 |
| **Total** | **404 (2.5 fps)** | **813 (1.2 fps)** |

The tracker alone takes 218 ms. 29 ms of that is preparing the picture.

---

## Speed round 1 (2026-10-11): one object

### Step 1 · Easy fixes: **kept**
- **Changes:**
  - the picture's normalisation moved to the GPU;
  - hand detection in parallel with the tracker;
  - the search reuses the tracker's image features instead of encoding the frame again (and its outlines are now full resolution);
  - the clothes result is reused for up to 1 s;
  - search candidates are checked in one batch.
- **Result:**
  - visible 404 → 310 ms (3.2 fps);
  - in hand 813 → 487 ms (2.1 fps);
  - all 609 → 411 ms;
  - **E0 617/624** (+2).
- **Data:** `progress/data/2026-10-11_speed-step1-easy-fixes.json`
- **Lesson:** a GPU sync from the hand thread crashed the timing script (Metal abort). The app itself was fine; only the measuring code was fixed.

### Step 2 · Skip the tracker while the object rests and nothing changes: **kept**
- **Rule:** fewer than 1% of the (blurred, quarter-size) pixels around the object changed by more than 25 grey levels, and no hand is near. Then reuse the last answer. The tracker still runs at least every 2 s.
- **Result (paired runs, same conditions):**
  - visible 404 → **243 ms (4.1 fps)**; skipped 31% of visible frames;
  - a skipped frame costs ~18 ms instead of ~300;
  - **E0 620/624** (+3).
- **Data:** `progress/data/2026-10-11_speed-step2-skip.json`, `..._speed-step2-noskip-paired.json`

### Step 3 · Track only around the object (`compass/window.py`) + faster resize: **kept**
- **Window:** the memory step and the outline decoder work on a square around the last outline (box + half its size on each side). The whole frame is used when there is no outline.
  - **Tracker only, compared with the whole frame:** mean overlap (IoU) 0.989; 0.6% of frames below 0.8.
- **Resize:** OpenCV instead of PIL. 15 → 1.7 ms, 0.3 grey levels different.
- **Tracker step:** 229 → 164 ms.

  | Part | Before | After |
  |---|---|---|
  | Preparing the picture | 29 | 7 |
  | Memory step | 62 | 22 |

  The image step is unchanged at 88 ms.
- **E0:**
  - 617/624 with the resize only (−3: borderline "turned in hand" frames);
  - **616/624** with the window too (−1, a paper sliver).
- **Why the gain is modest:** the wallet is big (~19 of 64 grid cells across), so its window is 36–48 cells. Close-ups use the whole frame.
- **Data:** `progress/data/2026-10-11_speed-step3-resize-only.json`, `..._speed-step3-window.json`, `..._window-tracker-only-margin-0.5.json`

### Tried and dropped
- **Tighter window** (quarter-box margin): IoU 0.984, but 7% of frames below 0.8 on the fast pocket move, for little extra speed. Data: `..._window-tracker-only-margin-0.25.json`
- **Outline only the search spots near the hand:**
  - with a fixed 80 px margin, it lost the close-up wallet (clip 4: 178 → 169);
  - with a margin based on the object's size, the score came back, but it saved no time. Removed.

### Round 1 final
- **Speed** (two runs of this exact code):
  - visible 217–224 ms (≈4.5 fps);
  - in hand 452–478 ms (≈2.1–2.2 fps);
  - all 351–366 ms (≈2.8 fps);
  - live replay 3.9 fps.
  - An earlier quote of 224–240 / 478–503 included a run with the dropped search change.
- **Accuracy:** E0 616/624; 0 other-wallet frames; pocket clip 100%; E2b 334/336, 8/8.
- **Live app:** the window and skipping are on by default; `--no-window` and `--no-skip` turn them off.
- **Data:** `progress/data/2026-10-11_speed-round1-final.json`

**What's left** (in hand, ~490 ms):
- tracker ~200 ms (image step 88, memory save 23, memory step ~22, decoder ~15);
- search outlines ~130 ms (17 ms × 10 candidates);
- search scan ~65 ms;
- appearance check ~40 ms.

---

## Speed round 2 (2026-10-12): one object, hidden states, search, explainer

**Test set:** two new clips recorded by the user, added to E0 (30 fps, replayed at the same 3 pictures/s):
- `20261012-113549`: wallet into the shorts pocket, taken out ~15 s later, put on the table, nudged, lifted, put down.
- `20261012-113643`: paper brought over the wallet; behind the paper the wallet is slid ~150 px back; the paper is lifted.

E0 now has 8 clips, 807 scored frames.

**Timing conditions:** the Mac was heavily loaded all afternoon (load average 6–12, against ~3 before).
- Unchanged code ran 1.25–1.8× slower than in the morning. The tracker step took 187 ms vs 149 ms; the old explainer 191–205 ms vs 113 ms.
- So round 2 is judged by **same-day, back-to-back** comparisons, not against round 1's numbers.

### Step 2a · Hidden states: watch only where the object can be: **kept**
- **Covered:**
  - watch a window around the last spot;
  - skip completely while nothing changes there;
  - no whole-picture search any more (it used to run every covered frame).
  - The spot shows plain background → "no longer where it was covered" → **lost**.
  - The cover is lifted and it's still there → visible.
- **In hand:** the tracker's window follows the holding hand.
- **In pocket:** the tracker runs only when a hand is in view, in a window around the hand(s).
- **Lost / left the picture:** whole picture (tracker + search) once per second of video (user's rule).
- **Result:**
  - old 6 clips unchanged (616/624);
  - paper-slide clip **64/64**: the wallet moved under the cover and was found at its new spot;
  - skipped 27–29% of covered frames and 35–47% of in-pocket frames.
- **But** the new pocket clip scored **40/119**. It exposed two pocket-rule gaps (next).
- **Data:** `progress/data/2026-10-12_r2a-hidden-states-window.json`

### Step 2a-fix · Pocket rules (found by the new clip): **kept**
- **Gap 1, going in:** the hand went *into* the pocket with the wallet, and the hand detector lost it for ~10 s. The rule needed a visible hand at the clothes.
  - **Fix:** the holding hand vanishing at the clothes together with the object also counts. Conditions:
    - not at the picture's edge;
    - **and the tracker sees < 25% of the object.**
  - A first version without the last condition broke clip 4 (178 → 135). There a hand holding the wallet close to the camera, in front of the shirt, also went undetected for ~10 s, but the tracker still saw ~100% of the wallet.
- **Gap 2, coming out:** back on the table the wallet scored 0.79–0.81, below the 0.85 the exit rule needed, so it stayed "in pocket" for 16 s.
  - **Fix:** a full-size outline (≥ 70%) away from the pocket spot, at the partial-view look level (0.75), also counts.
  - A faint outline at the pocket still doesn't.
- **Result:**
  - new pocket clip **40 → 114/119**;
  - all 8 clips **794/807 (98.4%)**;
  - old 6 back to 616/624.
- **Data:** `progress/data/2026-10-12_r2a-pocket-rule-fixes.json`

### Step 2b · Cheaper in-hand search (pre-check): **dropped** (off by default; see the quiet-Mac re-run below)
- **Before outlining,** do a quick look check of an object-sized box around each of the ≤ 10 spots, in one batch. Then outline only the best 3 that score ≥ 0.6, plus the spot at the object's last place.
- **Paired timing** (8 clips, back to back):

  | | Before | After |
  |---|---|---|
  | Wallet in hand | 681 ms | **576 ms (−15%)** |
  | All frames | 452 ms | **391 ms (−13%)** |
  | Search outlines (in hand) | 184 ms | 51 ms |
  | The pre-check's own look checks | — | +75 ms |

- **E0:** 794 → **790/807**. The −4 frames are a put-down or pick-up decided one picture earlier or later, mostly swapping between "visible" and "in hand"; a few "not sure".
- **Data:** `progress/data/2026-10-12_r2b-search-precheck.json`
- **Possible next:** make the pre-check's look checks cheaper (smaller crops) to recover most of the 75 ms.
- **Correction after the quiet-Mac re-run:** the saving came from the busy GPU making outlines slow. On a quiet Mac the pre-check costs about as much as it saves (look checks +75 ms, outlines −80 ms) and loses 4 frames. Now **off by default** (`ObjectState(precheck=True)` / `--precheck` to try it).

### Step 2c · Explainer on a crop: **kept** (same answers)
- **Profile:** 97 ms per call.
  - 30 ms: the "is it background?" colour test (2,000 × 1,500 colour pairs compared one by one).
  - 13 ms: growing masks with ~600 px squares on the full frame.
  - 11 ms: ORB detail matching.
  - The rest: full-frame colour conversions and blurs.
- **Changes:**
  - the colour test as one matrix product;
  - everything after the motion estimate done on a crop around the object;
  - large mask growth done with a distance map (exact).
- **Check:** old vs new explainer on 201 frames of 3 clips gave **identical labels and numbers**, and ran 2.6–4× faster (113 → 43 ms; in a busier run 205 → 51 ms).
- **Tried and dropped:** ORB on a crop. It changed details on ~20 of 201 frames, and a few labels, because ORB's image pyramid samples a crop differently.
- **Also checked:** using the distance map for small growth in the rules was *slower* (4 ms vs 0.7–3 ms), so it's used only above 100 px.

### Round 2 final
- **Accuracy:**
  - all 8 clips **790/807 (98%)**, 0 other-wallet frames;
  - drawn scenes 334/336, 8/8.
- **Same-day comparison, old 6 clips** (round-2 changes off vs on; the Mac equally busy):

  | | Before round 2 | After round 2 |
  |---|---|---|
  | All frames | 501 ms | **369 ms (−26%)** |
  | Wallet visible | 321 ms | **248 ms** |
  | Wallet in hand | 689 ms | **488 ms (−29%)** |
  | In pocket | 143 ms | **124 ms** |
  | Explainer when covered | ~110 ms | **~28 ms** |
  | Covered frames running a whole-picture search | all | none |

- **Measured today on all 8 clips** (busy Mac):
  - all 350 ms (2.9 fps);
  - visible 264 ms (3.8 fps);
  - in hand 489 ms (2.0 fps);
  - covered 253 ms (4.0 fps);
  - in pocket 187 ms (5.4 fps).
- **Against round 1:** a clean comparison needs a quiet Mac. Round 1 was measured in the morning at about half today's load.
- **Data:** `progress/data/2026-10-12_r2c-explainer-crop-final.json`

**What's left** (in hand, ~490 ms on a busy Mac):
- tracker ~210 ms;
- appearance checks ~115 ms (the pre-check 75 + the gate);
- search scan ~65 ms;
- search outlines ~40 ms.

---

### Round 2 · Quiet-Mac re-run (2026-10-12, 22:39–23:03, user asked)
- **How busy the Mac was:** a reference measurement (the old explainer on 201 fixed frames) took 125 ms, against 113 ms at its quietest and 191–205 ms in the busy afternoon. So this run is close to quiet. The load still rose during the in-hand pair (both runs slower), so its absolute numbers are high.
- **Round 1 vs round 2,** same clips, ms per picture:

  | State | Round 1 | Round 2 | Change |
  |---|---|---|---|
  | Covered | 475 | **169** | **2.8× faster** (no whole-picture search; explainer 184 → 26 ms; 27% of frames skipped) |
  | In pocket | 188 | **133** | **1.4× faster** (tracker only with a hand in view; 35–47% skipped) |
  | Visible | 217 | 211 | same (explainer 44 → 23 ms, tracker unchanged) |
  | In hand (clips 1 + 4) | 460 | ~479 | same within noise (pre-check off) |
  | Live replay, clip 2 | 3.9 fps | **4.4 fps** | |

- **Corrected two wrong conclusions from the busy run:**
  1. **The pre-check:** it doesn't pay off when quiet, so it's off.
  2. **"In pocket slower":** hand detection had spiked to 300–660 ms during the busy run. Alone, and in the quiet re-run, it takes 13–39 ms. The helper thread is not the cause: 13 ms on either thread.
- **Final accuracy** (pre-check off): **794/807 (98.4%)**.
  - 6 original clips 616/624, the same as round 1;
  - new pocket clip 114/119;
  - paper slide 64/64;
  - 0 other-wallet frames.
- **Lesson:** time only on a quiet Mac, and run a reference measurement first. Busy-Mac runs exaggerate GPU-heavy parts and can flip decisions.
- **Data:** `progress/data/2026-10-12_r2-final-quiet-mac.json`

---

## Speed round 3 (2026-10-13): Core ML image step + windowed search outlines

### Step 3a · EdgeTAM's image encoder through Core ML (`compass/coreml_encoder.py`): **kept** (default in the live app)
- **What:** our own export of EdgeTAM's `forward_image` (the three feature maps the tracker uses), with the ImageNet normalisation inside the model.
  - EdgeTAM's shipped export is for single images: it skips the normalisation and adds the "no memory" embedding, which the video tracker must not have.
  - The positional encodings don't depend on the picture, so they are computed once.
  - Export: `python -m compass.coreml_encoder` (also run by `setup_third_party.sh` on a Mac). It writes `third_party/models/edgetam_image_encoder_video.mlpackage`, 9.6 MB.
- **Encoder alone:**

  | Runtime | ms | Feature difference vs PyTorch |
  |---|---|---|
  | PyTorch (MPS) | 93 | — |
  | **Core ML, Neural Engine** | **27** | ~5% on average (fp16) |
  | Core ML, GPU | 57 | 0.8% |

- **Tracker only, all 8 clips** (window on in both): masks vs PyTorch mean IoU **0.993**, 0.3% of frames below 0.8 (only the new pocket clip). Tracker step 237 → 153 ms (−36%).
- **E0:** **792/807** (−2 vs round 2; clip 4 −4, new pocket clip +2). 0 other-wallet frames.
- **Data:** `progress/data/2026-10-13_r3a-coreml-encoder.json`, `..._r3a-coreml-tracker-only.json`

### Step 3b · Outline each search spot in a window around it: **kept, but no speed gain for the user's wallet**
- **What:** the mask decoder works on a square of EdgeTAM's feature grid around the spot (twice the registered object's size, at least 16 of 64 cells; the whole picture above 48), with the true positions, as in `compass/window.py`.
- **Outlines only:** 172 → 143 ms per in-hand frame. Mean IoU vs whole-picture outlines 0.970 (2.8% below 0.8).
  - The wallet is big (windows 48–52 of 64 cells, so nearly the whole picture).
  - Each spot is decoded separately.
- **E0 with 3a + 3b:** **796/807 (98.6%)**, 0 other-wallet frames.
- **Full system:** search outlines ~120 ms per in-hand frame either way. Small objects should gain more.

### Round 3 · Quiet-Mac paired timing (2026-10-13 00:48–01:10)
- **Conditions:** the old-explainer reference measured 104–130 ms around both runs; a first PyTorch run during a load spike was discarded.
- **Results** (all 8 clips):

  | State | Round-2 settings (PyTorch) | **Round 3 (Core ML + windowed outlines)** |
  |---|---|---|
  | Visible | 191 ms (5.2 fps) | **161 ms (6.2 fps)** |
  | Covered | 145 ms (6.9 fps) | **122 ms (8.2 fps)** |
  | In pocket | 133 ms (7.5 fps) | **109 ms (9.2 fps)** |
  | In hand | 452 ms (2.2 fps) | 445 ms (2.2 fps) |
  | All frames | 292 ms | 284 ms |
  | Live replay, clip 2 | 4.4 fps | **6.6 fps** |

- **Tracker part per state** (ms): visible 131 → 92, covered 109 → 74, pocket 100 → 70, in hand 188 → 148.
- **In hand is unchanged** because the in-hand search dominates it (scan 60–80 + outlines ~120 ms) and round 3 didn't reduce it.
- **Data:** `progress/data/2026-10-13_r3-pair-torch-quiet.json` (round-2 settings), `..._r3ab-final.json`

**What's left:**
- **In hand (~445 ms):** the search, ~200 ms. It runs on every in-hand frame to catch a put-down quickly. Next idea: search only when something new appears along the hand's path (a cheap pixel check against the picture from when the hand took the object), or a fast detector.
- **Visible (~160 ms):** tracker ~130 ms per tracked frame (memory steps + decoder in PyTorch now), hands ~30, explainer ~25.

---

### Round 3 · Assessment of 3a/3b and extra tries (2026-10-13)
- **3a improvement tried: the search's scan model (DINOv2 ViT-S at 448 px) on Core ML.**
  - The TorchScript route failed on timm's size arithmetic; the `torch.export` route worked.
  - 54 → 32 ms (Neural Engine). Tokens near-identical (cosine 0.9998; similarity maps within 0.04).
  - **Not wired in:** it saves ~22 ms of a ~445 ms in-hand frame. The round-4 idea (search only when something new appears along the hand's path) would make the scan rare anyway.
  - The model is at `third_party/models/dinov2_vits14_448_tokens.mlpackage` if needed later.
- **3b:** no change. Batching all windowed outlines into one decoder call would need a rewrite of SAM's mask decoder (it accepts one positional table per call). Not worth it for a big object.

### Step 3c · torch.compile on the tracker (memory step, decoder, memory encoder): **dropped**

| Mode | Tracker step | Outlines |
|---|---|---|
| Normal | 209 ms | — |
| `inductor` | 263 ms (slower; "does not support code generation for complex operators": the memory step's rotary positions use complex numbers) | identical (IoU 1.000) |
| `aot_eager` | 214 ms (no gain) | identical (IoU 1.000) |

Worth retrying on NVIDIA (TensorRT / inductor there is mature).

### Step 3d · 768 px input picture (EdgeTAM was trained at 1024)
- **What changes with size:** the feature grid becomes 48×48 instead of 64×64. The memory compressor (perceiver) adapts (16×16 windows of 3 cells instead of 4). The memory step's query position table must match: `q_sizes=[48,48]`, set in `compass/models.py`.
- **Tracker only, PyTorch both, all 8 clips:**
  - IoU vs 1024 is **0.978** mean, 0.9% of frames below 0.8 (worse on fast moves: pocket clips 2.4–4.9%);
  - tracker step 177 → 115 ms (−35%).
  - For comparison, Core ML at 1024 agreed at 0.993.
- **Full system (Core ML at both sizes, windowed outlines; paired on a quiet Mac, reference 108 ms before and after):**

  | State | 1024 px | 768 px |
  |---|---|---|
  | Visible | 131 ms (7.6 fps) | **108 ms (9.3 fps)** |
  | In hand | 352 ms (2.8 fps) | **289 ms (3.5 fps)** |
  | Covered | 94 ms (10.6 fps) | 77 ms (13.0 fps) |
  | In pocket | 93 ms (10.8 fps) | 74 ms (13.6 fps) |
  | All frames | 227 ms (4.4 fps) | **177 ms (5.6 fps), +28%** |
  | E0, 8 clips | **796/807 (98.6%)** | 782/807 (96.9%), −14 |

  - The 14 lost frames are in clip 1 (turned in hand) and clip 4 (close to the camera): 12× "in hand" → "not sure", 2× → "covered", 1× → "lost". Honest misses, 0 other-wallet frames. The pocket and paper clips are unchanged.
- **Decision:** keep **1024 as the default** (accuracy) and offer **768 as a speed option** (`python -m compass.live --size 768`; Core ML model: `python -m compass.coreml_encoder 768`).
- **Data:** `progress/data/2026-10-13_r3d-768px.json`, `..._r3-final-1024-quiet.json`, `..._r3d-768-tracker-only.json`

### Round 3 final
- **Default:** Core ML image encoder at 1024 px + windowed search outlines.
- **Quietest paired timing:**
  - visible **131 ms (7.6 fps)**;
  - in hand **352 ms (2.8 fps)**;
  - covered 94 ms (10.6 fps);
  - in pocket 93 ms (10.8 fps);
  - all 227 ms (4.4 fps).
- **E0:** 796/807, 0 other-wallet frames.
- **Speed option (768 px):** visible 9.3 fps, in hand 3.5 fps, E0 782/807.
- **Dropped:** torch.compile (slower on the Mac). Scan model on Core ML: tried, not wired in (−22 ms).
- **Timing noise:** the same code measured visible 131–161 ms on two quiet runs a few hours apart. Paired back-to-back runs remain the only fair comparison.

## Round 2 plan (agreed 2026-10-12; done above)

1. **Hidden states use a window too** (user's refinement):
   - **Covered:** window around the last spot. Skip entirely while nothing changes there.
   - **In hand:** window around the hand.
   - **In pocket:** watch only the pocket area. Run the tracker when a hand comes near.
   - **Lost or left the picture:** only then check the whole picture, every 1–2 s, or when something moves at the edge where it left.
   - If a window shows the object is not where we believed (for example, the cover is lifted and nothing is under it), switch to "lost". From then on the whole-picture checks run.
2. **Cheaper in-hand search:** a quick appearance pre-check of a box around each spot (one batch), then outline only the 1–3 that pass. YOLOE as an option later.
3. **Explainer and rules on half-size pictures.**
4. **Core ML for the image step**, then a 768 px input (accuracy test).
5. **Overlap pictures on the two chips:** the GPU works on the next picture while the CPU finishes the current one. Last, because it is the most complex.
