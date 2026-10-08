"""E9: can the system tell WHY the object looks different? blocked / turned / closer-farther / something else.

Six drawn scenes with the right answer on every frame, run through the same code as the live app
(compass.holder.ObjectState with the explainer compass.explain; no hand detector, the drawn hands are not real):

  book_slide    a book slides over up to 60% of the wallet, stays, slides away     -> "partly hidden" while >=15% covered
  hand_partial  a drawn hand covers half the wallet, stays, leaves                 -> "partly hidden"
  turn          the wallet turns 0 -> 90 degrees in place                          -> shown blue, never "blocked"
  flip          the wallet flips to its plain back side (edge-on frames unscored)  -> shown blue, never "blocked"
  closer        the wallet grows to 1.7x (toward the camera) and back             -> shown blue, never "taken by a hand"
  switched      the wallet vanishes; an identical wallet lies next to it           -> never blue on the other wallet

Critical mistakes (must be 0): "turned" while blocked, "blocked" while turned, "taken by a hand" while moving closer,
blue on the other wallet.

Run from repo root:  python experiments/e9_explain.py
Writes experiments/output/e9/: <scene>.mp4, index.html (open it), e9_results.json
"""
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

import cv2
import numpy as np
import torch

from compass.appearance import Appearance
from compass.holder import Finder, ObjectState, Segmenter
from compass.live import draw, text
from compass.models import video_predictor
from compass.tracker import StreamingTracker
from compass.video_io import make_web_playable

OUT = ROOT / "experiments" / "output" / "e9"
SIZE, N = 512, 45
rng = np.random.default_rng(7)


# ---------------------------------------------------------------- drawing


def table():
    img = np.full((SIZE, SIZE, 3), (60, 110, 150), np.uint8)
    return cv2.add(img, rng.integers(0, 25, img.shape, dtype=np.uint8))


def wallet_sprites():
    """Front (stitching, logo, snap, leather grain) and back (plain darker leather) of a 110x76 wallet."""
    front = np.full((76, 110, 3), (40, 70, 120), np.uint8)
    grain = np.random.default_rng(1).integers(0, 30, front.shape, dtype=np.uint8)
    front = cv2.add(front, grain)
    cv2.rectangle(front, (6, 6), (103, 69), (90, 140, 200), 2)
    cv2.putText(front, "CW", (22, 50), cv2.FONT_HERSHEY_DUPLEX, 1.1, (160, 200, 230), 2, cv2.LINE_AA)
    cv2.circle(front, (88, 38), 7, (200, 200, 200), -1)
    cv2.line(front, (70, 12), (70, 64), (25, 45, 85), 2)
    back = cv2.add(np.full((76, 110, 3), (30, 50, 85), np.uint8), np.random.default_rng(2).integers(0, 18, front.shape, dtype=np.uint8))
    return front, back


FRONT, BACK = wallet_sprites()


def place(img, sprite, centre, angle=0.0, sx=1.0, sy=1.0):
    """Paste the sprite rotated/scaled at `centre`. Returns its mask in the picture."""
    h, w = sprite.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    m[:, :2] = m[:, :2] @ np.diag([sx, sy])
    m[:, 2] += np.array(centre) - m[:, :2] @ np.array([w / 2, h / 2])
    warped = cv2.warpAffine(sprite, m, (SIZE, SIZE))
    mask = cv2.warpAffine(np.full((h, w), 255, np.uint8), m, (SIZE, SIZE)) > 127
    img[mask] = warped[mask]
    return mask


def box_of(mask):
    ys, xs = np.nonzero(mask)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)


# ---------------------------------------------------------------- scenes: frames + truth per frame


def book_slide():
    frames = []
    for i in range(N):
        img = table()
        full = place(img, FRONT, (256, 280))
        right = -10 + (i - 10) * 18 if 10 <= i < 25 else 266 if 25 <= i < 32 else 266 - (i - 32) * 23 if i < 44 else -10
        book = np.zeros((SIZE, SIZE), bool)
        if right > 0:
            book[180:390, max(right - 150, 0):right] = True
            img[book] = (150, 80, 40)
            cv2.rectangle(img, (max(right - 150, 0), 180), (right, 390), (110, 55, 25), 3)
        frames.append((img, {"full": full, "visible": full & ~book, "kind": "cover"}))
    return frames


def hand_partial():
    frames = []
    for i in range(N):
        img = table()
        full = place(img, FRONT, (256, 280))
        hand = np.zeros((SIZE, SIZE), bool)
        if 12 <= i < 32:
            top = min(150 + (i - 12) * 15, 285)  # palm bottom edge comes down to cover ~half the wallet
            hm = np.zeros((SIZE, SIZE), np.uint8)
            cv2.rectangle(hm, (200, 0), (300, top - 40), 1, -1)  # arm
            cv2.rectangle(hm, (185, top - 90), (320, top - 20), 1, -1)  # palm
            for k in range(4):  # fingers
                cv2.rectangle(hm, (192 + k * 33, top - 25), (215 + k * 33, top), 1, -1)
            hand = hm.astype(bool)
            img[hand] = (120, 160, 205)
        frames.append((img, {"full": full, "visible": full & ~hand, "kind": "cover"}))
    return frames


def turn():
    frames = []
    for i in range(N):
        img = table()
        angle = float(np.clip((i - 5) * 3, 0, 90))
        full = place(img, FRONT, (256, 280), angle)
        frames.append((img, {"full": full, "visible": full, "kind": "turn", "angle": angle}))
    return frames


def flip():
    frames = []
    for i in range(N):
        img = table()
        if i < 10:
            sx, sprite = 1.0, FRONT
        elif i < 20:
            sx, sprite = 1.0 - (i - 9) * 0.09, FRONT
        elif i < 30:
            sx, sprite = 0.1 + (i - 19) * 0.09, BACK
        else:
            sx, sprite = 1.0, BACK
        full = place(img, sprite, (256, 280), 0, max(sx, 0.08), 1.0)
        frames.append((img, {"full": full, "visible": full, "kind": "turn", "edge_on": sx < 0.35}))
    return frames


def closer():
    frames = []
    for i in range(N):
        img = table()
        s = 1.0 + 0.7 * (min(i, 24) - 5) / 19 if 5 <= i < 25 else 1.7 - 0.7 * (i - 24) / 20 if i >= 25 else 1.0
        full = place(img, FRONT, (256, 280), 0, s, s)
        frames.append((img, {"full": full, "visible": full, "kind": "scale", "scale": s}))
    return frames


def switched():
    frames = []
    for i in range(N):
        img = table()
        other = place(img, FRONT, (340, 280))
        full = place(img, FRONT, (215, 280)) if i < 15 else np.zeros((SIZE, SIZE), bool)
        frames.append((img, {"full": full, "visible": full, "other": other, "kind": "gone"}))
    return frames


SCENES = {
    "book_slide": ("A book slides over up to 60% of the wallet, stays, and slides away.", book_slide),
    "hand_partial": ("A (drawn) hand covers about half of the wallet, stays, and leaves.", hand_partial),
    "turn": ("The wallet turns in place from 0 to 90 degrees.", turn),
    "flip": ("The wallet flips over to its plain back side.", flip),
    "closer": ("The wallet grows to 1.7x its size (moved toward the camera) and back.", closer),
    "switched": ("The wallet vanishes; an identical wallet lies right next to it.", switched),
}


# ---------------------------------------------------------------- scoring


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / u if u else 0.0


def system_label(step):
    if step.mask.any():
        note = step.note or ""
        if "partly hidden" in note:
            return "partly hidden"
        if "turned" in note:
            return "turned"
        if "moved closer" in note or "moved farther" in note:
            return note.split(" - ")[0]
        return "visible"
    if step.maybe is not None:
        return "maybe"
    return step.state


def judge(truth, step):
    """(expected, ok, critical, scored) for one frame."""
    label = system_label(step)
    shown = step.mask.any()
    on_obj = shown and truth["visible"].any() and iou(step.mask, truth["visible"]) >= 0.3
    kind = truth["kind"]
    if kind == "cover":
        covered = 1 - truth["visible"].sum() / max(truth["full"].sum(), 1)
        if covered < 0.05:
            return "visible", on_obj and label == "visible", False, True
        if covered < 0.15:
            return "borderline", True, False, False
        return "partly hidden", on_obj and label == "partly hidden", label == "turned", True
    if kind == "turn":
        if truth.get("edge_on"):
            return "edge-on (not scored)", True, False, False
        expected = "turned" if truth.get("angle", 90) >= 15 or "angle" not in truth else "visible"
        return expected, on_obj and label in ("visible", "turned"), label == "partly hidden", True
    if kind == "scale":
        return "visible / closer", on_obj and label in ("visible", "moved closer", "moved farther"), step.state == "held", True
    # gone
    on_other = shown and iou(step.mask, truth["other"]) >= 0.3
    if truth["full"].any():
        return "visible", on_obj, on_other, True
    return "not shown", not shown, on_other, True


# ---------------------------------------------------------------- run + render


def render(img, step, truth_text, ok, scored, verdicts, t):
    vis = draw(img, step, "", "", "", False)
    canvas = np.full((SIZE + 70, SIZE, 3), (13, 13, 13), np.uint8)
    canvas[:SIZE] = vis
    colour = (12, 163, 12) if ok else (59, 59, 208)
    word = ("CORRECT" if ok else "WRONG") if scored else "not scored"
    text(canvas, f"frame {t}  truth: {truth_text}", (10, SIZE + 22), 0.5, (195, 194, 183))
    text(canvas, word, (10, SIZE + 52), 0.6, colour if scored else (137, 135, 129), 2)
    x0, x1 = 160, SIZE - 10
    cw = (x1 - x0) / len(verdicts)
    for i, v in enumerate(verdicts[: t + 1]):
        c = (12, 163, 12) if v == "ok" else (59, 59, 208) if v in ("bad", "critical") else (60, 60, 60)
        cv2.rectangle(canvas, (int(x0 + i * cw), SIZE + 38), (int(x0 + (i + 1) * cw) - 1, SIZE + 56), c, -1)
    return canvas


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("mps")
    obj = ObjectState(StreamingTracker(video_predictor(dev)), Appearance(dev), Finder(dev), Segmenter(dev))
    results, cards = {}, []
    for name, (what, make) in SCENES.items():
        frames = make()
        steps = [obj.start(frames[0][0], box_of(frames[0][1]["full"]))] + [obj.update(img) for img, _ in frames[1:]]
        judged = [judge(t, s) for (_, t), s in zip(frames, steps)]
        verdicts = ["skip" if not sc else "critical" if crit else "ok" if ok else "bad" for _, ok, crit, sc in judged]
        scored = [v for v in verdicts if v != "skip"]
        n_ok, n_crit = scored.count("ok"), scored.count("critical")
        labels = [system_label(s) for s in steps]
        results[name] = {"what": what, "scored": len(scored), "correct": n_ok, "critical": n_crit,
                         "accuracy": round(n_ok / max(len(scored), 1), 3),
                         "system_labels": {k: labels.count(k) for k in dict.fromkeys(labels)},
                         "per_frame": [{"frame": i, "expected": e, "system": l, "verdict": v, "note": s.note}
                                       for i, ((e, _, _, _), l, v, s) in enumerate(zip(judged, labels, verdicts, steps))]}
        print(f"{name:13s} correct {n_ok:2d}/{len(scored):2d} ({n_ok / max(len(scored), 1):.0%}) | critical mistakes {n_crit} | "
              f"system said: {results[name]['system_labels']}")
        path = OUT / f"{name}.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), 5, (SIZE, SIZE + 70))
        for t, ((img, truth), step, (exp, ok, _, sc)) in enumerate(zip(frames, steps, judged)):
            writer.write(render(img, step, exp, ok, sc, verdicts, t))
        writer.release()
        make_web_playable(path)
        cards.append(f"<section><h2>{name}</h2><p>{html.escape(what)}<br><b>Correct:</b> {n_ok} / {len(scored)} frames "
                     f"({n_ok / max(len(scored), 1):.0%}) &middot; <b>critical mistakes:</b> {n_crit}</p>"
                     f"<video src='{name}.mp4' controls loop muted playsinline preload='metadata'></video></section>")
    tot_ok = sum(r["correct"] for r in results.values())
    tot = sum(r["scored"] for r in results.values())
    crit = sum(r["critical"] for r in results.values())
    print(f"\ntotal correct {tot_ok}/{tot} ({tot_ok / tot:.0%}) | critical mistakes {crit}")
    (OUT / "e9_results.json").write_text(json.dumps(results, indent=1, default=str))
    rows = "".join(f"<tr><td>{n}</td><td>{html.escape(r['what'])}</td><td><b>{r['correct']}</b> / {r['scored']} "
                   f"({r['accuracy']:.0%})</td><td>{r['critical']}</td></tr>" for n, r in results.items())
    (OUT / "index.html").write_text(f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>E9 Explain The Change</title>
<style>
:root{{--plane:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e1e0d9}}
@media (prefers-color-scheme:dark){{:root{{--plane:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--line:#2c2c2a}}}}
body{{margin:0;padding:24px 16px;background:var(--plane);color:var(--ink);font:15px/1.55 system-ui,sans-serif}}
main{{max-width:1100px;margin:auto}} h1{{font-size:22px;margin:0 0 8px}} h2{{font-size:17px;margin:0 0 6px}}
p,td,th{{color:var(--ink2)}} b{{color:var(--ink)}}
section,.box{{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px;margin:16px 0}}
.wrap{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-size:14px}}
td,th{{text-align:left;padding:8px;border-bottom:1px solid var(--line);vertical-align:top}}
video{{width:100%;max-width:520px;border-radius:6px;background:#000}}
</style></head><body><main>
<h1>E9: does the system say <i>why</i> the object looks different?</h1>
<p>Drawn scenes, the same code as the live app. Blue = shown as your wallet, with a note (partly hidden / turned /
moved closer); grey = not sure; yellow = a hand. Under each video: CORRECT / WRONG per frame against the truth.</p>
<div class="box wrap"><table><tr><th>Scene</th><th>What happens</th><th>Correct frames</th><th>Critical mistakes</th></tr>
{rows}<tr><td><b>All</b></td><td></td><td><b>{tot_ok} / {tot} ({tot_ok / tot:.0%})</b></td><td><b>{crit}</b></td></tr></table></div>
{''.join(cards)}
</main></body></html>""")
    print(f"wrote {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
