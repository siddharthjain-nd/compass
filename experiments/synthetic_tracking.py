"""Synthetic tracking tests with known ground truth, rendered as videos you can watch.

Each scene is drawn by code, so on every frame we know where YOUR object is (even when it is
hidden in a hand) and where the OTHER, similar-looking wallet is. EdgeTAM gets one box on your
object at frame 0 and then runs on its own.

For every frame we compare:
  should highlight   your object if any of it can be seen, otherwise nothing
  tracker highlights your object | the hand | nothing | the OTHER wallet | both wallets
and mark the frame CORRECT when they match.

"Tracker highlights your object" means >= 70% of its mask lies inside your object's real outline.
The size chart (dashed line = 1.6x the starting size) is a signal a real system could use without
ground truth; compare it with the result rows to see where that signal fails.

Run from repo root:
  python experiments/synthetic_tracking.py                 # all scenarios
  python experiments/synthetic_tracking.py taken_hidden    # one scenario
Writes experiments/output/<scenario>.mp4, <scenario>.png (key frames) and index.html
(all videos on one page: open experiments/output/index.html).
"""
import html
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "third_party" / "EdgeTAM")]

import cv2
import numpy as np
import torch

from check_env import patch_attention_for_mps

OUT = ROOT / "experiments" / "output"
SIZE = 512
DRIFT = 1.6  # size heuristic shown on the chart: highlight grew onto a hand
KEY_HALF, WALLET_HALF = (42, 26), (44, 32)  # half-size of your object's real outline (+ a few px margin)

# ---------------------------------------------------------------- scenes
# Each frame's truth: A = your object's centre, B = other wallet's centre (or None),
# hidden = no part of your object can be seen, where = what is really happening (plain words).

TABLE, HAND = (60, 110, 150), (120, 160, 200)


def table():
    img = np.full((SIZE, SIZE, 3), TABLE, np.uint8)
    noise = np.zeros_like(img)
    cv2.randu(noise, 0, 25)
    return cv2.add(img, noise)


def wallet(img, c):
    x, y = c
    cv2.rectangle(img, (x - 40, y - 28), (x + 40, y + 28), (40, 70, 120), -1)
    cv2.rectangle(img, (x - 34, y - 22), (x + 34, y + 22), (90, 140, 200), 2)
    cv2.circle(img, (x + 25, y), 6, (200, 200, 200), -1)


def wallet_box(c):
    return np.array([c[0] - 42, c[1] - 30, c[0] + 42, c[1] + 30])


def occlusion(n=30):
    """Your key moves right; from frame 16 a grey 'hand' covers most of it."""
    frames = []
    for i in range(n):
        img = np.full((SIZE, SIZE, 3), 40, np.uint8)
        x = 100 + i * 8
        cv2.ellipse(img, (x, 256), (40, 18), 20, 0, 360, (0, 200, 255), -1)
        covered = i > n // 2
        if covered:
            cv2.rectangle(img, (x - 10, 200), (x + 80, 320), (150, 150, 150), -1)
        where = "a hand covers most of your key (the tip still shows)" if covered else "your key slides across the table"
        frames.append((img, {"obj": "key", "A": (x, 256), "A_half": KEY_HALF, "hidden": False, "where": where}))
    return frames, np.array([50, 220, 150, 290])


def placed(drop, hand_covers_a=False, n=40):
    """Your wallet sits still. Frames 10-24 a hand brings in the other wallet; it leaves by frame 35."""
    a = (180, 300)
    frames = []
    for i in range(n):
        img = table()
        wallet(img, a)
        b, hidden = None, hand_covers_a and 18 <= i < 25
        if 10 <= i < 25:
            hx = int(560 + (drop[0] - 560) * (i - 10) / 15)
            b = (hx, drop[1])
            wallet(img, b)
            left = a[0] - 60 if hidden else hx - 30
            cv2.rectangle(img, (left, 220), (hx + 90, 330), HAND, -1)
            where = "the hand completely covers your wallet" if hidden else "a hand brings in another, similar wallet"
        elif i >= 25:
            b = drop
            wallet(img, b)
            if i < 35:
                hx = drop[0] + (i - 25) * 25
                cv2.rectangle(img, (hx + 20, 220), (hx + 150, 260), HAND, -1)
            where = "the other wallet now lies " + ("touching yours" if drop[0] - a[0] < 90 else "next to yours")
        else:
            where = "your wallet lies on the table"
        frames.append((img, {"obj": "wallet", "A": a, "A_half": WALLET_HALF, "B": b, "B_hidden": 10 <= i < 25,
                             "hidden": hidden, "where": where}))
    return frames, wallet_box(a)


def taken(visible_in_hand, n=45):
    """Other wallet already next to yours. Frames 15-19 a hand covers yours, then carries it off top-left."""
    a, b = (180, 300), (330, 300)

    def hand(img, x, y):
        cv2.rectangle(img, (x - 55, y - 45), (x + 55, y + 45), HAND, -1)
        cv2.rectangle(img, (x - 20, y - 45), (x + 20, 0), HAND, -1)

    frames = []
    for i in range(n):
        img = table()
        wallet(img, b)
        pos = a
        if i < 15:
            wallet(img, a)
            where = "your wallet and the other wallet lie on the table"
        elif i < 20:
            wallet(img, a)
            hand(img, *a)
            where = "a hand covers your wallet and grabs it"
        else:
            t = (i - 20) / 15
            hx, hy = int(a[0] - 220 * t), int(a[1] - 250 * t)
            pos = (hx, hy + 50) if visible_in_hand else (hx, hy)
            if visible_in_hand:
                wallet(img, pos)
            hand(img, hx, hy)
            where = ("the hand carries your wallet away (part of it sticks out)" if visible_in_hand
                     else "the hand carries your wallet away inside its fist")
        off_frame = pos[0] + 40 <= 0
        if off_frame:
            where = "your wallet has been carried out of view"
        hidden = 15 <= i < 20 or (i >= 20 and (not visible_in_hand or off_frame))
        frames.append((img, {"obj": "wallet", "A": pos, "A_half": WALLET_HALF, "B": b, "hidden": hidden, "where": where}))
    return frames, wallet_box(a)


SCENARIOS = {  # name -> (what happens, what the tracker should do, scene)
    "occlusion": ("Your key slides along and a hand covers most of it.",
                  "Keep highlighting the visible tip of your key, never the hand.", occlusion),
    "placed_nearby": ("Someone puts a similar wallet next to yours.",
                      "Keep highlighting your wallet only; ignore the other one.", lambda: placed((330, 300))),
    "placed_touching": ("Someone puts a similar wallet touching yours.",
                        "Highlight your wallet only, without spilling onto the other one.", lambda: placed((262, 300))),
    "hand_covers_a": ("While putting down the other wallet, the hand covers yours completely.",
                      "Highlight nothing while yours is covered, then your wallet again (not the other one).",
                      lambda: placed((330, 300), hand_covers_a=True)),
    "taken_visible": ("Someone picks up your wallet (part of it sticks out of the hand) and walks off. The other wallet stays.",
                      "Highlight only the part of your wallet that sticks out; nothing once it is out of view. "
                      "Never the other wallet.", lambda: taken(True)),
    "taken_hidden": ("Someone picks up your wallet, hidden inside the fist, and walks off. The other wallet stays.",
                     "Highlight nothing while your wallet is covered or inside the fist (it can't be seen). "
                     "Never the other wallet.", lambda: taken(False)),
}

# ---------------------------------------------------------------- tracking + grading


def track(predictor, frames, box):
    """Run EdgeTAM from a box on frame 0. Returns per-frame masks and visibility score logits."""
    with tempfile.TemporaryDirectory() as d:
        for i, (img, _) in enumerate(frames):
            cv2.imwrite(f"{d}/{i:05d}.jpg", img)
        with torch.inference_mode():
            state = predictor.init_state(video_path=d)
            predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1, box=box)
            masks = {fi: (logits[0, 0] > 0).cpu().numpy() for fi, _, logits in predictor.propagate_in_video(state)}
            outs = state["output_dict_per_obj"][0]
            scores = []
            for fi in range(len(frames)):
                out = outs["cond_frame_outputs"].get(fi) or outs["non_cond_frame_outputs"][fi]
                scores.append(float(out["object_score_logits"].flatten()[0]))
    return [masks[i] for i in range(len(frames))], scores


def inside(mask, p):
    x, y = p
    return 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and bool(mask[y, x])


def expected(truth):
    return "nothing" if truth["hidden"] else "yours"


def highlighted(mask, truth):
    """What the tracker's mask is actually on: yours | hand | nothing | other | both."""
    area = int(mask.sum())
    if area == 0:
        return "nothing"
    ys, xs = np.nonzero(mask)
    c = np.array([xs.mean(), ys.mean()])
    a, b = np.array(truth["A"]), truth.get("B")
    if b is not None and inside(mask, b) and inside(mask, truth["A"]):
        return "both"
    if b is not None and np.linalg.norm(c - b) < np.linalg.norm(c - a):
        return "other"
    if truth["hidden"]:
        return "hand"
    (x, y), (hw, hh) = truth["A"], truth["A_half"]
    inside_a = int(mask[max(y - hh, 0) : max(y + hh + 1, 0), max(x - hw, 0) : max(x + hw + 1, 0)].sum())
    return "yours" if inside_a >= 0.7 * area else "hand"


# ---------------------------------------------------------------- rendering (dark theme)


def bgr(hex_):
    h = hex_.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (4, 2, 0))


def css(color):
    return "#%02x%02x%02x" % color[::-1]


SURFACE, PLANE = bgr("#1a1a19"), bgr("#0d0d0d")
INK, INK2, MUTED = bgr("#ffffff"), bgr("#c3c2b7"), bgr("#898781")
GRID, AXIS = bgr("#2c2c2a"), bgr("#383835")
SERIES = bgr("#3987e5")  # the tracker's highlight
GOOD, WARN, BAD = bgr("#0ca30c"), bgr("#fab219"), bgr("#d03b3b")
# what is (or should be) highlighted -> (colour, words); colour always comes with its words
TARGET = {
    "yours": (GOOD, "your {obj}"),
    "nothing": (INK2, "nothing (it can't be seen)"),
    "hand": (WARN, "the hand"),
    "other": (BAD, "the OTHER wallet"),
    "both": (BAD, "both wallets"),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def words(key, obj):
    return TARGET[key][1].format(obj=obj)


def text(img, s, org, color=INK, scale=0.5, thick=1):
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def pill(img, s, org, fill, ink=PLANE, scale=0.5, thick=1):
    """Text on a filled rounded-ish label; returns the x just after it."""
    (w, h), _ = cv2.getTextSize(s, FONT, scale, thick)
    x, y = org
    cv2.rectangle(img, (x, y - h - 7), (x + w + 14, y + 7), fill, -1, cv2.LINE_AA)
    text(img, s, (x + 7, y), ink, scale, thick)
    return x + w + 14


def dashed_rect(img, p0, p1, color, thick=2, dash=8):
    (x0, y0), (x1, y1) = p0, p1
    for xa in range(x0, x1, dash * 2):
        cv2.line(img, (xa, y0), (min(xa + dash, x1), y0), color, thick)
        cv2.line(img, (xa, y1), (min(xa + dash, x1), y1), color, thick)
    for ya in range(y0, y1, dash * 2):
        cv2.line(img, (x0, ya), (x0, min(ya + dash, y1)), color, thick)
        cv2.line(img, (x1, ya), (x1, min(ya + dash, y1)), color, thick)


def labelled_outline(img, centre, half, label, color, hidden):
    (x, y), (hw, hh) = centre, half
    if x + hw < 0 or y + hh < 0:
        return  # fully out of view
    p0, p1 = (x - hw, y - hh), (x + hw, y + hh)
    if hidden:
        dashed_rect(img, p0, p1, color)
    else:
        cv2.rectangle(img, p0, p1, color, 2, cv2.LINE_AA)
    pill(img, label, (max(x - hw, 4), max(y - hh - 10, 18)), color, PLANE, 0.42)


def overlay(img, mask, truth):
    """The tracker's highlight in blue, plus labelled real outlines of your object and the other wallet."""
    out = img.copy()
    if mask.any():
        tint = out.copy()
        tint[mask] = SERIES
        out = cv2.addWeighted(tint, 0.5, out, 0.5, 0)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, SERIES, 2, cv2.LINE_AA)
    if truth.get("B") is not None:
        b_hidden = truth.get("B_hidden", False)
        label = "OTHER WALLET" + (" (under the hand)" if b_hidden else "")
        labelled_outline(out, truth["B"], WALLET_HALF, label, MUTED, b_hidden)
    mine = f"YOUR {truth['obj'].upper()}" + (" (hidden here)" if truth["hidden"] else "")
    labelled_outline(out, truth["A"], truth["A_half"], mine, INK, truth["hidden"])
    return out


def caption(canvas, y, width, fi, truth, got):
    """Two lines: what is really happening, then should-highlight vs tracker-highlights -> CORRECT/WRONG."""
    cv2.rectangle(canvas, (0, y), (width, y + 64), SURFACE, -1)
    text(canvas, f"Frame {fi:2d}   Really happening: {truth['where']}", (14, y + 24), INK, 0.55)
    exp, obj = expected(truth), truth["obj"]
    x = 14
    text(canvas, "Should highlight:", (x, y + 52), INK2, 0.5)
    x = pill(canvas, words(exp, obj), (x + 150, y + 52), TARGET[exp][0], PLANE, 0.5) + 24
    text(canvas, "Tracker highlights:", (x, y + 52), INK2, 0.5)
    x = pill(canvas, words(got, obj), (x + 168, y + 52), TARGET[got][0], PLANE, 0.5) + 24
    ok = got == exp
    pill(canvas, "CORRECT" if ok else "WRONG", (x, y + 52), GOOD if ok else BAD, INK if not ok else PLANE, 0.55, 2)


def row_strip(img, x0, y0, w, label, keys, cur):
    text(img, label, (x0 + 10, y0 + 17), INK2, 0.48)
    cx0, cx1, n = x0 + 190, x0 + w - 14, len(keys)
    cw = (cx1 - cx0) / n
    for i, k in enumerate(keys):
        color = TARGET[k][0] if i <= cur else GRID
        cv2.rectangle(img, (int(cx0 + i * cw) + 1, y0 + 4), (int(cx0 + (i + 1) * cw) - 1, y0 + 22), color, -1)
    cv2.line(img, (int(cx0 + (cur + 0.5) * cw), y0), (int(cx0 + (cur + 0.5) * cw), y0 + 26), INK, 1)


def legend(img, x0, y0, keys, obj):
    x = x0 + 10
    for k in dict.fromkeys(keys):
        cv2.circle(img, (x + 6, y0), 6, TARGET[k][0], -1, cv2.LINE_AA)
        label = words(k, obj)
        text(img, label, (x + 18, y0 + 5), INK2, 0.42)
        x += 40 + cv2.getTextSize(label, FONT, 0.42, 1)[0][0]


def line_chart(img, x0, y0, w, h, values, cur, title, ref=None, ref_label="", zero=False):
    """Single-series line chart drawn up to frame `cur`, with fixed axes for the whole clip."""
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), SURFACE, -1)
    text(img, title, (x0 + 10, y0 + 20), INK2, 0.48)
    px0, px1, py0, py1 = x0 + 56, x0 + w - 14, y0 + 34, y0 + h - 26
    lo, hi = min(values + [ref or 0, 0 if zero else min(values)]), max(values + [ref or 0])
    pad = (hi - lo) * 0.08 or 1
    lo, hi = lo - pad, hi + pad
    sx = lambda i: int(px0 + (px1 - px0) * i / max(len(values) - 1, 1))
    sy = lambda v: int(py1 - (py1 - py0) * (v - lo) / (hi - lo))
    for v in (lo + pad, hi - pad):
        cv2.line(img, (px0, sy(v)), (px1, sy(v)), GRID, 1)
        if zero and abs(sy(v) - sy(0)) < 14:
            continue
        text(img, f"{v:,.0f}" if abs(v) >= 100 or v == 0 else f"{v:.1f}", (x0 + 6, sy(v) + 4), MUTED, 0.38)
    if zero:
        cv2.line(img, (px0, sy(0)), (px1, sy(0)), AXIS, 1)
        text(img, "0", (x0 + 6, sy(0) + 4), MUTED, 0.38)
    if ref is not None:
        for xx in range(px0, px1, 10):
            cv2.line(img, (xx, sy(ref)), (min(xx + 5, px1), sy(ref)), MUTED, 1)
        text(img, ref_label, (px1 - 7 * len(ref_label) - 4, sy(ref) - 6), MUTED, 0.38)
    cv2.line(img, (px0, py1), (px1, py1), AXIS, 1)
    text(img, "frame", (px1 - 40, y0 + h - 8), MUTED, 0.38)
    for i in range(0, len(values) - 5, 10):  # leave room for the "frame" axis label
        text(img, str(i), (sx(i) - 4, y0 + h - 8), MUTED, 0.38)
    pts = np.array([(sx(i), sy(v)) for i, v in enumerate(values[: cur + 1])], np.int32)
    cv2.polylines(img, [pts], False, SERIES, 2, cv2.LINE_AA)
    cv2.line(img, (sx(cur), py0), (sx(cur), py1), AXIS, 1)
    cv2.circle(img, (sx(cur), sy(values[cur])), 5, SURFACE, -1, cv2.LINE_AA)
    cv2.circle(img, (sx(cur), sy(values[cur])), 4, SERIES, -1, cv2.LINE_AA)


def bottom_panel(width, run, cur):
    frames, scores, areas, exp, got = run["frames"], run["scores"], run["areas"], run["exp"], run["got"]
    obj = frames[0][1]["obj"]
    panel = np.full((330, width, 3), PLANE, np.uint8)
    row_strip(panel, 0, 6, width, "Should highlight", exp, cur)
    row_strip(panel, 0, 36, width, "Tracker highlighted", got, cur)
    legend(panel, 0, 80, exp + got, obj)
    half = width // 2
    line_chart(panel, 6, 100, half - 9, 224, areas, cur, "Size of the blue highlight (pixels)",
               ref=DRIFT * areas[0], ref_label=f"{DRIFT}x starting size")
    line_chart(panel, half + 3, 100, half - 9, 224, scores, cur,
               f"Tracker's confidence it can see your {obj} (above 0 = 'I see it')", zero=True)
    return panel


def render(run, fi):
    img, truth = run["frames"][fi]
    width = 2 * SIZE
    head = np.full((40, width, 3), PLANE, np.uint8)
    text(head, run["title"], (12, 27), INK, 0.6)
    cap = np.full((64, width, 3), PLANE, np.uint8)
    caption(cap, 0, width, fi, truth, run["got"][fi])
    left, right = img.copy(), overlay(img, run["masks"][fi], truth)
    pill(left, "What the camera sees", (8, 26), SURFACE, INK, 0.5)
    pill(right, "Blue = what the tracker thinks is your " + truth["obj"], (8, 26), SURFACE, INK, 0.5)
    return np.vstack([head, cap, np.hstack([left, right]), bottom_panel(width, run, fi)])


def key_frames(run, k=5):
    """Middle frame of each stretch where (what's happening, result) stays the same, at most k of them."""
    segs, start = [], 0
    keys = [(t["where"], g) for (_, t), g in zip(run["frames"], run["got"])]
    for i in range(1, len(keys) + 1):
        if i == len(keys) or keys[i] != keys[start]:
            segs.append((start + i - 1) // 2)
            start = i
    if len(segs) > k:
        segs = [segs[j] for j in np.linspace(0, len(segs) - 1, k).astype(int)]
    return segs


def contact_sheet(run, thumb=300):
    picks = key_frames(run)
    width = thumb * max(len(picks), 4)
    head = np.full((40, width, 3), PLANE, np.uint8)
    text(head, run["title"], (12, 27), INK, 0.6)
    tiles = []
    for fi in picks:
        img, truth = run["frames"][fi]
        tile = np.full((thumb + 92, thumb, 3), PLANE, np.uint8)
        tile[:thumb] = cv2.resize(overlay(img, run["masks"][fi], truth), (thumb, thumb))
        exp, got = expected(truth), run["got"][fi]
        text(tile, f"frame {fi}", (8, thumb + 20), INK, 0.45)
        text(tile, "should: " + words(exp, truth["obj"]), (8, thumb + 42), INK2, 0.4)
        text(tile, "tracker: " + words(got, truth["obj"]), (8, thumb + 62), INK2, 0.4)
        pill(tile, "CORRECT" if got == exp else "WRONG", (8, thumb + 84), GOOD if got == exp else BAD,
             PLANE if got == exp else INK, 0.42)
        tiles.append(tile)
    row = np.full((thumb + 92, width, 3), PLANE, np.uint8)
    row[:, : thumb * len(tiles)] = np.hstack(tiles)
    return np.vstack([head, row, bottom_panel(width, run, len(run["frames"]) - 1)])


def write_index(results):
    """One local page: plain-language summary table, then each scenario's video."""
    rows, cards = [], []
    for name, r in results.items():
        n, ok = len(r["got"]), sum(g == e for g, e in zip(r["got"], r["exp"]))
        obj = r["frames"][0][1]["obj"]
        wrong = {}
        for g, e in zip(r["got"], r["exp"]):
            if g != e:
                wrong[g] = wrong.get(g, 0) + 1
        mistakes = ", ".join(f"highlighted {words(g, obj)} in {c} frames" for g, c in wrong.items()) or "none"
        swapped = sum(g in ("other", "both") for g in r["got"])
        rows.append(f"<tr><td><a href='#{name}'>{name}</a></td><td>{html.escape(r['what'])}</td>"
                    f"<td><b>{ok} / {n}</b></td><td>{'never' if not swapped else f'{swapped} frames'}</td>"
                    f"<td>{html.escape(mistakes)}</td></tr>")
        cards.append(f"<section id='{name}'><h2>{name}</h2>"
                     f"<p><b>What happens:</b> {html.escape(r['what'])}<br><b>The tracker should:</b> {html.escape(r['should'])}<br>"
                     f"<b>Result:</b> correct in {ok} of {n} frames. Mistakes: {html.escape(mistakes)}.</p>"
                     f"<video src='{name}.mp4' controls loop muted autoplay playsinline></video></section>")
    chip = lambda c, s: f"<span class='chip'><i style='background:{css(c)}'></i>{s}</span>"
    (OUT / "index.html").write_text(f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Synthetic Tracking Results</title>
<style>
:root{{--plane:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e1e0d9}}
@media (prefers-color-scheme:dark){{:root{{--plane:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--line:#2c2c2a}}}}
body{{margin:0;padding:24px 16px;background:var(--plane);color:var(--ink);font:15px/1.55 system-ui,sans-serif}}
main{{max-width:1100px;margin:auto}} h1{{font-size:22px;margin:0 0 8px}} h2{{font-size:17px;margin:0 0 6px}}
p,li{{color:var(--ink2)}} b{{color:var(--ink)}} a{{color:inherit}}
section,.box{{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px;margin:16px 0}}
video{{width:100%;border-radius:6px;margin-top:10px;background:#000}} ul{{padding-left:18px;margin:6px 0}}
.wrap{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-size:14px}}
td,th{{text-align:left;padding:8px;border-bottom:1px solid var(--line);vertical-align:top}} th{{color:var(--ink2);font-weight:600}}
.chip{{display:inline-flex;align-items:center;gap:6px}} .chip i{{width:10px;height:10px;border-radius:50%;display:inline-block}}
</style></head><body><main>
<h1>Can the tracker keep track of your wallet?</h1>
<p>These are drawn (cartoon) scenes, so we know the real answer for every frame. The tracker is shown your object
once, on the first frame, and then has to follow it on its own.</p>
<div class="box"><b>How to read each video</b><ul>
<li><b>Left:</b> what the camera sees. <b>Right:</b> the same picture with the tracker's guess in <b>blue</b>.</li>
<li><b>White box "YOUR WALLET":</b> where your wallet really is. Dashed = it's hidden there (under or inside a hand).</li>
<li><b>Grey box "OTHER WALLET":</b> someone else's similar-looking wallet.</li>
<li><b>Top line:</b> what is really happening, what the tracker <i>should</i> highlight, what it <i>did</i> highlight,
and <b>CORRECT / WRONG</b>.</li>
<li><b>Two coloured rows:</b> "should" and "did" for every frame. Where the colours differ, the tracker made a mistake.
{chip(GOOD, "your wallet")} {chip(INK2, "nothing")} {chip(WARN, "the hand")} {chip(BAD, "the other wallet")}</li>
<li><b>Charts:</b> size of the blue area (a jump = it grabbed the hand too), and how sure the tracker is that it can see
your wallet (above 0 = "I see it").</li></ul></div>
<div class="box wrap"><table><tr><th>Test</th><th>What happens</th><th>Frames correct</th><th>Picked the other wallet?</th><th>Mistakes</th></tr>
{"".join(rows)}</table></div>
<div class="box"><b>Why "the hand" counts as a mistake:</b> following the hand does show where your wallet went, which
is useful. But the tracker should say "I can't see the wallet" rather than claim the hand <i>is</i> your wallet,
because otherwise the system can't tell "wallet on the table" from "wallet in someone's fist". The planned fix is a
second check: "does the blue area still look like my registered wallet?"</div>
{"".join(cards)}
</main></body></html>""")


# ---------------------------------------------------------------- main


def main():
    names = sys.argv[1:] or list(SCENARIOS)
    OUT.mkdir(parents=True, exist_ok=True)
    patch_attention_for_mps()
    from sam2.build_sam import build_sam2_video_predictor

    predictor = build_sam2_video_predictor(
        "configs/edgetam.yaml", str(ROOT / "third_party/EdgeTAM/checkpoints/edgetam.pt"), device=torch.device("mps")
    )
    cv2.setRNGSeed(0)
    results = {}
    for name in names:
        what, should, make = SCENARIOS[name]
        frames, box = make()
        masks, scores = track(predictor, frames, box)
        run = {
            "title": f"{name}: {what}", "what": what, "should": should, "frames": frames, "masks": masks,
            "scores": scores, "areas": [int(m.sum()) for m in masks],
            "exp": [expected(t) for _, t in frames], "got": [highlighted(m, t) for m, (_, t) in zip(masks, frames)],
        }
        first = render(run, 0)
        writer = cv2.VideoWriter(str(OUT / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"avc1"), 5, first.shape[1::-1])
        for fi in range(len(frames)):
            writer.write(render(run, fi))
        writer.release()
        cv2.imwrite(str(OUT / f"{name}.png"), contact_sheet(run))
        results[name] = run
        ok = sum(g == e for g, e in zip(run["got"], run["exp"]))
        print(f"{name:16s} correct {ok}/{len(frames)}  tracker highlighted: "
              + ", ".join(f"{k} {run['got'].count(k)}" for k in dict.fromkeys(run["got"])))
    write_index(results)
    print(f"\nwrote videos, key-frame images and index.html to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
