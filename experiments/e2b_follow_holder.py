"""E2b: follow the holder. When a hand takes the object, watch the hand; when the hand lets go, find the object.

Builds on E2's variant B (the "does it still look like my object?" check on the output). Variant D adds a
per-frame state and a hand-off rule:

  VISIBLE      the tracker's mask passes the check                         -> show it
  COVERED      the mask vanished without growing first (something on top)   -> "covered, last seen here"
  HELD         the mask grew onto something bigger (a hand took the object) -> "in a hand", outline the hand
  OUT_OF_VIEW  the hand left the picture while holding it                   -> "carried out of view to the <side>"
  UNKNOWN      the hand was lost mid-picture for more than 15 frames        -> "lost track"

Hand-off (HELD -> VISIBLE), checked every HELD frame:
  1. find object-looking things: peaks of DINOv2 patch similarity to the registered object (frame 0), each
     outlined separately by SAM from a point on the peak (no training)
  2. keep those where the hand is or was in its last 6 frames (+40 px)
  3. drop lookalikes already lying elsewhere: seen at registration, or when the hand took the object
  4. confirm with the E2 appearance check and restart the tracker on it
A visible object that disappears at the picture edge is OUT_OF_VIEW, not COVERED.

v1 drew one thresholded region around everything object-like; two wallets 70 px apart merged into one, the
other wallet was not recorded as "already elsewhere", and the hand-off picked it (25 wrong frames).

The "hand" is followed with EdgeTAM's own drift onto it (what version A does by accident). On real video
a hand detector (e.g. MediaPipe) would be added; the drawn hands here are plain rectangles.

Pass criteria: no regressions vs B, 0 frames on the other wallet, and the system's final answer
("visible at ..." / "carried out of view to the left") matches what really happened.

Run from repo root (after experiments/e2_gated_memory.py, whose A/B results it compares against):
  python experiments/e2b_follow_holder.py
Writes experiments/output/e2b/: <scenario>.mp4 / .png, index.html (open it), e2b_results.json
"""
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "third_party" / "EdgeTAM"), str(ROOT / "experiments")]

import cv2
import numpy as np
import torch

import synthetic_tracking as st
from compass.appearance import Appearance
from compass.holder import Finder, ObjectState, Segmenter
from compass.models import video_predictor
from compass.tracker import StreamingTracker
from compass.video_io import make_web_playable

OUT = ROOT / "experiments" / "output" / "e2b"
E2_RESULTS = ROOT / "experiments" / "output" / "e2" / "e2_results.json"

# What the system should be saying at the end of each scene: (state, detail)
EXPECTED_END = {
    "occlusion": ("visible", None), "placed_nearby": ("visible", None), "placed_touching": ("visible", None),
    "hand_covers_a": ("visible", None), "taken_visible": ("out_of_view", "left"),
    "taken_hidden": ("out_of_view", "left"), "moved": ("visible", None), "moved_alone": ("visible", None),
}


# The tracker, appearance check and holder logic live in the compass package (compass/holder.py), shared with
# the live app; this experiment runs them on the drawn scenes, one frame at a time, exactly as live.


def follow(obj, frames, box):
    """Run compass.holder.ObjectState over a drawn scene. Returns per-frame outputs."""
    steps = [obj.start(frames[0][0], box)] + [obj.update(img) for img, _ in frames[1:]]
    last = steps[-1]
    return {"outs": [s.mask for s in steps], "holders": [s.holder for s in steps], "scores": [s.score for s in steps],
            "states": [s.state if s.state != "out_of_view" else f"out_of_view:{s.detail}" for s in steps],
            "events": obj.events, "final": (last.state, last.detail), "lookalikes_at_start": len(obj.lookalikes0)}


# ---------------------------------------------------------------- rendering


STATE_TEXT = {
    "visible": "System: your {obj} is visible",
    "covered": "System: covered - last seen here",
    "held": "System: in a hand (yellow) - following the hand",
    "unknown": "System: lost track of the hand",
    "unsure": "System: not sure - something here looks different",
}


def state_text(s, obj):
    if s.startswith("out_of_view:"):
        return f"System: carried out of view to the {s.split(':')[1]}"
    return STATE_TEXT[s].format(obj=obj)


def render(run, res, fi):
    canvas = st.render(run, fi)
    x0, y0 = st.SIZE, 40 + 64  # right panel origin inside st.render's layout
    holder = res["holders"][fi]
    if holder is not None:
        contours, _ = cv2.findContours(holder.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas[y0:y0 + st.SIZE, x0:x0 + st.SIZE], contours, -1, st.WARN, 3, cv2.LINE_AA)
    st.pill(canvas, state_text(res["states"][fi], run["frames"][0][1]["obj"]), (x0 + 8, y0 + st.SIZE - 16),
            st.SURFACE, st.INK, 0.5)
    return canvas


def final_answer(res, frames):
    state, detail = res["final"]
    if state == "visible":
        ys, xs = np.nonzero(res["outs"][-1])
        return f"your {frames[0][1]['obj']} is visible at ({xs.mean():.0f}, {ys.mean():.0f})" if len(xs) else "visible"
    return {"covered": "covered, last seen here", "held": "in a hand", "unknown": "lost track", "unsure": "not sure",
            "out_of_view": f"carried out of view to the {detail}"}[state]


def final_ok(name, res, run):
    want_state, want_detail = EXPECTED_END[name]
    state, detail = res["final"]
    if want_state == "visible":
        return state == "visible" and run["got"][-1] == "yours"
    return state == want_state and detail == want_detail


def write_index(rows, cards):
    (OUT / "index.html").write_text(f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>E2b Follow The Hand</title>
<style>
:root{{--plane:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e1e0d9;--bad:#d03b3b;--good:#006300}}
@media (prefers-color-scheme:dark){{:root{{--plane:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--line:#2c2c2a;--bad:#e66767;--good:#0ca30c}}}}
body{{margin:0;padding:24px 16px;background:var(--plane);color:var(--ink);font:15px/1.55 system-ui,sans-serif}}
main{{max-width:1200px;margin:auto}} h1{{font-size:22px;margin:0 0 8px}} h2{{font-size:17px;margin:0 0 6px}}
p,li,figcaption{{color:var(--ink2)}} b{{color:var(--ink)}} a{{color:inherit}} .bad{{color:var(--bad)}} .good{{color:var(--good)}}
section,.box{{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px;margin:16px 0}}
.wrap{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-size:14px}}
td,th{{text-align:left;padding:8px;border-bottom:1px solid var(--line);vertical-align:top}} th{{color:var(--ink2);font-weight:600}}
.vids{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}} figure{{margin:0}}
video{{width:100%;border-radius:6px;background:#000}} figcaption{{font-size:13px;margin-bottom:4px}} ol{{margin:6px 0}}
</style></head><body><main>
<h1>E2b: follow the hand that took the object</h1>
<p>Same drawn scenes as E2. <b>A</b> = tracker alone, <b>B</b> = + "does it still look like my object?" check (chosen in
E2), <b>D</b> = B + follow the hand: when a hand takes the object, watch the hand (yellow outline); when it lets go of
something that looks like the object near where it is or was, that is the object. Objects that were already lying
elsewhere when the hand took it (the other wallet) are ignored.</p>
<div class="box wrap"><table><tr><th>Scenario</th><th>A frames correct</th><th>B frames correct</th>
<th>D frames correct</th><th>D: system's final answer</th></tr>{''.join(rows)}</table></div>
{''.join(cards)}
</main></body></html>""")


# ---------------------------------------------------------------- main


def main():
    if not E2_RESULTS.exists():
        sys.exit("run experiments/e2_gated_memory.py first (E2b compares against its A and B results)")
    e2 = {r["scenario"]: r for r in json.loads(E2_RESULTS.read_text())["table"]}
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps")
    obj = ObjectState(StreamingTracker(video_predictor(device)), Appearance(device), Finder(device), Segmenter(device),
                      spot_tests="--no-spot-tests" not in sys.argv)

    rows, cards, summary = [], [], []
    for name, (what, should, make) in st.SCENARIOS.items():
        cv2.setRNGSeed(0)
        frames, box = make()
        res = follow(obj, frames, box)
        exp = [st.expected(t) for _, t in frames]
        got = [st.highlighted(m, t) for m, (_, t) in zip(res["outs"], frames)]
        run = {"title": f"{name}, D: + follow the hand", "what": what, "should": should, "frames": frames,
               "masks": res["outs"], "scores": res["scores"], "areas": [int(m.sum()) for m in res["outs"]],
               "exp": exp, "got": got}
        correct = sum(g == e for g, e in zip(got, exp))
        other = got.count("other") + got.count("both")
        answer, ok = final_answer(res, frames), final_ok(name, res, run)
        a, b = e2[name]["A"]["correct"], e2[name]["B"]["correct"]
        summary.append({"scenario": name, "frames": len(frames), "A": a, "B": b, "D": correct, "D_other_wallet_frames": other,
                        "final_answer": answer, "final_answer_correct": ok, "expected_end": EXPECTED_END[name],
                        "events": res["events"], "lookalikes_at_registration": res["lookalikes_at_start"]})
        print(f"{name:16s} correct A {a:2d}  B {b:2d}  D {correct:2d} / {len(frames)} | other-wallet frames {other} | "
              f"final: {answer} [{'OK' if ok else 'WRONG'}] | lookalikes at registration {res['lookalikes_at_start']}")
        for fi, ev in res["events"]:
            print(f"      frame {fi:2d}: {ev}")

        size = render(run, res, 0).shape[1::-1]
        writer = cv2.VideoWriter(str(OUT / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"avc1"), 5, size)
        for fi in range(len(frames)):
            writer.write(render(run, res, fi))
        writer.release()
        make_web_playable(OUT / f"{name}.mp4")
        cv2.imwrite(str(OUT / f"{name}.png"), st.contact_sheet(run))

        warn = f"<br><span class='bad'>{other} frames on the other wallet</span>" if other else ""
        verdict = "<span class='good'>correct</span>" if ok else "<span class='bad'>wrong</span>"
        rows.append(f"<tr><td><a href='#{name}'>{name}</a></td><td>{a} / {len(frames)}</td><td>{b} / {len(frames)}</td>"
                    f"<td><b>{correct}</b> / {len(frames)}{warn}</td><td>{html.escape(answer)} ({verdict})</td></tr>")
        evs = "".join(f"<li>frame {fi}: {html.escape(ev)}</li>" for fi, ev in res["events"]) or "<li>no state changes</li>"
        vids = "".join(f"<figure><figcaption>{cap}</figcaption><video src='{src}' controls loop muted playsinline "
                       f"preload='metadata'></video></figure>" for src, cap in (
                           (f"../e2/A/{name}.mp4", "A: tracker alone"), (f"../e2/B/{name}.mp4", "B: + check"),
                           (f"{name}.mp4", "D: + follow the hand")))
        cards.append(f"<section id='{name}'><h2>{name}</h2><p><b>What happens:</b> {html.escape(what)}<br>"
                     f"<b>What D noticed:</b></p><ol>{evs}</ol><div class='vids'>{vids}</div></section>")

    tot = {k: sum(s[k] for s in summary) for k in ("frames", "A", "B", "D", "D_other_wallet_frames")}
    finals = sum(s["final_answer_correct"] for s in summary)
    rows.append(f"<tr><td><b>All {len(summary)} scenes</b></td><td>{tot['A']} / {tot['frames']}</td>"
                f"<td>{tot['B']} / {tot['frames']}</td><td><b>{tot['D']}</b> / {tot['frames']} "
                f"({tot['D_other_wallet_frames']} on the other wallet)</td><td>{finals} / {len(summary)} correct</td></tr>")
    write_index(rows, cards)
    (OUT / "e2b_results.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"\ntotal correct: A {tot['A']}/{tot['frames']}  B {tot['B']}/{tot['frames']}  D {tot['D']}/{tot['frames']} | "
          f"D other-wallet frames {tot['D_other_wallet_frames']} | final answers correct "
          f"{finals}/{len(summary)}")
    print(f"wrote {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
