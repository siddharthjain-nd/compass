"""E2: gated memory. Does refusing to remember frames that don't look like the object stop drift onto hands?

Hypothesis (RESEARCH.md §4.2): the tracker drifts onto the hand because frames where its mask covers the
hand get written into its memory and then define "the object". If such frames are kept out of memory, the
tracker should report "not visible" while the object is held, and find the object again when it reappears.

Check per frame, using only signals a real system has (no ground truth). "Typical" values come from
recently accepted frames (area) and the first 5 frames (DINOv2 ViT-S/14 similarity of the masked crop to
the frame-0 crop):
  grown    area > 1.5x typical                       -> reject (it has spread onto a holder)
  partial  area < 0.7x typical (object partly hidden) -> show if similarity >= 0.75x typical,
                                                        but never write it to memory
  normal                                              -> show and remember if similarity >= 0.85x typical
An empty mask is "not visible" and is not checked.
v1 used the 0.85 rule for partial views too and wrongly rejected a partly covered key (0.79-0.83).
The 0.75 partial threshold was chosen after seeing those values; real clips must confirm it.

Variants (same tracker, same check):
  A  baseline        no check
  B  output check    a failing frame is reported as "held / covered" (empty), but still enters memory
  C  gated memory    like B, and the failing frame is also removed from the tracker's memory

Scenarios are the synthetic ones from synthetic_tracking.py, with ground truth on every frame.
Run from repo root:  python experiments/e2_gated_memory.py
Writes experiments/output/e2/: {A,B,C}/<scenario>.mp4 / .png, index.html (open it), e2_results.json
"""
import html
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "third_party" / "EdgeTAM"), str(ROOT / "experiments")]

import cv2
import numpy as np
import torch

import synthetic_tracking as st
from check_env import patch_attention_for_mps
from compass.video_io import make_web_playable
from compass.appearance import AREA_GROWTH, AREA_PARTIAL, SIM_NORMAL, SIM_PARTIAL, Appearance, Gate

OUT = ROOT / "experiments" / "output" / "e2"
VARIANTS = {
    "A": "A: tracker alone",
    "B": "B: + 'still looks like my object?' check on the output",
    "C": "C: + the check also keeps failing frames out of memory",
}
# The check itself (Appearance, Gate and its thresholds) lives in compass/appearance.py, shared with the live app.


def run(predictor, appearance, frames, box, variant):
    """Track one object with EdgeTAM; apply variant A/B/C. Returns output masks, raw masks, visibility
    scores and the check log."""
    with tempfile.TemporaryDirectory() as d:
        for i, (img, _) in enumerate(frames):
            cv2.imwrite(f"{d}/{i:05d}.jpg", img)
        with torch.inference_mode():
            state = predictor.init_state(video_path=d)
            predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1, box=box)
            gate, outs, raws, scores, log = None, [], [], [], []
            for fi, _, logits in predictor.propagate_in_video(state):
                raw = (logits[0, 0] > 0).cpu().numpy()
                od = state["output_dict"]
                cur = od["cond_frame_outputs"].get(fi) or od["non_cond_frame_outputs"][fi]
                scores.append(float(cur["object_score_logits"].flatten()[0]))
                img = frames[fi][0]
                if fi == 0:
                    gate = Gate(appearance, img, raw)
                show, remember, info = (True, True, {}) if variant == "A" or not raw.any() or fi == 0 else gate.check(img, raw)
                if not remember and variant == "C":  # keep this frame out of the tracker's memory
                    state["output_dict"]["non_cond_frame_outputs"].pop(fi, None)
                    state["output_dict_per_obj"][0]["non_cond_frame_outputs"].pop(fi, None)
                outs.append(raw if show else np.zeros_like(raw))
                raws.append(raw)
                log.append({"frame": fi, **info})
    return outs, raws, scores, log


def write_index(table, results):
    """One local page: the A/B/C table, then each scenario's three videos."""
    head = "".join(f"<th>{html.escape(VARIANTS[v])}</th>" for v in "ABC")
    rows = []
    for row in table:
        n, total = row["scenario"], row["A"]["frames"]
        cells = []
        for v in "ABC":
            other = row[v]["other_wallet_frames"]
            warn = f"<br><span class='bad'>{other} frames on the other wallet</span>" if other else ""
            cells.append(f"<td><b>{row[v]['correct']}</b> / {total}{warn}</td>")
        rows.append(f"<tr><td><a href='#{n}'>{n}</a></td>{''.join(cells)}</tr>")
    cards = []
    for n, by_variant in results.items():
        r = by_variant["A"]
        vids = "".join(f"<figure><figcaption>{html.escape(VARIANTS[v])}</figcaption>"
                       f"<video src='{v}/{n}.mp4' controls loop muted playsinline preload='metadata'></video></figure>"
                       for v in "ABC")
        cards.append(f"<section id='{n}'><h2>{n}</h2><p><b>What happens:</b> {html.escape(r['what'])}<br>"
                     f"<b>The tracker should:</b> {html.escape(r['should'])}</p><div class='vids'>{vids}</div></section>")
    (OUT / "index.html").write_text(f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>E2 Gated Memory</title>
<style>
:root{{--plane:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e1e0d9;--bad:#d03b3b}}
@media (prefers-color-scheme:dark){{:root{{--plane:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--line:#2c2c2a;--bad:#e66767}}}}
body{{margin:0;padding:24px 16px;background:var(--plane);color:var(--ink);font:15px/1.55 system-ui,sans-serif}}
main{{max-width:1200px;margin:auto}} h1{{font-size:22px;margin:0 0 8px}} h2{{font-size:17px;margin:0 0 6px}}
p,li,figcaption{{color:var(--ink2)}} b{{color:var(--ink)}} a{{color:inherit}} .bad{{color:var(--bad)}}
section,.box{{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px;margin:16px 0}}
.wrap{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-size:14px}}
td,th{{text-align:left;padding:8px;border-bottom:1px solid var(--line);vertical-align:top}} th{{color:var(--ink2);font-weight:600}}
.vids{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}} figure{{margin:0}}
video{{width:100%;border-radius:6px;background:#000}} figcaption{{font-size:13px;margin-bottom:4px}}
</style></head><body><main>
<h1>E2: can a "does it still look like my object?" check stop the tracker following the hand?</h1>
<p>Drawn scenes with known answers, as before. Each scenario is tracked three ways (A, B, C). The check compares
the tracker's blue area with what the object looked like at the start (size + DINOv2 appearance).</p>
<div class="box wrap"><table><tr><th>Scenario</th>{head}</tr>{''.join(rows)}</table></div>
<div class="box"><b>Verdict:</b> use <b>B</b>. It removes the "highlighting the hand" mistakes and never picks the
other wallet. C (also keeping bad frames out of memory) makes the tracker search again after the hand, and with a
lookalike present it can lock onto the <i>other</i> wallet. Being confidently wrong is worse than saying "can't see
it", so C waits until an identity check (E4) can confirm what it finds.</div>
{''.join(cards)}
</main></body></html>""")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    patch_attention_for_mps()
    import sam2.sam2_video_predictor as vp
    from sam2.build_sam import build_sam2_video_predictor

    vp.tqdm = lambda it, **_: it
    device = torch.device("mps")
    predictor = build_sam2_video_predictor(
        "configs/edgetam.yaml", str(ROOT / "third_party/EdgeTAM/checkpoints/edgetam.pt"), device=device)
    appearance = Appearance(device)

    results, table = {}, []
    for name, (what, should, make) in st.SCENARIOS.items():
        cv2.setRNGSeed(0)
        frames, box = make()
        exp = [st.expected(t) for _, t in frames]
        row = {"scenario": name}
        for variant in "ABC":
            outs, raws, scores, log = run(predictor, appearance, frames, box, variant)
            got = [st.highlighted(m, t) for m, (_, t) in zip(outs, frames)]
            raw_got = [st.highlighted(m, t) for m, (_, t) in zip(raws, frames)]
            row[variant] = {
                "correct": sum(g == e for g, e in zip(got, exp)), "frames": len(frames),
                "output": {k: got.count(k) for k in dict.fromkeys(got)},
                "tracker_raw": {k: raw_got.count(k) for k in dict.fromkeys(raw_got)},
                "hidden_from_output": [l["frame"] for l in log if l.get("show") is False],
                "kept_out_of_memory": [l["frame"] for l in log if l.get("remember") is False],
            }
            row[variant]["other_wallet_frames"] = got.count("other") + got.count("both")
            results.setdefault(name, {})[variant] = {
                "title": f"{name}, {VARIANTS[variant]}", "what": what, "should": should, "frames": frames,
                "masks": outs, "scores": scores, "areas": [int(m.sum()) for m in outs], "exp": exp, "got": got,
                "log": log}
        table.append(row)
        print(f"{name:16s} correct  A {row['A']['correct']:2d}  B {row['B']['correct']:2d}  C {row['C']['correct']:2d}"
              f"  / {len(frames)}   | C output {row['C']['output']} | C tracker itself {row['C']['tracker_raw']}")

    # one video per scenario and variant, in the same format as the synthetic baseline videos
    for name, by_variant in results.items():
        for variant, r in by_variant.items():
            (OUT / variant).mkdir(exist_ok=True)
            first = st.render(r, 0)
            writer = cv2.VideoWriter(str(OUT / variant / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"avc1"), 5,
                                     first.shape[1::-1])
            for fi in range(len(r["frames"])):
                writer.write(st.render(r, fi))
            writer.release()
            make_web_playable(OUT / variant / f"{name}.mp4")
            cv2.imwrite(str(OUT / variant / f"{name}.png"), st.contact_sheet(r))
    write_index(table, results)
    (OUT / "e2_results.json").write_text(json.dumps(
        {"table": table, "checks": {n: r["C"]["log"] for n, r in results.items()},
         "rule": {"area_growth": AREA_GROWTH, "area_partial": AREA_PARTIAL, "similarity_normal": SIM_NORMAL,
                  "similarity_partial": SIM_PARTIAL}}, indent=1, default=str))
    print(f"\nwrote {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
