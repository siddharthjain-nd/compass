"""Save a small summary of one step's results to progress/data/ (tracked in git; experiments/output/ is not).

Reads an E0 results file (experiments/e0_real_clips.py) and/or a timing file (experiments/e5_time_breakdown.py) and
keeps only the summary: per-clip scores, and ms / fps per state with the per-part split. PROGRESS.md points to these.

Run:  python experiments/progress_snapshot.py <step-name> [--e0 FILE] [--timing FILE] [--note TEXT] [--date YYYY-MM-DD]
Example:
  python experiments/progress_snapshot.py speed-round1-final --e0 experiments/output/e0/e0_results.json \\
      --timing experiments/output/e5/time_breakdown.json --note "window + skip + faster prep"
"""
import argparse
import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def e0_summary(path):
    clips = json.loads(Path(path).read_text())["clips"]
    per_clip = {name: {"correct": c["correct"], "scored": c["scored"], "other_wallet": int(c["other_wallet"]),
                       "by_truth": {k: f"{v['correct']}/{v['total']}" for k, v in c["by_truth"].items()}}
                for name, c in clips.items()}
    ok, n = sum(c["correct"] for c in per_clip.values()), sum(c["scored"] for c in per_clip.values())
    return {"correct": ok, "scored": n, "percent": round(100 * ok / max(n, 1), 1),
            "other_wallet_frames": sum(c["other_wallet"] for c in per_clip.values()), "clips": per_clip}


def timing_summary(path):
    s = json.loads(Path(path).read_text())["summary"]
    return {state: {k: v[k] for k in ("frames", "total_ms", "fps", "median_total_ms", "parts_ms", "share_skipped")
                    if k in v} for state, v in s.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--e0")
    ap.add_argument("--timing")
    ap.add_argument("--note", default="")
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    args = ap.parse_args()
    out = {"step": args.name, "date": args.date, "note": args.note}
    if args.e0:
        out["e0"] = e0_summary(args.e0)
    if args.timing:
        out["timing"] = timing_summary(args.timing)
    path = ROOT / "progress" / "data" / f"{args.date}_{args.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
