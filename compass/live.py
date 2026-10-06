"""Live tracking from the Mac camera.

  python -m compass.live                    # Mac camera; press SPACE, draw a box around your object, ENTER
  python -m compass.live --video clip.mp4   # replay a video file as if it were the camera (for testing)

Keys (window mode):
  SPACE  choose the object (draw a box, then ENTER; ESC cancels)
  r      choose again
  c      start / stop recording the raw camera feed to clips/<time>.mp4 (for building test clips)
  q/ESC  quit

Headless mode, for automated tests:
  python -m compass.live --video clip.mp4 --box x0,y0,x1,y1 --headless --out out.mp4 --report report.json

What happens per frame (the newest camera frame; older ones are skipped while busy):
EdgeTAM tracks the object (compass.tracker), the appearance check decides whether the mask still looks like
the object (compass.appearance), and the holder logic says visible / covered / in a hand / out of view, and
finds the object again when the hand lets go (compass.holder).
"""
import argparse
import json
import time
from collections import deque
from pathlib import Path

try:
    import cv2
    import numpy as np
    import torch  # noqa: F401  (checked here so a wrong environment fails with a clear message)
except ModuleNotFoundError as e:
    raise SystemExit(f"{e}\nRun this inside the project's environment:  conda activate compass")

WINDOW = "Compass"
BLUE, YELLOW, RED, INK, PANEL, GREY = (229, 135, 57), (25, 178, 250), (59, 59, 208), (255, 255, 255), (25, 26, 26), (183, 194, 195)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_box(s):
    x0, y0, x1, y1 = (float(v) for v in s.split(","))
    return np.array([x0, y0, x1, y1], np.float32)


def text(img, s, org, scale=0.6, color=INK, thick=1):
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def draw(frame, step, status, hint, event, recording):
    vis = frame.copy()
    if step is not None:
        if step.mask.any():
            tint = vis.copy()
            tint[step.mask] = BLUE
            vis = cv2.addWeighted(tint, 0.45, vis, 0.55, 0)
            contours, _ = cv2.findContours(step.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, INK, 5, cv2.LINE_AA)  # white edge: visible on any colour
            cv2.drawContours(vis, contours, -1, BLUE, 3, cv2.LINE_AA)
        unconfirmed = step.maybe if step.maybe is not None else step.raw if step.state == "unsure" else None
        if unconfirmed is not None and unconfirmed.any():  # grey outline: seen, but not confirmed as your object
            contours, _ = cv2.findContours(unconfirmed.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, GREY, 2, cv2.LINE_AA)
        if step.holder is not None:
            contours, _ = cv2.findContours(step.holder.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, YELLOW, 3, cv2.LINE_AA)
    h, w = vis.shape[:2]
    cv2.rectangle(vis, (0, 0), (w, 64), PANEL, -1)
    headline = step.describe() if step is not None else "Not tracking yet"
    color = {"visible": BLUE, "held": YELLOW}.get(step.state if step else "", INK)
    text(vis, headline, (14, 30), 0.8, color, 2)
    text(vis, status, (14, 54), 0.5)
    if event:
        text(vis, event, (14, 92), 0.6, YELLOW)
    if recording:
        cv2.circle(vis, (w - 24, 24), 9, RED, -1, cv2.LINE_AA)
        text(vis, "REC", (w - 80, 30), 0.6, RED, 2)
    text(vis, hint, (14, h - 14), 0.5)
    return vis


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="camera index (0 = built-in)")
    ap.add_argument("--video", help="replay this video file as if it were the camera")
    ap.add_argument("--box", type=parse_box, help="x0,y0,x1,y1: start tracking this box on the first frame")
    ap.add_argument("--finder-every", type=int, default=1,
                    help="while the object is in a hand, look for the hand-off every N processed frames (default 1)")
    ap.add_argument("--headless", action="store_true", help="no window (needs --box); use with --out/--report")
    ap.add_argument("--out", help="write the annotated video here")
    ap.add_argument("--report", help="write per-frame states, timings and events here (JSON)")
    ap.add_argument("--save-masks", help="save each processed frame's mask (320x180) + camera frame index here (.npz)")
    ap.add_argument("--max-seconds", type=float, default=0, help="stop after this long (0 = no limit)")
    args = ap.parse_args()
    if args.headless and args.box is None:
        ap.error("--headless needs --box")
    from compass.models import ROOT

    if args.report is None and not args.headless:  # keep a log of every live session, for diagnosing issues
        (ROOT / "clips" / "sessions").mkdir(parents=True, exist_ok=True)
        args.report = str(ROOT / "clips" / "sessions" / time.strftime("%Y%m%d-%H%M%S.json"))

    print("loading models (EdgeTAM, DINOv2) ...", flush=True)
    from compass.appearance import Appearance
    from compass.holder import Finder, ObjectState, Segmenter
    from compass.models import device, video_predictor
    from compass.sources import CameraSource, VideoSource
    from compass.tracker import StreamingTracker

    dev = device()
    obj = ObjectState(StreamingTracker(video_predictor(dev)), Appearance(dev), Finder(dev), Segmenter(dev),
                      finder_every=args.finder_every)
    src = VideoSource(args.video) if args.video else CameraSource(args.camera)
    print(f"source: {'video ' + args.video if args.video else 'camera ' + str(args.camera)} at {src.fps:.0f} fps", flush=True)

    writer, report, saved = None, {"frames": [], "events": []}, {"frame": [], "mask": []}
    step, seq, tracking = None, -1, False
    proc_times, last_event, event_until, t_start = deque(maxlen=20), "", 0.0, time.time()
    hint = "SPACE: choose object   r: choose again   c: record clip   q: quit"
    if not args.headless:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    try:
        while True:
            frame, new_seq, captured = src.latest(after_seq=seq)
            if frame is None:
                break
            dropped = new_seq - seq - 1 if seq >= 0 else 0
            seq = new_seq
            t0 = time.time()
            if not tracking and args.box is not None:
                step, tracking = obj.start(frame, args.box), True
            elif tracking:
                step = obj.update(frame)
                for _, e in step.events:
                    last_event, event_until = e, time.time() + 3
                    report["events"].append({"camera_frame": seq, "event": e})
                    print(f"[frame {seq}] {e}", flush=True)
            proc_times.append(time.time() - t0)
            fps = len(proc_times) / sum(proc_times) if tracking and sum(proc_times) else 0
            lag = time.time() - captured
            processed = len(report["frames"]) + 1
            status = (f"{fps:.1f} fps processed | lag {lag:.2f} s | camera frames used {processed}/{src.frames_seen}"
                      if tracking else "Press SPACE and draw a box around your object")
            vis = draw(frame, step if tracking else None, status, hint, last_event if time.time() < event_until else "",
                       src.recording)
            if tracking and args.save_masks:
                saved["frame"].append(seq)
                saved["mask"].append(cv2.resize(step.mask.astype(np.uint8), (320, 180), interpolation=cv2.INTER_NEAREST))
            if tracking:
                report["frames"].append({"camera_frame": seq, "state": step.state, "detail": step.detail,
                                         "shown": bool(step.mask.any()), "score": round(step.score, 2),
                                         "process_s": round(proc_times[-1], 3), "lag_s": round(lag, 3),
                                         "dropped_before": dropped, "debug": step.debug})
            if args.out:
                if writer is None:
                    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"avc1"), 10, vis.shape[1::-1])
                writer.write(vis)
            if not args.headless:
                cv2.imshow(WINDOW, vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break  # q, ESC, or the window was closed
                if key in (ord(" "), ord("r")):
                    box = cv2.selectROI(WINDOW, frame, showCrosshair=True)  # x, y, w, h; ENTER confirms
                    if box[2] > 0 and box[3] > 0:
                        x, y, w, h = box
                        step, tracking = obj.start(frame, (x, y, x + w, y + h)), True
                        proc_times.clear()
                        report = {"frames": [], "events": []}
                if key == ord("c") and isinstance(src, CameraSource):
                    if src.recording:
                        print(f"saved clip {src.stop_recording()}", flush=True)
                    else:
                        print(f"recording to {src.start_recording(ROOT / 'clips')} (press c again to stop)", flush=True)
            if args.max_seconds and time.time() - t_start > args.max_seconds:
                break

    except KeyboardInterrupt:
        print("stopped (Ctrl+C)")
    finally:  # always save the clip, video and session log
        if isinstance(src, CameraSource) and src.recording:
            print(f"saved clip {src.stop_recording()}")
        src.stop()
        if writer is not None:
            writer.release()
            from compass.video_io import make_web_playable

            make_web_playable(args.out)
        cv2.destroyAllWindows()
        frames = report["frames"]
        if frames:
            t = np.array([f["process_s"] for f in frames[1:]] or [0])
            report["summary"] = {
                "processed_frames": len(frames), "source_frames": src.frames_seen,
                "fraction_processed": round(len(frames) / max(src.frames_seen, 1), 3),
                "processing_fps": round(1 / t.mean(), 2) if t.mean() else None,
                "median_lag_s": float(np.median([f["lag_s"] for f in frames])), "final_state": frames[-1]["state"],
                "final_detail": frames[-1]["detail"],
            }
            print(json.dumps(report["summary"]))
        if args.report:
            Path(args.report).write_text(json.dumps(report, indent=1, default=str))
            print(f"session log: {args.report}")
        if args.save_masks and saved["frame"]:
            np.savez_compressed(args.save_masks, frame=np.array(saved["frame"]), mask=np.stack(saved["mask"]).astype(bool))


if __name__ == "__main__":
    main()
