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
HAND = (150, 200, 230)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_box(s):
    x0, y0, x1, y1 = (float(v) for v in s.split(","))
    return np.array([x0, y0, x1, y1], np.float32)


def text(img, s, org, scale=0.6, color=INK, thick=1):
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def as_seen_by_user(text_, mirror):
    """Exit sides are computed in camera coordinates; in a mirrored (selfie) view, left and right swap.
    Hand names ("your left hand") are the person's own and are not swapped."""
    if not mirror or not text_:
        return text_
    return (text_.replace("to the left", "to the @R").replace("to the right", "to the left")
            .replace("to the @R", "to the right"))


def outline(vis, mask, colour, thickness, edge=False):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if edge:
        cv2.drawContours(vis, contours, -1, INK, thickness + 2, cv2.LINE_AA)  # white edge: visible on any colour
    cv2.drawContours(vis, contours, -1, colour, thickness, cv2.LINE_AA)


def draw(frame, step, status, hint, event, recording, mirror=False):
    vis = frame.copy()
    if step is not None:
        for hand in step.hands:  # every detected hand, faintly
            outline(vis, hand.mask, HAND, 1)
        if step.mask.any():
            tint = vis.copy()
            tint[step.mask] = BLUE
            vis = cv2.addWeighted(tint, 0.45, vis, 0.55, 0)
            outline(vis, step.mask, BLUE, 3, edge=True)
        unconfirmed = step.maybe if step.maybe is not None else step.raw if step.state == "unsure" else None
        if unconfirmed is not None and unconfirmed.any():  # grey outline: seen, but not confirmed as your object
            outline(vis, unconfirmed, GREY, 2)
        if step.holder is not None:
            outline(vis, step.holder, YELLOW, 3)
    if mirror:
        vis = cv2.flip(vis, 1)  # text is drawn after flipping, so it stays readable
    h, w = vis.shape[:2]
    cv2.rectangle(vis, (0, 0), (w, 64), PANEL, -1)
    headline = as_seen_by_user(step.describe(), mirror) if step is not None else "Not tracking yet"
    color = {"visible": BLUE, "held": YELLOW, "in_pocket": YELLOW}.get(step.state if step else "", INK)
    size = 0.8 if w >= 1000 else max(0.45, 0.8 * w / 1000)  # narrow pictures: smaller text, so it fits
    while cv2.getTextSize(headline, FONT, size, 2)[0][0] > w - 28 and size > 0.35:
        size -= 0.05
    text(vis, headline, (14, 30), size, color, 2)
    text(vis, status, (14, 54), 0.5)
    if event:
        text(vis, as_seen_by_user(event, mirror), (14, 92), 0.6, YELLOW)
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
    ap.add_argument("--no-hands", action="store_true", help="don't run the hand detector (MediaPipe)")
    ap.add_argument("--no-pocket", action="store_true", help="don't run pocket detection (clothes segmenter)")
    ap.add_argument("--no-window", action="store_true",
                    help="track on the whole frame instead of only around the object (slower; for comparison)")
    ap.add_argument("--encoder", default="coreml", choices=["coreml", "coreml-gpu", "torch"],
                    help="tracker image encoder: Core ML on the Neural Engine (default; ~3x faster), on the GPU, or "
                         "PyTorch. Falls back to PyTorch if the Core ML model is missing")
    ap.add_argument("--no-window-outlines", action="store_true", help="search: outline spots on the whole picture")
    ap.add_argument("--size", type=int, default=1024, choices=[1024, 768],
                    help="tracker input size: 1024 (trained size, most accurate) or 768 (~25%% faster, a few more "
                         "'not sure' while held; Core ML model: python -m compass.coreml_encoder 768)")
    ap.add_argument("--no-skip", action="store_true",
                    help="run the tracker on every frame, even while the object rests and nothing changes")
    ap.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=None,
                    help="mirror the picture like a selfie view, with left/right from your side "
                         "(default: on for the camera, off for a video file)")
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
    hands = clothes = None
    if not args.no_hands:
        from compass.hands import HandDetector

        hands = HandDetector()
        if not args.no_pocket:  # pockets are found from the hand that holds the object, so they need hands
            from compass.body import ClothesSegmenter

            clothes = ClothesSegmenter()
    predictor = video_predictor(dev, args.size)
    try:
        tracker = StreamingTracker(predictor, window=not args.no_window, encoder=args.encoder)
    except (FileNotFoundError, ImportError) as e:  # no Core ML model / coremltools: PyTorch works everywhere
        print(f"Core ML encoder unavailable ({e}); using PyTorch", flush=True)
        tracker = StreamingTracker(predictor, window=not args.no_window)
    obj = ObjectState(tracker, Appearance(dev), Finder(dev),
                      Segmenter(dev, args.size), skip_when_still=not args.no_skip, window_outlines=not args.no_window_outlines,
                      finder_every=args.finder_every, hands=hands, clothes=clothes)
    mirror = args.mirror if args.mirror is not None else not args.video
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
                step, tracking = obj.start(frame, args.box, t_ms=captured * 1000), True
            elif tracking:
                step = obj.update(frame, t_ms=captured * 1000)
                for _, e in step.events:
                    last_event, event_until = e, time.time() + 3
                    report["events"].append({"camera_frame": seq, "event": e})
                    print(f"[frame {seq}] {as_seen_by_user(e, mirror)}", flush=True)
            proc_times.append(time.time() - t0)
            fps = len(proc_times) / sum(proc_times) if tracking and sum(proc_times) else 0
            lag = time.time() - captured
            processed = len(report["frames"]) + 1
            status = (f"{fps:.1f} fps processed | lag {lag:.2f} s | camera frames used {processed}/{src.frames_seen}"
                      if tracking else "Press SPACE and draw a box around your object")
            vis = draw(frame, step if tracking else None, status, hint, last_event if time.time() < event_until else "",
                       src.recording, mirror)
            if tracking and args.save_masks:
                saved["frame"].append(seq)
                saved["mask"].append(cv2.resize(step.mask.astype(np.uint8), (320, 180), interpolation=cv2.INTER_NEAREST))
            if tracking:
                report["frames"].append({"camera_frame": seq, "state": step.state, "detail": step.detail, "note": step.note,
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
                    shown = cv2.flip(frame, 1) if mirror else frame
                    box = cv2.selectROI(WINDOW, shown, showCrosshair=True)  # x, y, w, h; ENTER confirms
                    if box[2] > 0 and box[3] > 0:
                        x, y, w, h = box
                        if mirror:  # the box was drawn on the mirrored picture
                            x = frame.shape[1] - (x + w)
                        step, tracking = obj.start(frame, (x, y, x + w, y + h), t_ms=captured * 1000), True
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
