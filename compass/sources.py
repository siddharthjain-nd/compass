"""Frame sources that always hand over the NEWEST frame, like a live camera.

A background thread reads frames at the source's own rate. The tracker takes whatever is newest when it is
ready; frames that arrive while it is busy are skipped (counted as dropped), so the picture never lags behind
real time by more than one processing step.

  CameraSource   the Mac's camera (AVFoundation). Can also record the raw camera feed to a clip.
  VideoSource    a video file replayed at its real speed, as if it were a camera (for testing).
"""
import threading
import time
from pathlib import Path

import cv2


class _LatestFrame:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame, self._seq, self._ts = None, -1, 0.0
        self.ended = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _publish(self, frame):
        with self._lock:
            self._frame, self._seq, self._ts = frame, self._seq + 1, time.time()

    def latest(self, after_seq=-1, timeout=5.0):
        """Newest frame with sequence number > after_seq, waiting for one if needed.
        Returns (frame, seq, capture_time) or (None, seq, ts) if the source ended or timed out."""
        deadline = time.time() + timeout
        while True:
            with self._lock:
                if self._seq > after_seq:
                    return self._frame, self._seq, self._ts
            if self.ended or time.time() > deadline:
                return None, self._seq, self._ts
            time.sleep(0.002)

    @property
    def frames_seen(self):
        return self._seq + 1

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)


class CameraSource(_LatestFrame):
    def __init__(self, index=0, width=1280, height=720):
        super().__init__()
        self.cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if not self.cap.isOpened():
            raise RuntimeError(
                "Could not open the camera. On macOS, allow camera access for your terminal app in "
                "System Settings > Privacy & Security > Camera, then run again.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._writer, self.recording_path = None, None
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                self.ended = True
                break
            if self._writer is not None:
                self._writer.write(frame)
            self._publish(frame)
        self.cap.release()

    def start_recording(self, folder="clips"):
        """Record the raw camera feed (every frame, not just the processed ones) to clips/<time>.mp4."""
        frame, _, _ = self.latest()
        Path(folder).mkdir(exist_ok=True)
        self.recording_path = Path(folder) / time.strftime("%Y%m%d-%H%M%S.mp4")
        h, w = frame.shape[:2]
        self._writer = cv2.VideoWriter(str(self.recording_path), cv2.VideoWriter_fourcc(*"avc1"), self.fps, (w, h))
        return self.recording_path

    def stop_recording(self):
        writer, self._writer = self._writer, None
        if writer is not None:
            writer.release()
        return self.recording_path

    @property
    def recording(self):
        return self._writer is not None


class VideoSource(_LatestFrame):
    """Replays a video file at its own frame rate, dropping frames the consumer is too slow to take."""

    def __init__(self, path, speed=1.0):
        super().__init__()
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.speed = speed
        self.recording = False
        self._thread.start()

    def _run(self):
        start = time.time()
        i = 0
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                break
            due = start + i / (self.fps * self.speed)
            time.sleep(max(0.0, due - time.time()))
            self._publish(frame)
            i += 1
        self.ended = True
        self.cap.release()
