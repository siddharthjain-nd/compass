"""Make the MP4s from OpenCV play in every browser, Safari included.

OpenCV writes H.264 with the index ("moov" atom) at the end of the file; Safari often refuses to play such
local videos. This re-encodes to H.264 Main profile, yuv420p, with the index first (+faststart), using the
ffmpeg binary bundled with the imageio-ffmpeg package.

Called after writing any video (experiments, live app). To fix videos that already exist:
  python -m compass.video_io                     # every .mp4 under experiments/output/
  python -m compass.video_io path/to/dir_or_file.mp4
"""
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]


def make_web_playable(path):
    path = Path(path)
    tmp = path.with_name(path.stem + ".web-tmp.mp4")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(path),
                    "-c:v", "libx264", "-profile:v", "main", "-pix_fmt", "yuv420p", "-preset", "veryfast",
                    "-crf", "23", "-movflags", "+faststart", "-an", str(tmp)], check=True)
    tmp.replace(path)


def is_web_playable(path):
    """True if the index ("moov") comes before the media data ("mdat"), the part Safari cares about."""
    b = Path(path).read_bytes()
    moov, mdat = b.find(b"moov"), b.find(b"mdat")
    return moov != -1 and (mdat == -1 or moov < mdat)


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "experiments" / "output"
    files = [target] if target.is_file() else sorted(target.rglob("*.mp4"))
    for f in files:
        if not is_web_playable(f):
            make_web_playable(f)
    print(f"{len(files)} videos checked; all browser-playable: {all(is_web_playable(f) for f in files)}")
