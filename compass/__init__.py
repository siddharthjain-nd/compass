"""Compass: keep track of a registered personal object through hands, covers and lookalikes.

Modules:
  models      load EdgeTAM (tracker + image segmenter) on the Mac GPU, with the MPS attention patch
  tracker     StreamingTracker: EdgeTAM fed one frame at a time, bounded memory (live camera)
  appearance  Appearance (DINOv2 embedding) + Gate: "does the mask still look like my object?"
  holder      Finder, Segmenter, ObjectState: visible / covered / held / out of view + hand-off
  sources     CameraSource (Mac camera) and VideoSource (a file replayed as if live)
  live        the live app:  python -m compass.live
"""
