#!/usr/bin/env bash
# Fetch the third-party model repos at pinned commits, apply our patches, download checkpoints.
# third_party/ is git-ignored; this script is how it gets re-created.
#
# Run from anywhere, inside the conda env:  conda activate compass && bash scripts/setup_third_party.sh
# Safe to re-run: existing checkouts, applied patches and downloaded checkpoints are skipped.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TP="$ROOT/third_party"
mkdir -p "$TP"

fetch() {  # fetch <dir> <url> <commit>
  local dir="$TP/$1" url="$2" sha="$3"
  if [ -d "$dir/.git" ]; then
    echo "$1: already present at $(git -C "$dir" rev-parse --short HEAD) (pinned ${sha:0:7})"
    return
  fi
  git init -q "$dir"
  git -C "$dir" remote add origin "$url"
  git -C "$dir" fetch -q --depth 1 origin "$sha"
  git -C "$dir" checkout -q FETCH_HEAD
  echo "$1: checked out ${sha:0:7}"
}

patch_once() {  # patch_once <dir> <patch file>
  local dir="$TP/$1" patch="$ROOT/patches/$2"
  if git -C "$dir" apply --reverse --check "$patch" 2>/dev/null; then
    echo "$1: $2 already applied"
  else
    git -C "$dir" apply "$patch"
    echo "$1: applied $2"
  fi
}

download() {  # download <url> <dest>
  if [ -s "$2" ]; then echo "$(basename "$2"): already downloaded"; return; fi
  curl -fsSL -o "$2" "$1"
  echo "$(basename "$2"): downloaded"
}

fetch sam2 https://github.com/facebookresearch/sam2.git 2b90b9f5ceec907a1c18123530e92e794ad901a4
fetch EdgeTAM https://github.com/facebookresearch/EdgeTAM.git 7711e012a30a2402c4eaab637bdb00a521302c91

# EdgeTAM crashes when tracking more than one object at once without this (.view -> .reshape).
patch_once EdgeTAM edgetam-multi-object-reshape.patch

SAM2_URL=https://dl.fbaipublicfiles.com/segment_anything_2/092824
for m in tiny small; do
  download "$SAM2_URL/sam2.1_hiera_$m.pt" "$TP/sam2/checkpoints/sam2.1_hiera_$m.pt"
done
# EdgeTAM ships edgetam.pt inside its repo, so no download is needed.

# SAM 2 is pip-installed (without its optional CUDA extension). EdgeTAM is NOT installed:
# its package is also named `sam2`, so our scripts put third_party/EdgeTAM first on sys.path.
if python -c "import sam2" 2>/dev/null; then
  echo "sam2 package: already installed"
else
  (cd "$TP/sam2" && SAM2_BUILD_CUDA=0 python -m pip install -q -e .)
  echo "sam2 package: installed"
fi
echo "done"
