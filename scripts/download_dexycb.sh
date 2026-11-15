#!/usr/bin/env bash
# DexYCB (CC BY-NC 4.0, non-commercial; https://dex-ycb.github.io): 10 subjects x 100 sequences,
# 8 RealSense cameras, MANO hand + YCB object poses. Files are on the authors' Google Drive.
#
# Usage: scripts/download_dexycb.sh [--all-cams | --full] [--bop] [subjects 1..10, default all]
#   (default)   what OneTouch trains on: colour images + labels of 4 of the 8 cameras + per-sequence
#               meta.yml / pose.npz. Streamed and filtered, no archive is kept:
#               ~12.5 GB download, ~2.5 GB kept per subject (~25 GB for all 10).
#   --all-cams  the same for all 8 cameras (no depth): same download, ~5 GB kept per subject
#               (estimate: twice the 4-camera size).
#   --full      the original archives, unmodified (everything incl. depth), saved to
#               <dir>/archives/ (resumable, size-checked) and then extracted: 124.9 GB download
#               for the 10 subjects; needs the archives + the extracted data on disk.
#   --bop       also bop.tar.gz (BOP-format object-pose version, 1.25 GB).
# Always fetched: calibration/ (camera extrinsics/intrinsics, 14 KB) and models/ (YCB object
# meshes, 1.46 GB).
# Destination: $DEXYCB_DIR (default data/dexycb). The MANO hand model is NOT part of DexYCB:
# register at https://mano.is.tue.mpg.de and put MANO_RIGHT.pkl / MANO_LEFT.pkl in
# ~/.cache/fah/mano. (The single 127.6 GB file dex-ycb-20210415.tar.gz on the website holds the
# same data; per-subject files are used here so a failed download only repeats one subject.)
set -euo pipefail

cd "$(dirname "$0")/.."
DIR=${DEXYCB_DIR:-data/dexycb}
mkdir -p "$DIR"
cd "$DIR"

# name, Google Drive id, size in bytes (verified 2026-11-13 from the Drive download headers)
NAMES=(20200709-subject-01 20200813-subject-02 20200820-subject-03 20200903-subject-04
       20200908-subject-05 20200918-subject-06 20200928-subject-07 20201002-subject-08
       20201015-subject-09 20201022-subject-10)
IDS=(1Ehh92wDE3CWAiKG7E9E73HjN2Xk2XfEk 1Uo7MLqTbXEa-8s7YQZ3duugJ1nXFEo62
     1FkUxas8sv8UcVGgAzmSZlJw1eI5W5CXq 14up6qsTpvgEyqOQ5hir-QbjMB_dHfdpA
     1NBA_FPyGWOQF5-X9ueAat5g8lDMz-EmS 1UWIN2-wOBZX2T0dkAi4ctAAW8KffkXMQ
     1oWEYD_o3PVh39pLzMlJcArkDtMj4nzI0 1GTNZwhWbs7Mfez0krTgXwLPndvrw1Ztv
     1j0BLkaCjIuwjakmywKdOO9vynHTWR0UH 1FvFlRfX-p5a5sAWoKEGc17zKJWwKaSB-)
SIZES=(12412314463 12004145048 12197037343 12792618020 12815420651
       12610980973 12818333033 12418522598 12493882548 12341740204)
CALIB_ID=1UAwVKT4Rgb1fLcFoa1o71_-0NtSvvLAQ;  CALIB_SIZE=14464
MODELS_ID=1cAzlQBpcTatI5ykYQ8ziQiHLUG_a_UpM; MODELS_SIZE=1455399753
BOP_ID=1CPqLjsaYNjE3xSJbuWmqaMsGvyGIxiKL;    BOP_SIZE=1250767497
# all 8 serials (dex-ycb-toolkit _SERIALS order); OneTouch trains on [0, 2, 4, 6]
SERIALS=(836212060125 839512060362 840412060917 841412060263
         932122060857 932122060861 932122061900 932122062010)

MODE=cams4; BOP=0; SUBJ=()
for a in "$@"; do
  case $a in
    --all-cams) MODE=cams8 ;;
    --full) MODE=full ;;
    --bop) BOP=1 ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) SUBJ+=("$a") ;;
  esac
done
[ ${#SUBJ[@]} -eq 0 ] && SUBJ=(1 2 3 4 5 6 7 8 9 10)

url() { echo "https://drive.usercontent.google.com/download?id=$1&export=download&confirm=t"; }
size_of() { wc -c < "$1" | tr -d ' '; }

fetch() {  # id file size: resumable download into archives/, checked against the known size
  mkdir -p archives
  local f="archives/$2"
  if [ -f "$f" ] && [ "$(size_of "$f")" = "$3" ]; then return 0; fi
  echo "[$(date +%T)] downloading $2 ($(($3 / 1000000)) MB)"
  curl -L --fail --retry 5 -C - -o "$f" "$(url "$1")"
  [ "$(size_of "$f")" = "$3" ] || { echo "size mismatch for $2"; return 1; }
}

# tar include patterns: bsdtar (macOS) uses --include, GNU tar (Linux) positional --wildcards
if tar --version 2>/dev/null | grep -q "GNU tar"; then GNU=1; else GNU=0; fi
patterns() {
  local idx=(0 2 4 6)
  [ "$MODE" = cams8 ] && idx=(0 1 2 3 4 5 6 7)
  echo '*/meta.yml'; echo '*/pose.npz'
  for i in "${idx[@]}"; do
    echo "*/${SERIALS[$i]}/color_*.jpg"; echo "*/${SERIALS[$i]}/labels_*.npz"
  done
}
stream_filtered() {  # id: stream the archive through tar, keep only the patterns
  local args=() p
  while IFS= read -r p; do
    if [ $GNU = 1 ]; then args+=("$p"); else args+=(--include "$p"); fi
  done < <(patterns)
  if [ $GNU = 1 ]; then
    curl -sL --fail "$(url "$1")" | tar xzf - --wildcards "${args[@]}"
  else
    curl -sL --fail "$(url "$1")" | tar xzf - "${args[@]}"
  fi
}

# calibration + models (+ bop)
for spec in "$CALIB_ID calibration.tar.gz $CALIB_SIZE calibration" \
            "$MODELS_ID models.tar.gz $MODELS_SIZE models"; do
  set -- $spec
  if [ ! -d "$4" ]; then fetch "$1" "$2" "$3" && tar xzf "archives/$2"; fi
done
if [ $BOP = 1 ] && [ ! -d bop ]; then fetch "$BOP_ID" bop.tar.gz "$BOP_SIZE" && tar xzf archives/bop.tar.gz; fi

for s in "${SUBJ[@]}"; do
  i=$((10#$s - 1))
  name=${NAMES[$i]}
  done_f=".done_subject_$(printf %02d $((i + 1)))"
  [ "$MODE" != cams4 ] && done_f="${done_f}_$MODE"
  if [ -f "$done_f" ]; then echo "$name already done ($MODE)"; continue; fi
  echo "[$(date +%T)] $name ($MODE)"
  if [ "$MODE" = full ]; then
    fetch "${IDS[$i]}" "$name.tar.gz" "${SIZES[$i]}" && tar xzf "archives/$name.tar.gz"
  else
    stream_filtered "${IDS[$i]}"
  fi
  touch "$done_f"
  echo "[$(date +%T)] $name ok: $(du -sh "$name" | cut -f1)"
done
echo "done -> $(pwd)"
