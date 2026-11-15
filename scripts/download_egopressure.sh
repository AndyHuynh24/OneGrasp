#!/usr/bin/env bash
# Stream EgoPressure tars from Google Drive and extract only the files we use (no full tar on disk).
#
# Usage (from anywhere):
#   scripts/download_egopressure.sh                      # all 63 p_XXX_index_press_{high,low,no-contact}_x5_right
#   scripts/download_egopressure.sh p_005_index_press_low_x5_right   # specific sequence(s)
#   scripts/download_egopressure.sh $(cat data/egopressure/lists/tier1.txt)
#   VIEWS="d" scripts/download_egopressure.sh            # egocentric only (default: "d 4")
#
# Needs: rclone remote "gdrive" (read-only Drive scope), macOS tar (bsdtar) and md5.
# Re-run safe: sequences with a .done file are skipped. Every stream is checked against Drive's MD5.
# A failed sequence is logged to $OUT/failed.txt and skipped; re-run to retry it.
# Stops when free disk space drops below MIN_FREE_GB (default 40).
set -uo pipefail
cd "$(dirname "$0")/.."

ROOT_ID="1FNZJG0D-tyev6fXs0Ql_IA5eLG-8fW4L"  # EgoPressure_pre-release Drive folder
REMOTE="${REMOTE:-gdrive}"
OUT="${OUT:-data/egopressure/raw}"
VIEWS="${VIEWS:-d 4}"  # cam-d = egocentric; cam-4 = overhead (optical axis 9.6 deg from pad normal)
MIN_FREE_GB="${MIN_FREE_GB:-40}"
R=(--drive-root-folder-id "$ROOT_ID" -q)

patterns=('*.force.bin' '*.anno.pkl')
for v in $VIEWS; do patterns+=("*.cam-${v}_rgb.jpeg" "*.cam-${v}_mask.png"); done

if [ $# -gt 0 ]; then
  seqs=("$@")
else
  seqs=()
  for i in $(seq -f "%03g" 1 21); do
    for g in high low no-contact; do seqs+=("p_${i}_index_press_${g}_x5_right"); done
  done
fi

mkdir -p "$OUT"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

free_gb() { df -k "$OUT" | awk 'NR==2 {print int($4 / 1048576)}'; }

fetch_one() {
  local s=$1 p=${1:0:5} dir="$OUT/$1"
  # Drive layout: p_XXX/p_XXX/<seq>.tar and p_XXX/p_XXX_configs/<seq>.json
  local meta want size src secs got n line
  meta=$(rclone lsf -F hs --hash MD5 "$REMOTE:$p/$p/$s.tar" "${R[@]}") || return 1
  [ -n "$meta" ] || { echo "not on Drive: $s" >&2; return 1; }
  want=${meta%%;*}
  size=${meta##*;}
  printf "get   %s (%.2f GB)\n" "$s" "$(echo "$size / 1000000000" | bc -l)"

  if [ -f "$OUT/$s.tar" ]; then
    src=(cat "$OUT/$s.tar")  # already downloaded in full earlier
  else
    src=(rclone cat "$REMOTE:$p/$p/$s.tar" "${R[@]}")
  fi

  # Hash the stream while tar extracts it.
  rm -f "$tmp/fifo" "$tmp/md5"
  mkfifo "$tmp/fifo"
  md5 -q <"$tmp/fifo" >"$tmp/md5" &
  local md5_pid=$!
  local start=$SECONDS
  "${src[@]}" | tee "$tmp/fifo" | tar -xf - -C "$OUT" "${patterns[@]}"
  local status=$?
  wait "$md5_pid"
  [ "$status" -eq 0 ] || { echo "stream/extract failed for $s (exit $status)" >&2; return 1; }
  secs=$((SECONDS - start))

  got=$(cat "$tmp/md5")
  if [ "$got" != "$want" ]; then
    echo "MD5 MISMATCH $s: got $got, Drive says $want" >&2
    return 1
  fi

  rclone copyto "$REMOTE:$p/${p}_configs/$s.json" "$dir/$s.json" "${R[@]}" || return 1
  n=$(find "$dir" -name '*.force.bin' | wc -l | tr -d ' ')
  line=$(printf "%s  %s.tar  frames=%s  %ss  %.1f MB/s" "$want" "$s" "$n" "$secs" \
    "$(echo "$size / 1000000 / ($secs + 0.001)" | bc -l)")
  echo "$line" >"$dir/.done"
  echo "$line" >>"$OUT/checksums.md5"
  echo "done  $line"
}

n_fail=0
for s in "${seqs[@]}"; do
  if [ -f "$OUT/$s/.done" ]; then echo "skip  $s"; continue; fi
  if [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; then
    echo "STOP: less than ${MIN_FREE_GB} GB free on disk" >&2
    exit 2
  fi
  # Several copies of this script may run on overlapping lists; mkdir is atomic, so only one
  # takes each sequence (a stale lock after a crash: remove $OUT/<seq>.lock by hand).
  if ! mkdir "$OUT/$s.lock" 2>/dev/null; then echo "busy  $s (another worker)"; continue; fi
  if [ -f "$OUT/$s/.done" ]; then rmdir "$OUT/$s.lock"; echo "skip  $s"; continue; fi
  if ! fetch_one "$s"; then
    n_fail=$((n_fail + 1))
    echo "$(date '+%F %T')  $s" >>"$OUT/failed.txt"
    echo "FAIL  $s (logged to $OUT/failed.txt)" >&2
  fi
  rmdir "$OUT/$s.lock"
done
echo "finished: ${#seqs[@]} requested, $n_fail failed"
[ "$n_fail" -eq 0 ]
