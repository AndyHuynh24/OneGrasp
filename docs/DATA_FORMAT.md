# Canonical data format (all sources)

Every source (EgoPressure, PressureVisionDB, later our own recordings) is converted into the same
per-sequence files under `data/processed/`. Models, metrics and patch extraction (P3) read only
this format, through `touch.data.catalog`. Raw files stay where they were downloaded; images are
**not** copied (meta.json points at them).

Built by `scripts/prepare_data.py` (code: `src/touch/data/canonical.py`, `landmarks.py`).
Summary numbers: `scripts/data_report.py`.

## Layout
```
data/processed/
  index.csv                                  one row per sequence (counts, split, detection rates)
  <source>/<participant>/<sequence>/
    meta.json
    pressure.npz      kpa        [N, 105, 185] float16   PV-convention kPa, raw (not floored)
    labels.npz        (see below)
    fingers.npz       (see below)
    landmarks_<view>.npz                     one per view
```
- `source` ∈ {`egopressure`, `pvdb`}. `participant` is prefixed by source: `ep_001`…`ep_021`,
  `pv_9002`…`pv_9039` (no collisions; these ids are what `configs/splits.yaml` lists).
- `N` = number of frames in the sequence. Row `i` of every array is the same frame;
  `meta.json["frame_ids"][i]` is the raw frame number.
- Views: `ego` (EgoPressure cam-d, head-mounted, 1920×1080) and `overhead` (EgoPressure cam-4,
  2560×1440, 9.6° from vertical; PVDB camera_1, 1920×1080, 1.6° from vertical).

## meta.json
| key | meaning |
|---|---|
| `source`, `participant`, `sequence` | ids |
| `format_version` | `canonical.FORMAT_VERSION` (now 3); `prepare_data.py` rebuilds any sequence with an older version |
| `gesture`, `family`, `variant`, `reps`, `take`, `hand` | parsed name, e.g. `index_press_high` → family `index_press`, variant `high`; hand `left`/`right` |
| `fps` | 30 (EgoPressure) or 15 (PVDB). **Define temporal windows in seconds, not frames.** |
| `n_frames`, `frame_ids` | |
| `views.<view>` | `image` (path template, `.format(frame=...)`), `width`, `height`, `K`, `dist` (OpenCV 8-coef: k1 k2 p1 p2 k3 k4 k5 k6), static `R`, `t_m` (world→camera, metres); PVDB also `sensel_homography` (pad col,row → undistorted px) |
| `raw_dir` | where the raw files live |
| `pressure_units` | `pv_convention_kpa` — comparable across sources and with PV++'s 1 kPa threshold; **not** a trustworthy physical scale, never report newtons |
| `noise_floor_kpa`, `contact_kpa` | 0.5 and 1.0 |
| `finger_source` | how fingers were located for `fingers.npz` |
| `participant_info` | as released (age, gender, lighting, …) |
| `pvdb_split_folder` | PVDB only: `train_fold_k` / `val_fold_5` / `test` |

## labels.npz (hand level, every source)
| array | shape, dtype | meaning |
|---|---|---|
| `frame_id` | [N] int32 | raw frame number |
| `t` | [N] float64 | seconds since the first frame |
| `max_kpa` | [N] float32 | frame max after zeroing < 0.5 kPa |
| `area_mm2` | [N] float32 | area ≥ 1 kPa (1 cell = 1.5625 mm²) |
| `total_kpa` | [N] float32 | sum after zeroing < 0.5 kPa |
| `centroid_rc` | [N, 2] float32 | pressure-weighted (row, col) over cells ≥ 1 kPa; NaN if none |
| `contact` | [N] bool | `max_kpa ≥ 1.0` (no minimum-area or duration rule) |

EgoPressure adds:
| array | shape | meaning |
|---|---|---|
| `has_anno` | [N] bool | MANO/joint annotation present (~88% of frames; missing at sequence start/end) |
| `joints_m` | [N, 21, 3] | GT 3D joints, world frame, metres, MediaPipe order; NaN if no annotation |
| `ego_R`, `ego_t_m` | [N, 3, 3], [N, 3] | head camera world→camera per frame (metres) |
| `gt2d_ego`, `gt2d_overhead` | [N, 21, 2] | GT joints projected into each view (pixels); NaN if missing |

## fingers.npz (per finger: thumb, index, middle, ring, pinky)
| array | shape | meaning |
|---|---|---|
| `max_kpa` | [N, 5] float32 | max pressure assigned to each finger (NaN where `valid` is False) |
| `area_mm2` | [N, 5] float32 | area ≥ 1 kPa assigned to each finger |
| `contact` | [N, 5] bool | `max_kpa ≥ 1.0` (False where not valid) |
| `valid` | [N] bool | the hand was located this frame (EgoPressure: `has_anno`; PVDB: MediaPipe found the hand) |
| `other_max_kpa` | [N] float32 | max pressure not within 20 mm of any distal phalanx (speckles, palm, blob tails) |
| `tip_height_mm` | [N, 5] float32 | EgoPressure: fingertip joint height above the pad (GT); NaN for PVDB |

How fingers are located: project the hand onto the pad (EgoPressure: GT joints via the fixed
pad↔world mapping `PAD_ORIGIN_XY_MM`, fitted on train participants only; PVDB: MediaPipe landmarks on the overhead view via the
pad homography). Each pressure cell ≥ 0.5 kPa goes to the nearest DIP→tip segment if within
20 mm (pressure spreads ~15 mm through the texture overlay; at 15 mm most press blobs spilled
into "other"). Finger spacing is ~15–20 mm, so the nearest-segment rule separates fingers.
EgoPressure only (3D joints): a finger whose tip is more than **12 mm** above the pad cannot take
pressure (`MAX_TIP_HEIGHT_MM`; chosen on train participants — stops hovering thumbs from stealing
the index blob: thumb share in index presses 10–12% → 2–3%). PVDB has no 3D, so no gate.

## landmarks_<view>.npz (MediaPipe Hand Landmarker 0.10.35, VIDEO mode, min confidence 0.3)
| array | shape | meaning |
|---|---|---|
| `lm_px` | [N, 2, 21, 3] float32 | up to 2 hands: x, y pixels; z = MediaPipe relative depth × image width; NaN if absent |
| `world_m` | [N, 2, 21, 3] float32 | MediaPipe world landmarks (metres, hand-centred) |
| `handed` | [N, 2] int8 | 0 Left, 1 Right, −1 none (MediaPipe's label; on these non-mirrored images it matched the true hand 100% of GT-checked frames) |
| `score` | [N, 2] float32 | handedness score |
| `primary` | [N] int8 | **deployable** choice of the hand: label matches the sequence's hand, else highest score; −1 none |
| `primary_gt` | [N] int8 | EgoPressure only: hand nearest the projected GT joints (≤ 80 px mean), else −1 |
| `gt_err_px` | [N] float32 | mean landmark–GT distance for `primary_gt` |

VIDEO mode tracks using past frames only, matching live use (causal). Use `primary` for
anything evaluated as deployed; `primary_gt` only for label hygiene on training data.

## Python access
```python
from touch.data.catalog import load_catalog

train = load_catalog(split="train")  # all sources
ego_val = load_catalog(source="egopressure", split="val", family="index_press")
s = ego_val[0]
s.labels["contact"], s.fingers["contact"][:, 1], s.labels["t"]
img = cv2.imread(str(s.image_path("ego", i)))
lm = s.landmarks("ego")
hand = lm["lm_px"][i, lm["primary"][i]]  # [21, 3] if primary >= 0
kpa = s.pressure()  # [N, 105, 185] float32
```

## Adding a new source (e.g. our own recordings, P7)
Write `build_<source>(raw_dir, out_root)` in `canonical.py` that produces the same three files
(+ `fingers.npz` from whatever ground truth the rig gives), add its participants to
`configs/splits.yaml`, and the rest (landmarks, index, report, catalog) works unchanged.
