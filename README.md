# 3DP Lab 2 — Structure from Motion

University of Padova, *3D Data Processing* course. Group of three students.

This project implements the full SfM pipeline required by the lab:
keypoint matching (ORB and SuperGlue variants), incremental
reconstruction with Ceres-based bundle adjustment, and two optional
improvements on the basic seed selection and next-best-view strategies.

## Build

The project uses CMake. From the `3DP_lab_2` directory:

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

This produces two executables under `bin/`:

* `matcher` — runs the ORB matcher on a folder of images and a YAML
  calibration file, writes a matching file for `basic_sfm`.
* `basic_sfm` — incremental reconstruction. Reads a matching file,
  writes a `.ply` point cloud.

The SuperGlue input variants (`data*_superglue.txt`) are produced by
`../SuperGluePreTrainedNetwork/superglue_script.py`. They are already
included in `matching_files/` and do not need to be regenerated to
reproduce the report.

## Running `basic_sfm`

```bash
./bin/basic_sfm <matching_file.txt> <output_cloud.ply>
```

The behaviour is configured at runtime through environment variables.
None of them are mandatory; defaults reproduce the basic pipeline.

| Variable        | Values                | Effect                                                                                          |
| :-------------- | :-------------------- | :---------------------------------------------------------------------------------------------- |
| `SFM_opt1`      | `0` (default), `1`    | Enable OPTIONAL 1 (alternative seed selection).                                                  |
| `SFM_opt2`      | `0` (default), `1`    | Enable OPTIONAL 2 (alternative next-best-view selection, Schoenberger and Frahm pyramid score). |
| `SFM_LOSS`      | `cauchy` (default), `huber`, `null` | Robust loss kernel used inside Ceres for the reprojection residuals.              |
| `SFM_SCALE`     | floating-point, default `2.0` | Loss-kernel scale parameter, ignored when `SFM_LOSS=null`.                              |
| `SFM_FOCAL_PX`  | floating-point        | If set, the final reprojection error is reported in pixels by multiplying the normalized error by this focal length. If not set, the error stays in normalized canonical units. |

The matching file already encodes the calibration, so observations
arrive at `basic_sfm` pre-multiplied by `K^-1` and the reprojection
error is computed in normalized image units. `SFM_FOCAL_PX` only
affects how the error is *displayed*, never how it is computed.

### Focal lengths used for our datasets

| Dataset       | Calibration source         | Focal length (px) |
| :------------ | :------------------------- | ----------------: |
| DS1, DS2      | `3dp_cam.yml` (teacher)    | 892.37            |
| DS3, DS4      | `calibrationfile.yml` (self-captured) | 1295.997 |

### Examples

```bash
# Baseline reconstruction on DS1 (angels), error reported in pixels.
SFM_FOCAL_PX=892.37 \
    ./bin/basic_sfm matching_files/data1.txt cloud1.ply

# Same dataset with OPTIONAL 1 only.
SFM_FOCAL_PX=892.37 SFM_opt1=1 \
    ./bin/basic_sfm matching_files/data1.txt cloud1_opt1.ply

# DS3 (coffee mug), SuperGlue input, both optionals, Huber loss.
SFM_FOCAL_PX=1295.997 SFM_opt1=1 SFM_opt2=1 SFM_LOSS=huber \
    ./bin/basic_sfm matching_files/data3_superglue.txt cloud3.ply
```

## OPTIONAL 1 — alternative seed selection

The basic strategy in `solve()` picks the camera pair with the largest
number of correspondences. On well-textured consecutive-frame pairs
this is often a short-baseline pair, which makes the triangulation
poorly conditioned. OPTIONAL 1 keeps the same enumeration but replaces
the score with a geometric quality measure computed per candidate
pair:

1. Skip pairs with fewer than 8 common observations.
2. Estimate the essential matrix `E` and a homography `H` with RANSAC
   on the matched points (threshold `0.001` in normalized coords).
   Discard the pair if the homography explains as many inliers as the
   essential matrix, which is a strong indicator of a planar scene or
   a pure rotation between the two views.
3. Decompose `E` with `cv::recoverPose` and keep the lateral and
   forward components of the recovered translation. Reject the pair
   if `|t_z| >= sqrt(t_x^2 + t_y^2)`, that is, if the camera mostly
   moved along the optical axis.
4. Score each surviving candidate as a weighted sum of pose inliers,
   common observations, the E-vs-H inlier margin, lateral motion
   (rewarded) and forward motion (penalised). The full formula is in
   `basic_sfm.cpp`.
5. Sort by score and call `incrementalReconstruction` on the
   highest-scoring pair. If it fails, fall back to the next candidate
   and so on.

Enable with `SFM_opt1=1`.

## OPTIONAL 2 — alternative next-best-view selection

The basic next-best-view strategy in `incrementalReconstruction` picks
the unregistered camera that observes the highest number of already
triangulated points. This biases the selection toward cameras whose
observations are clustered in one image region, which produces a
poorly conditioned PnP problem. OPTIONAL 2 follows the strategy from
Schoenberger and Frahm, *Structure-from-Motion Revisited* (CVPR 2016,
section 4.2):

1. Compute the bounding box of all observations once, before the
   incremental loop starts.
2. For every unregistered camera at every iteration, collect the
   observations of already triangulated points.
3. Score the camera with a three-level image pyramid (`2x2`, `4x4`,
   `8x8` grids over the bounding box). Each level contributes the
   number of occupied cells multiplied by the total cell count of
   that level (`4`, `16`, `64`), so finer levels weigh more.
4. Validate the candidate with a `solvePnPRansac` call. Reject it if
   fewer than 20 visible points are available, if fewer than 15 PnP
   inliers are found, or if the PnP inlier ratio is below 0.35.
5. Pick the surviving candidate with the highest pyramid score. Break
   ties first on the number of PnP inliers, then on raw visible-point
   count.

Enable with `SFM_opt2=1`. Both optionals can be combined; the report
discusses when this is beneficial and when it is not.

## Reproducing the experimental sweep

A driver script runs every combination required by the report:
4 datasets × 2 pipelines (ORB and SuperGlue) × 4 configurations
(baseline, OPT1, OPT2, OPT1+OPT2) = 32 runs.

```bash
./run_experiments.sh                # all 32 runs
./run_experiments.sh DS3            # only the 8 DS3 runs
./run_experiments.sh DS1 DS3        # any subset
```

Each run produces two artifacts:

* `log/<name>.log` — full stdout/stderr including the `FINAL METRICS`
  block. The naming convention is
  `cloud<N>[_dl][_opt1|_opt2|_opt1+2]`.
* `results/<name>.ply` — the reconstructed point cloud, openable in
  MeshLab or any PLY viewer.

After the sweep finishes, the script prints a summary table built by
parsing the `FINAL METRICS` block from each log:

```
name                   cams       points     err_px
------------------------------------------------------------
cloud1                 9/9        9384       0.573381
cloud1_opt1            9/9        9384       0.573381
cloud1_opt2            9/9        9532       0.574647
...
```

The pre-existing `log/` directory shipped with this submission already
contains the 32 logs that produced the numbers in the report.

## Directory layout

```
3DP_lab_2/
├── CMakeLists.txt
├── README.md                 ← this file
├── run_experiments.sh        ← sweep driver
├── src/                      ← C++ sources
│   ├── basic_sfm.cpp         ← incremental SfM, OPTIONAL 1 and 2 included
│   ├── basic_sfm.h
│   ├── features_matcher.cpp  ← ORB pipeline
│   ├── features_matcher.h
│   ├── io_utils.cpp
│   ├── io_utils.h
│   ├── matcher_app.cpp
│   └── sfm_app.cpp
├── matching_files/           ← per-dataset matching outputs
│   ├── data1.txt             ← DS1 ORB
│   ├── data1_superglue.txt   ← DS1 SuperGlue
│   ├── data2.txt …data4.txt
│   └── data2_superglue.txt …data4_superglue.txt
├── datasets/                 ← raw images and calibration YAMLs
│   ├── images_1/             ← DS1 angels
│   ├── images_2/             ← DS2 aloe
│   ├── images_3/             ← DS3 coffee mug
│   ├── images_4/             ← DS4 gnome
│   ├── 3dp_cam.yml           ← calibration for DS1, DS2
│   └── calibrationfile.yml   ← calibration for DS3, DS4
├── log/                      ← per-run logs from the reference sweep
└── results/                  ← per-run .ply clouds (populated by the script)
```

## Dataset notes

* **DS1, angels** (9 frames): four small painted objects on a white
  tabletop with a green wall behind. 
* **DS2, aloe** (15 frames): single, heavily textured potted plant
  against a green wall. 
* **DS3, coffee mug** (17 frames): mug with an airmail print on a
  saucer.
* **DS4, gnome** (13 frames): painted ceramic gnome figurine on a
  uniform background.
