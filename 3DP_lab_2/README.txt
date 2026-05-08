3D Data Processing, Lab 2: Structure from Motion
=================================================

This project implements a complete incremental Structure from Motion (SfM)
pipeline in C++ for the "3D Data Processing" course at the University of
Padova. Given a set of images of a static scene captured with a calibrated
camera, the pipeline produces a sparse 3D reconstruction of the scene and
recovers the pose of every input image. The implementation covers all
required tasks (1 through 7) plus the two optional extensions (custom seed
selection and custom next-best-view strategy).


Project layout
--------------

    3DP_lab_2/
    |
    |-- src/                                  C++ source code
    |   |-- matcher_app.cpp                   entry point of the matcher binary
    |   |-- sfm_app.cpp                       entry point of the basic_sfm binary
    |   |-- features_matcher.cpp / .h         feature extraction and matching (tasks 1, 2)
    |   |-- basic_sfm.cpp / .h                full SfM pipeline (tasks 3 to 7, OPTIONAL 1, OPTIONAL 2)
    |   |-- io_utils.cpp / .h                 PLY writer and data file I/O helpers
    |
    |-- datasets/                             provided datasets and calibrations
    |   |-- 3dp_cam.yml                       intrinsics for images_1 and images_2
    |   |-- calibrationfile.yml               alternative calibration file
    |   |-- images_1/                         9 frames, indoor object
    |   |-- images_2/                        15 frames, indoor object
    |   |-- images_3/                        17 frames, self-captured smartphone dataset
    |   |-- test_data1.txt                    pre-matched data file for images_1
    |   |-- test_data2.txt                    pre-matched data file for images_2
    |
    |-- matching_files/                       cached intermediate matching results
    |-- results/                              output ply files from sweep runs (created by run_experiments.sh)
    |
    |-- CMakeLists.txt                        build configuration
    |-- run_experiments.sh                    optional sweep helper, see below
    |-- HOW_TO_USE.pdf                        extra notes
    |-- Lab2 - Structure from Motion.pdf      official lab specification
    |-- README.txt                            this file


Prerequisites
-------------

The course Virtual Machine already includes everything needed. On a Debian
or Ubuntu system the following packages are sufficient:

    sudo apt install build-essential cmake libboost-filesystem-dev libopencv-dev libomp-dev
    sudo apt install libceres-dev libyaml-cpp-dev libgtest-dev libeigen3-dev gfortran

The pipeline has been tested against gcc 11.4 / 13.3, OpenCV 4.5 / 4.6 and
Ceres 2.0. Other recent versions are expected to work as well.


Build
-----

Out-of-source CMake build, Release mode for full optimization:

    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make

The two executables are written into the bin/ folder of the project root:

    bin/matcher
    bin/basic_sfm

If parallel make produces transient errors on this CMake setup, fall back to
serial: make -j1.


Running the matcher
-------------------

The matcher reads a folder of input images and a calibration file, extracts
feature points from each image, matches them across image pairs, applies
geometric verification, and writes the result to a single text data file
that basic_sfm can consume:

    ./matcher <calibration.yml> <images_folder> <output_data_file> <use_modern_features> [<focal_length_scale>]

Arguments:

    calibration.yml         OpenCV YAML calibration file containing K and the distortion vector D
    images_folder           folder with the images of the sequence
    output_data_file        path of the text data file to produce (e.g. data1.txt)
    use_modern_features     0 = classical ORB pipeline (the lab baseline)
                            1 = modern feature pipeline (SIFT/SuperPoint depending on build)
    focal_length_scale      optional multiplier applied to the focal length read from the
                            calibration file. For the lab-provided datasets, set this to 1.1.

Example for the two provided datasets:

    ./matcher ../datasets/3dp_cam.yml ../datasets/images_1 data1.txt 0 1.1
    ./matcher ../datasets/3dp_cam.yml ../datasets/images_2 data2.txt 0 1.1


Running basic_sfm
-----------------

The SfM binary takes a data file produced by the matcher and writes a PLY
point cloud:

    ./basic_sfm <input_data_file> <output_ply_file>

Examples on the pre-matched files shipped with the lab:

    ./basic_sfm ../datasets/test_data1.txt cloud1.ply
    ./basic_sfm ../datasets/test_data2.txt cloud2.ply

The output PLY can be opened in MeshLab, CloudCompare, or any other point
cloud viewer.


Pipeline overview
-----------------

The implementation covers the seven mandatory tasks and the two optional
extensions specified in the lab document. The signposts below mark where
each task lives in the source.

    Task 1 (geometric verification, in features_matcher.cpp)
        Run RANSAC-based fundamental matrix estimation on every candidate
        match set; reject pairs of frames whose epipolar geometry is not
        coherent with the matches.

    Task 2 (feature extraction and matching, in features_matcher.cpp)
        Detect and describe interest points in every frame and match them
        between every pair of overlapping frames. Both the classical ORB
        path and the modern feature path are implemented.

    Task 3 (seed pair selection, in basic_sfm.cpp)
        For every candidate seed pair, fit both an essential matrix E and
        a homography H. Accept the pair only if E has more inliers than H
        and the recovered relative motion is dominated by sideward
        translation, that is, the baseline is wide enough to triangulate
        reliably.

    Task 4 (triangulation, in basic_sfm.cpp)
        Linear triangulation of new 3D points from every newly-registered
        pair of cameras. Cheirality is checked and points behind the camera
        are pruned.

    Task 5 (memory layout, in basic_sfm.cpp)
        Layout of the parameter buffer that backs all camera and point
        blocks for Ceres bundle adjustment.

    Task 6 (bundle adjustment, in basic_sfm.cpp)
        Build the Ceres residual block list using the ReprojectionError
        functor and run Levenberg-Marquardt optimization on the active set
        of cameras and points.

    Task 7 (divergence detection and recovery, in basic_sfm.cpp)
        Detect "exploding" geometry (max_dist greater than 100, scene
        scale exceeds the initial baseline by two orders of magnitude) and
        catastrophic point loss (fewer than 10 valid points), and trigger
        a restart from a different seed pair when either condition is met.

    OPTIONAL 1 (custom seed selection, in basic_sfm.cpp)
        Replaces the simple "iterate until accepted" seed search with a
        scoring strategy that ranks candidate seed pairs by the strength
        of their relative motion before submitting them to Task 3.

    OPTIONAL 2 (custom next-best-view, in basic_sfm.cpp)
        Replaces the basic next-best-view strategy with a grid-based view
        selection inspired by the Schoenberger and Frahm "Structure from
        Motion Revisited" paper, which weighs candidate views by the
        spatial coverage of the points they observe in addition to the
        raw count.


Final metrics block
-------------------

At the end of every successful run, basic_sfm prints a "===== FINAL METRICS
=====" block with the three quantities required by the lab specification
for the comparison tables across datasets and configurations:

    Registered cameras : how many cameras were successfully registered, plus
                         the corresponding success rate as a percentage of the
                         total number of input frames.

    Valid 3D points    : the reconstruction density, that is, the number of
                         points that survived all cheirality and outlier
                         pruning passes. This number matches the vertex count
                         in the output PLY.

    Mean reprojection error : the arithmetic mean over all (camera, point)
                              pairs of the Euclidean distance between every
                              observation and the projection of the
                              corresponding 3D point through its camera. The
                              projection model is identical to the one used
                              by the Ceres ReprojectionError functor (axis-
                              angle rotation, translation, divide by depth).

The reprojection error is in normalized canonical image coordinates by
default, because every observation in the data file is pre-divided by the
calibration matrix K when the matcher writes it (the SfM pipeline operates
in a canonical K = identity space). To convert to pixels, multiply by the
focal length used during matching. Rather than asking the user to multiply
by hand, the binary itself performs the conversion when the focal length
is supplied through the SFM_FOCAL_PX environment variable described below.

Example output without SFM_FOCAL_PX:

    ===== FINAL METRICS =====
    Registered cameras  : 9 / 9  (success rate 100 %)
    Valid 3D points     : 3223
    Mean reprojection err: 0.000620759 (normalized image units)
      set SFM_FOCAL_PX=<focal length in pixels> before running to print pixel error directly
    =========================

Example output with SFM_FOCAL_PX=892.37 (the focal length of the provided
3dp_cam.yml calibration):

    ===== FINAL METRICS =====
    Registered cameras  : 9 / 9  (success rate 100 %)
    Valid 3D points     : 3223
    Mean reprojection err: 0.553947 px  (focal = 892.37 px)
    =========================

For the provided datasets the mean reprojection error is typically in the
range 0.5 to 0.8 pixels, which is the expected order of magnitude after
convergence of bundle adjustment for a well-textured scene.


Runtime environment variables
-----------------------------

basic_sfm reads three environment variables at runtime. They default to
sensible values, so a plain invocation reproduces the lab's nominal
configuration:

    SFM_LOSS       Robust loss kernel passed to Ceres. One of:
                   "cauchy" (default), "huber", or "null" (squared L2).

    SFM_SCALE      Scale parameter of the loss kernel, multiplied by the
                   internal max_reproj_err_ before being handed to Ceres.
                   Default: 2.0.

    SFM_FOCAL_PX   Focal length in pixels. When set, the mean reprojection
                   error printed in the final metrics block is converted
                   from normalized image units to pixels. Read fx from the
                   K matrix of the calibration file used during matching:
                   for the provided datasets this is the (0,0) entry of K
                   in datasets/3dp_cam.yml, which is approximately 892.37.

Examples:

    SFM_LOSS=huber SFM_SCALE=1.5 ./basic_sfm data1.txt cloud1.ply
    SFM_FOCAL_PX=892.37 ./basic_sfm data1.txt cloud1.ply
    SFM_LOSS=cauchy SFM_SCALE=2.0 SFM_FOCAL_PX=892.37 ./basic_sfm data2.txt cloud2.ply


Sweep helper: run_experiments.sh
--------------------------------

The shell script run_experiments.sh runs basic_sfm on the two provided
datasets across several combinations of SFM_LOSS and SFM_SCALE, writes one
PLY per cell into the results/ folder, captures per-cell logs in
results/_logs/, and prints a tabulated summary with point counts and
registered-camera ratios. It is meant to populate the comparison tables
required by the report.

Build first, then:

    chmod +x run_experiments.sh   # only the first time
    ./run_experiments.sh

The script reads basic_sfm from ./bin/basic_sfm and the data files from
datasets/. Total runtime depends on the host but is roughly 5 to 15
minutes on a modern laptop.


Datasets
--------

The datasets/ folder contains three input sequences:

    images_1/    9 frames, captured with the calibration in 3dp_cam.yml
    images_2/   15 frames, same calibration as above
    images_3/   17 frames, self-captured smartphone sequence

Each sequence can be processed end to end via the matcher, or skipped
straight to basic_sfm using the pre-matched test_data*.txt files for
images_1 and images_2.

Important: the lab evaluation pipeline runs the matcher on the original
images, not on the pre-matched files. Always verify that the full
matcher + basic_sfm chain reconstructs each scene before submitting.


Output: the PLY file
--------------------

basic_sfm writes a binary PLY containing the reconstructed 3D points with
RGB color sampled from the matched feature descriptors, plus camera
centers as additional points (typically rendered as a separate visual
cluster in MeshLab). The vertex count printed in the PLY header agrees
with the "Valid 3D points" value reported in the final metrics block.


Reproducibility note
--------------------

Two sources of non-determinism affect basic_sfm:

    1. cv::solvePnPRansac inside the registration loop draws random
       sample subsets, so different runs on the same input can register
       cameras in slightly different orders.

    2. Ceres bundle adjustment is multi-threaded, so floating-point
       summation across threads is non-associative.

As a consequence, two consecutive runs on the same input will produce
similar but not bit-identical reconstructions. For the comparison tables,
either run each configuration once and document the seed conditions, or
run it several times and report the median.
