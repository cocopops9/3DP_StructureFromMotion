#!/usr/bin/env bash
#
# run_experiments.sh
#
# Reproduces the full experimental sweep for the SfM lab: every dataset
# (DS1..DS4) is reconstructed with both the ORB pipeline (data*.txt) and
# the SuperGlue pipeline (data*_superglue.txt), and each pair is run four
# times: baseline, OPTIONAL 1, OPTIONAL 2, OPTIONAL 1+2. The pixel error
# is recovered by exporting the per-dataset focal length in SFM_FOCAL_PX
# before each run.
#
# Each run produces:
#   log/<name>.log    full stdout/stderr, including the FINAL METRICS block
#   results/<name>.ply  the reconstructed point cloud
#
# Naming convention for <name>:
#   cloudN              baseline ORB on DSN
#   cloudN_dl           baseline SuperGlue on DSN
#   cloudN_opt1         OPTIONAL 1 only
#   cloudN_dl_opt1      OPTIONAL 1, SuperGlue input
#   cloudN_opt2         OPTIONAL 2 only
#   cloudN_dl_opt2      OPTIONAL 2, SuperGlue input
#   cloudN_opt1+2       OPTIONAL 1 and 2 combined
#   cloudN_dl_opt1+2    OPTIONAL 1 and 2 combined, SuperGlue input
#
# Usage:
#   ./run_experiments.sh                     run all 32 configurations
#   ./run_experiments.sh DS3                 run only DS3 (8 configurations)
#   ./run_experiments.sh DS1 DS2             run only DS1 and DS2

set -u

# Project root is the directory containing this script.
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
BIN="${PROJECT_ROOT}/bin/basic_sfm"
MATCH_DIR="${PROJECT_ROOT}/matching_files"
LOG_DIR="${PROJECT_ROOT}/log"
PLY_DIR="${PROJECT_ROOT}/results"

mkdir -p "${LOG_DIR}" "${PLY_DIR}"

if [ ! -x "${BIN}" ]; then
    echo "ERROR: binary not found at ${BIN}." >&2
    echo "Build first: cd build && cmake .. && make" >&2
    exit 1
fi

# Per-dataset focal length in pixels, used to convert the normalized
# reprojection error returned by basic_sfm into a pixel error. DS1 and
# DS2 share the teacher's 3dp_cam.yml calibration; DS3 and DS4 share the
# self-captured calibrationfile.yml.
declare -A FOCAL_PX
FOCAL_PX[DS1]=892.37
FOCAL_PX[DS2]=892.37
FOCAL_PX[DS3]=1295.997
FOCAL_PX[DS4]=1295.997

# Datasets selected on the command line. Default to all four.
if [ $# -eq 0 ]; then
    DATASETS=(DS1 DS2 DS3 DS4)
else
    DATASETS=("$@")
fi

# Pipelines and optional flags. The combination of (pipeline, config)
# determines both the input file suffix and the env vars set on the run.
PIPELINES=(orb dl)
CONFIGS=(base opt1 opt2 "opt1+2")

# Per-run wall-clock cap. The two optionals are noticeably slower than
# the baseline on the bigger datasets, and OPTIONAL 1 with SuperGlue can
# go through many seed candidates before converging. Three minutes is a
# safe upper bound for all 32 cells on a recent laptop; bump if needed.
TIMEOUT_SECONDS=180

run_one()
{
    local ds_label="$1"
    local pipeline="$2"
    local cfg="$3"

    # Strip the DS prefix to recover the integer index.
    local ds_idx="${ds_label#DS}"
    local focal="${FOCAL_PX[$ds_label]}"

    # Resolve the input file path based on the pipeline.
    local data_file
    if [ "${pipeline}" = "orb" ]; then
        data_file="${MATCH_DIR}/data${ds_idx}.txt"
    else
        data_file="${MATCH_DIR}/data${ds_idx}_superglue.txt"
    fi

    if [ ! -f "${data_file}" ]; then
        echo "  [SKIP] ${data_file} not found, skipping run."
        return
    fi

    # Build the run name following the team's convention.
    local name="cloud${ds_idx}"
    if [ "${pipeline}" = "dl" ]; then
        name="${name}_dl"
    fi
    if [ "${cfg}" != "base" ]; then
        name="${name}_${cfg}"
    fi

    # Resolve env vars from the config flag.
    local env_opt1=0
    local env_opt2=0
    case "${cfg}" in
        base)    env_opt1=0; env_opt2=0 ;;
        opt1)    env_opt1=1; env_opt2=0 ;;
        opt2)    env_opt1=0; env_opt2=1 ;;
        opt1+2)  env_opt1=1; env_opt2=1 ;;
    esac

    local log_file="${LOG_DIR}/${name}.log"
    local ply_file="${PLY_DIR}/${name}.ply"

    echo "  [RUN ] ${name}  (focal=${focal} px, opt1=${env_opt1}, opt2=${env_opt2})"

    SFM_FOCAL_PX="${focal}" SFM_opt1="${env_opt1}" SFM_opt2="${env_opt2}" \
        timeout "${TIMEOUT_SECONDS}" "${BIN}" "${data_file}" "${ply_file}" \
        > "${log_file}" 2>&1

    local rc=$?
    if [ ${rc} -eq 124 ]; then
        echo "  [TIME] ${name}  reached ${TIMEOUT_SECONDS}s cap, partial log saved."
    elif [ ${rc} -ne 0 ]; then
        echo "  [FAIL] ${name}  exit ${rc}, log: ${log_file}"
    fi
}

# Main sweep.
for ds in "${DATASETS[@]}"; do
    if [ -z "${FOCAL_PX[$ds]+x}" ]; then
        echo "Unknown dataset label: ${ds}. Use one of: DS1 DS2 DS3 DS4." >&2
        continue
    fi
    echo "=== ${ds} ==="
    for pipeline in "${PIPELINES[@]}"; do
        for cfg in "${CONFIGS[@]}"; do
            run_one "${ds}" "${pipeline}" "${cfg}"
        done
    done
done

# Summary table.
echo
echo "=== Summary (parsed from FINAL METRICS in each log) ==="
printf "%-22s %-10s %-10s %-12s\n" "name" "cams" "points" "err_px"
printf -- "------------------------------------------------------------\n"
for log in "${LOG_DIR}"/*.log; do
    [ -f "${log}" ] || continue
    name="$(basename "${log}" .log)"
    block="$(awk '/===== FINAL METRICS =====/,/=========================/' "${log}")"
    cams="$(echo "${block}" | sed -nE 's/.*Registered cameras[[:space:]]*:[[:space:]]*([0-9]+)[[:space:]]*\/[[:space:]]*([0-9]+).*/\1\/\2/p')"
    pts="$(echo  "${block}" | sed -nE 's/.*Valid 3D points[[:space:]]*:[[:space:]]*([0-9]+).*/\1/p')"
    err="$(echo  "${block}" | sed -nE 's/.*Mean reprojection err:[[:space:]]*([0-9.eE+\-]+)[[:space:]]*px.*/\1/p')"
    [ -z "${cams}" ] && cams="incomplete"
    [ -z "${pts}" ]  && pts="-"
    [ -z "${err}" ]  && err="-"
    printf "%-22s %-10s %-10s %-12s\n" "${name}" "${cams}" "${pts}" "${err}"
done
