#!/usr/bin/env bash
# run_experiments.sh
#
# Sweep the runtime knobs of basic_sfm and tabulate the resulting density per
# configuration per dataset. Wraps the binary in /bin/basic_sfm with the three
# environment variables it understands:
#
#   SFM_LOSS      one of {cauchy, huber, null}    default: cauchy
#   SFM_SCALE     scale factor for the loss       default: 2.0
#   SFM_SIDEWARD  threshold in task 3/7           default: 0.5
#
# Usage: ./run_experiments.sh
# Requires: built binary at ./bin/basic_sfm
# Output:   results/<config>_ds{1,2}.ply  + a printed table

set -u

BIN=./bin/basic_sfm
DATA1=datasets/test_data1.txt
DATA2=datasets/test_data2.txt
OUT=results
LOG=results/_logs
mkdir -p "$OUT" "$LOG"

if [ ! -x "$BIN" ]; then
  echo "ERROR: $BIN not found. Build first:  mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j" >&2
  exit 1
fi

run_cell () {
  local label="$1"; shift
  local data="$1"; shift
  local ply="$1"; shift
  local cmd="env $* timeout 240 $BIN $data $ply"
  eval "$cmd" > "$LOG/$label.log" 2>&1
  local pts cams
  pts=$(grep "element vertex" "$ply" 2>/dev/null | head -1 | awk '{print $3}')
  cams=$(grep -oE "Using [0-9]+ over [0-9]+ cameras" "$LOG/$label.log" | tail -1)
  pts=${pts:-FAIL}
  cams=${cams:-no-summary}
  printf "  %-35s %8s   %s\n" "$label" "$pts" "$cams"
}

echo "================================================================"
echo " EXPERIMENT 1 - Loss kernel comparison (spec section 6/7)"
echo "================================================================"
echo "  config                              points       cameras"
echo "  ---------------------------------------------------------"
for loss in cauchy huber null; do
  run_cell "ds1_${loss}_s2"   "$DATA1" "$OUT/ds1_${loss}_s2.ply"   SFM_LOSS=$loss SFM_SCALE=2.0
done
echo "  ---------------------------------------------------------"
for loss in cauchy huber null; do
  run_cell "ds2_${loss}_s2"   "$DATA2" "$OUT/ds2_${loss}_s2.ply"   SFM_LOSS=$loss SFM_SCALE=2.0
done

echo
echo "================================================================"
echo " EXPERIMENT 2 - Loss scale sweep (Cauchy and Huber)"
echo "================================================================"
echo "  config                              points       cameras"
echo "  ---------------------------------------------------------"
for s in 1.0 1.5 2.0 3.0 4.0; do
  run_cell "ds1_cauchy_s$s"  "$DATA1" "$OUT/ds1_cauchy_s$s.ply"  SFM_LOSS=cauchy SFM_SCALE=$s
  run_cell "ds1_huber_s$s"   "$DATA1" "$OUT/ds1_huber_s$s.ply"   SFM_LOSS=huber  SFM_SCALE=$s
done
echo "  ---------------------------------------------------------"
for s in 1.0 1.5 2.0 3.0 4.0; do
  run_cell "ds2_cauchy_s$s"  "$DATA2" "$OUT/ds2_cauchy_s$s.ply"  SFM_LOSS=cauchy SFM_SCALE=$s
  run_cell "ds2_huber_s$s"   "$DATA2" "$OUT/ds2_huber_s$s.ply"   SFM_LOSS=huber  SFM_SCALE=$s
done

echo
echo "================================================================"
echo " EXPERIMENT 3 - Sideward threshold (task 3/7)"
echo "================================================================"
echo "  config                              points       cameras"
echo "  ---------------------------------------------------------"
for sw in 0.3 0.4 0.5 0.6 0.7; do
  run_cell "ds1_cauchy_sw$sw"  "$DATA1" "$OUT/ds1_cauchy_sw$sw.ply"  SFM_LOSS=cauchy SFM_SIDEWARD=$sw
done
echo "  ---------------------------------------------------------"
for sw in 0.3 0.4 0.5 0.6 0.7; do
  run_cell "ds2_cauchy_sw$sw"  "$DATA2" "$OUT/ds2_cauchy_sw$sw.ply"  SFM_LOSS=cauchy SFM_SIDEWARD=$sw
done

echo
echo "All clouds in: $OUT/"
echo "Logs in:       $LOG/"
echo "Open the .ply files in MeshLab (with View -> Show trackball OFF)."
