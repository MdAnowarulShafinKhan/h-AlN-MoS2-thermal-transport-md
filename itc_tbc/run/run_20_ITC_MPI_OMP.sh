#!/usr/bin/env bash
# Twenty-seed launcher for AlN/MoS2 ITC simulations using MPI + OpenMP.
# Each simulation runs sequentially with 4 MPI ranks x 4 OpenMP threads/rank.
# KOKKOS is not enabled by this launcher.

set -u
set -o pipefail

# OpenMP settings required by the requested mpirun command.
export OMP_NUM_THREADS=4
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores

LMP="${LMP:-lmp_gpu}"
MPIEXEC="${MPIEXEC:-mpirun}"
INPUT="${INPUT:-AlN_MoS2_ITC_FINAL.in}"
T0="${T0:-300.0}"
DT0="${DT0:-50.0}"
HOTLAYER="${HOTLAYER:-MoS2}"
LJMODE="${LJMODE:-LB}"
CHI="${CHI:-1.0}"
NX="${NX:-1}"
NY="${NY:-1}"

# Twenty distinct positive seeds.
# The first ten preserve the seeds from the original launcher;
# the final ten are additional independent seeds.
seeds=(
    1000001 1000002 1000004 1000005 2000005
    3000003 9000009 1999993 1888889 1333337
    1777771 2444447 3555559 4666663 5777779
    6888881 7111117 8222221 8444443 8777777
)

TOTAL_RUNS="${#seeds[@]}"

for idx in "${!seeds[@]}"; do
    run_number=$((idx + 1))
    seed="${seeds[$idx]}"
    runid=$(printf "seed%02d" "${run_number}")

    echo
    echo "================================================================="
    echo "Starting ${runid}/${TOTAL_RUNS} | seed=${seed} | MPI+OpenMP | 4 ranks x 4 threads"
    echo "T0=${T0} K | DT0=${DT0} K | hot=${HOTLAYER} | ${LJMODE} | chi=${CHI}"
    echo "================================================================="

    if ! ${MPIEXEC} -np 4 \
        -x OMP_NUM_THREADS \
        -x OMP_DYNAMIC \
        -x OMP_PROC_BIND \
        -x OMP_PLACES \
        --bind-to core \
        --map-by ppr:4:node:PE=4 \
        ${LMP} \
        -sf omp \
        -pk omp 4 \
        -in "${INPUT}" \
        -var SEED "${seed}" \
        -var RUNID "${runid}" \
        -var T0 "${T0}" \
        -var DT0 "${DT0}" \
        -var HOTLAYER "${HOTLAYER}" \
        -var LJMODE "${LJMODE}" \
        -var CHI "${CHI}" \
        -var NX "${NX}" \
        -var NY "${NY}" \
        -log "log_${runid}.lammps" \
        2>&1 | tee "screen_${runid}.out"; then
        echo "ERROR: ${runid} failed. See log_${runid}.lammps and screen_${runid}.out" >&2
        exit 1
    fi

    echo "${runid} finished successfully."
done

echo
echo "ALL ${TOTAL_RUNS} ITC SIMULATIONS FINISHED SUCCESSFULLY."
