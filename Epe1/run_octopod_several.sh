#!/usr/bin/env bash

# --- Modules ---
module load nbi
module load intel-oneapi/2023.0
module load mpi/2021.8.0

# --- Paths ---
LMP="/nbi/user-scratch/x/xnk400/lammps/lmp"
BASE_DIR="/nbi/user-scratch/x/xnk400/simulations/epe1"

# --- Folder list ---
DIRS=(
"2_nucleation_sites"
"3_nucleation_sites"
"Replication_10_long_cycles/2_nucleation_sites"
"Replication_10_long_cycles/3_nucleation_sites"
)

# --- Settings ---
CPUS_PER_JOB=24

# --- Run simulations in batches of 2 ---
for ((i=0; i<${#DIRS[@]}; i+=2)); do
    echo "Starting batch $((i/2 + 1))..."
    for ((j=i; j<i+2 && j<${#DIRS[@]}; j++)); do
        SIM_DIR="${BASE_DIR}/${DIRS[$j]}"
        echo "Running simulation in: $SIM_DIR on $(hostname)"
        cd "$SIM_DIR" || exit 1
        mkdir -p output

        # OpenMP settings
        export OMP_NUM_THREADS=$CPUS_PER_JOB
        export OMP_PROC_BIND=spread
        export OMP_PLACES=cores

        # Launch simulation in background
        mpirun -np 1 $LMP -sf omp -pk omp $OMP_NUM_THREADS \
            -in input.lammps \
            -log output/log.lammps &

    done
    # Wait for this batch to finish before starting the next
    wait
    echo "Batch $((i/2 + 1)) finished."
done