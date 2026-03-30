#!/bin/bash

# ----------------------------
# Resume-Friendly Parallel LAMMPS Runner
# ----------------------------

# Path to your compiled LAMMPS executable
LAMMPS_EXEC=~/PhD/Documents/Berard2025/modified_lammps_Feb2024/build/lmp

# Base directory containing all simulation variants
BASE_DIR="/home/xnk400/PhD/Documents/Berard2025/ParameterScanDifferentSwi6Differentp2-11-03-26"

# Activate virtual environment if needed
source ~/venvs/venvLAMMPS/bin/activate

# Detect number of CPU cores
MAX_JOBS=6 
echo "Detected $N_CORES CPU cores. Running up to $N_CORES simulations in parallel."

# Disable OpenMP threads
export OMP_NUM_THREADS=1

# Find all input.lammps files that do NOT have log.lammps yet
INPUT_FILES=$(find "$BASE_DIR" -type f -name "input.lammps" ! -execdir test -e log.lammps \; -print)

if [ -z "$INPUT_FILES" ]; then
    echo "No new simulations to run. All log.lammps files exist."
    exit 0
fi

# Run simulations in parallel
echo "$INPUT_FILES" | parallel -j $MAX_JOBS --bar '
    SIM_DIR=$(dirname {})
    cd "$SIM_DIR" || exit
    echo "Running simulation in $SIM_DIR ..."
    '"$LAMMPS_EXEC"' -in input.lammps > log.lammps
    echo "Finished simulation in $SIM_DIR"
'

echo "All simulations completed."