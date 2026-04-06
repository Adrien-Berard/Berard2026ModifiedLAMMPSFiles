#!/usr/bin/env bash
# --- Load modules ---
module load nbi
module load intel-oneapi/2023.0
module load mkl/2023.0.0
module load mpi/2021.8.0
module load tbb/2021.8.0 compiler-rt/2023.0.0 oclfpga/2023.0.0
module load compiler/2023.0.0

# --- Paths ---
LMP="/nbi/user-scratch/x/xnk400/lammps/lmp"
SIM_DIR="/nbi/user-scratch/x/xnk400/simulations/A_sustained_100_cycles"
OUTPUT_DIR="/nbi/user-scratch/x/xnk400/simulations/A_sustained_100_cycles/output"

# --- Settings ---
MPI_TASKS=24        # one per physical core
OMP_THREADS=2       # x2 hyperthreads

# --- Setup ---
mkdir -p ${OUTPUT_DIR}
cd ${OUTPUT_DIR}

# --- Run ---
export OMP_NUM_THREADS=${OMP_THREADS}
mpirun -np ${MPI_TASKS} ${LMP} -sf omp -in ${SIM_DIR}/input.lammps > output.log