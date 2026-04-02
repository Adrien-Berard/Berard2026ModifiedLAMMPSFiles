#!/usr/bin/env bash

# --- Create safe_remove() function ---
# Not define officialy here as alreayd define on muon
# safe_remove() {
#     for target in "$@"; do
#         [ -e "$target" ] && mv "$target" "/tmp/trash_$$_$(basename "$target")" \
#             && echo "Moved $target to /tmp/trash_$$_$(basename $target)"
#     done
# }
# --- Load modules (adjust names) ---
module load nbi 
module load intel-oneapi/2023.0 # in nbi
module load mkl/2023.0.0 # in nbi intel..
module load mpi/2021.8.0 # in nbi intel..
module load tbb/2021.8.0 compiler-rt/2023.0.0 oclfpga/2023.0.0 # in nbi intel.. to get compiler load possible
module load compiler/2023.0.0

set -euo pipefail
set -x


# --- Paths ---
INSTALL_DIR="${HOME}/lammps_src_Apr2024"
SCRATCH_BUILD="${SCRATCH:-$HOME}/lammps" #refine real path

# --- Clone ---
safe_remove "$INSTALL_DIR"
git clone --depth 1 --branch patch_17Apr2024 \
    https://github.com/lammps/lammps.git "$INSTALL_DIR"
cd "$INSTALL_DIR"
git switch -c my_modified_lammps

# --- Patch REACTION fix ---
safe_remove src/REACTION/fix_bond_react.cpp src/REACTION/fix_bond_react.h
BASE_REACT="https://raw.githubusercontent.com/Adrien-Berard/Berard2026ModifiedLAMMPSFiles/master/fix_bond_react_modified_version"
wget -q "${BASE_REACT}/fix_bond_react.cpp" -O src/REACTION/fix_bond_react.cpp
wget -q "${BASE_REACT}/fix_bond_react.h"   -O src/REACTION/fix_bond_react.h


# --- Could be done on a laptop until this line ---
# just transfer to muon through: rsync -avz lammps_src_Apr2024/ youruser@muon.nbi.dk:/nbi/home/youruser/lammps_src_Apr2024/
# --- Build ---
safe_remove "$SCRATCH_BUILD"
mkdir -p "$SCRATCH_BUILD"
cd "$SCRATCH_BUILD"

cmake \
      -D BUILD_MPI=ON \
      -D CMAKE_CXX_COMPILER=mpicxx \
      -D BUILD_OMP=ON \
      -D PKG_KOKKOS=ON \
      -D Kokkos_ENABLE_OPENMP=ON \
      -D Kokkos_ARCH_ZEN=ON \
      -D FFT_KOKKOS=MKL \
      -D CMAKE_BUILD_TYPE=Release \
      -D PKG_REACTION=ON \
      -D PKG_MOLECULE=ON \
      -D PKG_EXTRA-PAIR=ON \
      ${HOME}/lammps_src_Apr2024/cmake

cmake --build . -j 16

# --- Verify ---
./lmp -h

# --- Add to PATH (idempotent) ---
EXPORT_LINE="export PATH=\$PATH:${SCRATCH_BUILD}"
grep -qxF "$EXPORT_LINE" ${HOME}/.bashrc || echo "$EXPORT_LINE" >> ${HOME}/.bashrc
source ${HOME}/.bashrc
lmp -h
