#!/usr/bin/env bash
set -euo pipefail
set -x

# --- Load modules (adjust names) ---
module load nbi intel-oneapi-mpi/2023.0 mkl/2023.0.0
module load nbi intel-oneapi-mpi/2023.0 mkl/2023.0.0 mpi/2021.8.0

# --- Paths ---
INSTALL_DIR="${HOME}/modified_lammps_Apr2024"
SCRATCH_BUILD="${SCRATCH:-$HOME}/lammps"

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

# --- Build ---
safe_remove "$SCRATCH_BUILD"
mkdir -p "$SCRATCH_BUILD"
cd "$SCRATCH_BUILD"

cmake -C "$INSTALL_DIR/cmake/presets/most.cmake" \
      -C "$INSTALL_DIR/cmake/presets/nolib.cmake" \
      -D BUILD_MPI=ON \
      -D CMAKE_CXX_COMPILER=mpicxx \
      -D BUILD_OMP=ON \
      -D PKG_KOKKOS=ON \
      -D Kokkos_ENABLE_OPENMP=ON \
      -D Kokkos_ARCH_ZEN3=ON \
      -D FFT_KOKKOS=MKL \
      -D CMAKE_BUILD_TYPE=Release \
      -D PKG_REACTION=ON \
      -D PKG_MOLECULE=ON \
      "$INSTALL_DIR/cmake"

cmake --build . -j 16

# --- Verify ---
./lmp -h

# --- Add to PATH (idempotent) ---
EXPORT_LINE="export PATH=\$PATH:${SCRATCH_BUILD}"
grep -qxF "$EXPORT_LINE" ${HOME}/.bashrc || echo "$EXPORT_LINE" >> ${HOME}/.bashrc
source ${HOME}/.bashrc
lmp -h
