#!/usr/bin/env bash
set -e  # stop on error
set -x  # print every command before executing it

# --- Clone LAMMPS at the specific tag ---
# rm -rf modified_lammps_Apr2024
trash modified_lammps_Apr2024

git clone --depth 1 --branch patch_17Apr2024 \
    https://github.com/lammps/lammps.git modified_lammps_Apr2024

cd modified_lammps_Apr2024
git switch -c my_modified_lammps

# --- Clean previous builds ---
# rm -rf build
trash build

# --- Replace REACTION fix ---
# rm -f src/REACTION/fix_bond_react.cpp
# rm -f src/REACTION/fix_bond_react.h
trash src/REACTION/fix_bond_react.cpp
trash src/REACTION/fix_bond_react.h

base_react="https://raw.githubusercontent.com/Adrien-Berard/Berard2026ModifiedLAMMPSFiles/master/fix_bond_react_modified_version"
wget -q "${base_react}/fix_bond_react.cpp" -O src/REACTION/fix_bond_react.cpp
wget -q "${base_react}/fix_bond_react.h"   -O src/REACTION/fix_bond_react.h


# --- Build ---
mkdir build
cd build

cmake -C ../cmake/presets/most.cmake \
      -C ../cmake/presets/nolib.cmake \
      cmake ../cmake \
      -D BUILD_MPI=ON \
      -D CMAKE_CXX_COMPILER=/nbi/software/intel-oneapi/2023.0.0.25537/mpi/2021.8.0/bin/mpicxx \
      -D BUILD_OMP=ON \
      -D PKG_KOKKOS=ON \
      -D Kokkos_ENABLE_OPENMP=ON \
      -D Kokkos_ARCH_ZEN3=ON \
      -D FFT_KOKKOS=MKL \
      -D CMAKE_BUILD_TYPE=Release
      -D PKG_REACTION=ON \
      -D PKG_MOLECULE=ON \
      ../cmake

cmake --build . -j 16

# --- Verify ---
./lmp -h

# --- Add lmp to PATH ---
echo "export PATH=\$PATH:/home/adrien/modified_lammps_Apr2024/build" >> ~/.bashrc  # Change your path accordingly
source ~/.bashrc
lmp -h

