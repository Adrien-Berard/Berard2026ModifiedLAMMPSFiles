#!/usr/bin/env bash
set -e  # stop on error
set -x  # print every command before executing it

# --- Clone LAMMPS at the specific tag ---
rm -rf modified_lammps_Apr2024

git clone --depth 1 --branch patch_17Apr2024 \
    https://github.com/lammps/lammps.git modified_lammps_Apr2024

cd modified_lammps_Apr2024
git switch -c my_modified_lammps

# --- Clean previous builds ---
rm -rf build

# --- Replace REACTION fix ---
rm -f src/REACTION/fix_bond_react.cpp
rm -f src/REACTION/fix_bond_react.h

base_react="https://raw.githubusercontent.com/Adrien-Berard/Berard2026ModifiedLAMMPSFiles/master/fix_bond_react_modified_version"
wget -q "${base_react}/fix_bond_react.cpp" -O src/REACTION/fix_bond_react.cpp
wget -q "${base_react}/fix_bond_react.h"   -O src/REACTION/fix_bond_react.h

# --- USER-LE package (loop extrusion) ---
wget -q https://raw.githubusercontent.com/polly-code/lammps_le/main/src/compute_cb.cpp \
     -O src/compute_cb.cpp
wget -q https://raw.githubusercontent.com/polly-code/lammps_le/main/src/compute_cb.h \
     -O src/compute_cb.h

mkdir -p src/USER-LE


files=(
  fix_ex_load.cpp
  fix_ex_load.h
  fix_ex_unload.cpp
  fix_ex_unload.h
  fix_extrusion.cpp
  fix_extrusion.h
)

for f in "${files[@]}"; do
  wget -q "https://raw.githubusercontent.com/polly-code/lammps_le/main/src/USER-LE/${f}" -O "src/USER-LE/${f}"
done

# --- Build ---
mkdir build
cd build

cmake -C ../cmake/presets/most.cmake \
      -C ../cmake/presets/nolib.cmake \
      -D BUILD_MPI=ON \
      -D PKG_REACTION=ON \
      -D PKG_MOLECULE=ON \
      -D PKG_MISC=ON \
      -D PKG_MPIIO=ON \
      -D PKG_MC=ON \
      -D LAMMPS_EXTRA_SOURCE_DIR=../src/USER-LE \
      ../cmake

cmake --build . -j$(nproc)

# --- Verify ---
./lmp -h | grep USER

# --- Add lmp to PATH ---
echo "export PATH=\$PATH:/home/adrien/modified_lammps_Apr2024/build" >> ~/.bashrc  # Change your path accordingly
source ~/.bashrc
lmp -h
