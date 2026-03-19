rm -rf modified_lammps_Apr2024

#!/usr/bin/env bash
set -e  # stop on error

# --- Clone LAMMPS ---
git clone --depth 1 --branch patch_17Apr2024 https://github.com/lammps/lammps.git modified_lammps_Apr2024
cd modified_lammps_Apr2024

git switch --detach patch_17Apr2024

# --- Clean previous builds ---
rm -rf build

# --- Replace REACTION fix ---
rm -f src/REACTION/fix_bond_react.cpp
rm -f src/REACTION/fix_bond_react.h

wget -q https://raw.githubusercontent.com/adrien-berard/modified_lammps_reacter/master/modified_lammps_Apr2024/fix_bond_react_modified_version/fix_bond_react.cpp -O src/REACTION/fix_bond_react.cpp
wget -q https://raw.githubusercontent.com/adrien-berard/modified_lammps_reacter/master/modified_lammps_Apr2024/fix_bond_react_modified_version/fix_bond_react.h -O src/REACTION/fix_bond_react.h

# --- USER-LE package (loop extrusion) ---
# NOTE: must download actual files, not directory

rm -f src/compute_cb.cpp
rm -f src/compute_cb.h

wget https://raw.githubusercontent.com/polly-code/lammps_le/main/src/compute_cb.cpp
wget https://raw.githubusercontent.com/polly-code/lammps_le/main/src/compute_cb.h

mkdir -p src/USER-LE

base_url="https://raw.githubusercontent.com/polly-code/lammps_le/main/src/USER-LE"

files=(
  fix_ex_load.cpp
  fix_ex_load.cpp
  fix_ex_load.cpp
  fix_ex_load.cpp
  fix_extrusion.cpp
  fix_extrusion.h
)

for f in "${files[@]}"; do
  wget -q "${base_url}/${f}" -O "src/USER-LE/${f}"
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
