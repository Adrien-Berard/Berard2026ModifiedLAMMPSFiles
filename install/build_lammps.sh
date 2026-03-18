rm -rf modified_lammps_Apr2024

git clone https://github.com/lammps/lammps.git modified_lammps_Apr2024
cd modified_lammps_Apr2024

git checkout patch_17Apr2024

rm -rf build
rm -f src/REACTION/fix_bond_react.cpp
rm -f src/REACTION/fix_bond_react.h

wget https://raw.githubusercontent.com/adrien-berard/modified_lammps_reacter/main/modified_lammps_Apr2024/fix_bond_react_modified_version/fix_bond_react.cpp -P src/REACTION/

wget https://raw.githubusercontent.com/adrien-berard/modified_lammps_reacter/main/modified_lammps_Apr2024/fix_bond_react_modified_version/fix_bond_react.h -P src/REACTION/

mkdir build
cd build

cmake -C ../cmake/presets/most.cmake \
      -C ../cmake/presets/nolib.cmake \
      -D PKG_REACTION=ON \
      -D PKG_MOLECULE=ON \
      ../cmake

make -j$(nproc)
