git clone https://github.com/lammps/lammps.git modified_lammps_Apr2024
cd modified_lammps_Apr2024/lammps

git checkout patch_17Apr2024

rm -rf build
rm src/REACTION/fix_bond_react.cpp src/REACTION/fix_bond_react.h



cp https://github.com/adrien-berard/modified_lammps_reacter/modified_lammps_Apr2024/fix_bond_react_modified_version/fix_bond_react.cpp src/REACTION/
cp https://github.com/adrien-berard/modified_lammps_reacter/modified_lammps_Apr2024/fix_bond_react_modified_version/fix_bond_react.h src/REACTION/

mkdir build
cd build

cmake -C ../cmake/presets/most.cmake \
      -C ../cmake/presets/nolib.cmake \
      -D PKG_REACTION=ON \
      -D PKG_MOLECULE=ON \
      ../cmake

make -j$(nproc)