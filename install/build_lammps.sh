rm -rf modified_lammps_Apr2024

git clone https://github.com/lammps/lammps.git modified_lammps_Apr2024
cd modified_lammps_Apr2024

git checkout patch_17Apr2024

rm -rf build
rm -f src/REACTION/fix_bond_react.cpp
rm -f src/REACTION/fix_bond_react.h

wget https://raw.githubusercontent.com/adrien-berard/modified_lammps_reacter/master/modified_lammps_Apr2024/fix_bond_react_modified_version/fix_bond_react.cpp -P src/REACTION/
wget https://raw.githubusercontent.com/adrien-berard/modified_lammps_reacter/master/modified_lammps_Apr2024/fix_bond_react_modified_version/fix_bond_react.h -P src/REACTION/

# Build for loop extrusion module (to comment if not using)
wget https://raw.githubusercontent.com/polly-code/lammps_le/tree/6e6c3cf461ca75c3755cc7fdf7c9891934d69dc7/src/USER-LE -P src/

mkdir build
cd build

cmake -C ../cmake/presets/most.cmake \
      -C ../cmake/presets/nolib.cmake \
      -D PKG_REACTION=ON \
      -D PKG_MOLECULE=ON \
      -D -D PKG_MISC=ON \
      -D PKG_USER-LE=ON \
      -D PKG_MPIIO=ON \
      -D PKG_MOLECULE=ON \
      -D PKG_MC=ON
      ../cmake

make -j$(nproc)
