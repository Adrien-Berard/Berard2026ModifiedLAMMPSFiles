This repository allows the build of a modified version of LAMMPS
(17 April 2024 / patch_17Apr2024).

LAMMPS is developed by Sandia National Laboratories.
Original project:
https://github.com/lammps/lammps

Modifications in this repository:
- modified fix_bond_react.cpp
- modified fix_bond_react.h
- using USER-LE package from polly-code to add boundary elements
- additional simulation scripts

# Example of use for USER-LE package, quoted from polly-code repo
- fix loop all extrusion 17500 1 2 3 1.0 2 4

Here is standard LAMMPS syntax:
fix - keyword
loop - the name of the fix
all - group of particles to which we apply this fix
extrusion - specific name of the fix
17500 - N1, amount of steps we try to shift links
1 - the type of neutral type monomers from the chain
2 - left barrier
3 - right barrier
1.0 - the probability of going through the block
2 - the bond type of the extruder
4 - the type of roadblocks

To run the code, one needs proper loading and unloading of extruders. There are two more fixes:

fix loading all ex_load 7000 1 1 1.12 2 prob 0.001 684474 iparam 1 1 jparam 1 1
fix unloading all ex_unload 7000 2 0.5 prob 0.001 456456

Their syntax is similar to creating and breaking bonds from the MC package from the LAMMPS website.
https://docs.lammps.org/fix_bond_create.html
https://docs.lammps.org/fix_bond_break.html
