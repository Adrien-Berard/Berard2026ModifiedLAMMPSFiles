# bash run input.lammps file on octopod current build (in your folder)

export OMP_NUM_THREADS=24
export OMP_PROC_BIND=close
export OMP_PLACES=cores

mpirun -np 2 ./lmp -sf kk -in input.lammps > output.log
