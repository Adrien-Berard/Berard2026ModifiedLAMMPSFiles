import os
import shutil
import numpy as np


def write_initial_file(n_HP1, dir_path):
    with open(os.path.join(dir_path, 'InitialFile.txt'), 'w') as f:
        f.write('# Initialization file of the particles\n\n')
        
        # Define parameters for polymers
        n_monomers_per_polymer = 80
        n_polymers = 3
        n_ghost = 0
        
        total_monomers = n_monomers_per_polymer * n_polymers
        n_chromatin_bonds = n_monomers_per_polymer - 1
        n_angles = n_monomers_per_polymer - 2
        n_dihedral = n_angles - 1

        f.write(f'{total_monomers + n_HP1 + n_ghost}  atoms\n')
        f.write(f'{n_chromatin_bonds * n_polymers}  bonds\n')

        f.write('5  atom types\n')
        f.write('1  bond types\n\n')

        f.write('0.0000   50.0000 xlo xhi\n')
        f.write('0.0000   50.0000 ylo yhi\n')
        f.write('0.0000   50.0000 zlo zhi\n\n')

        f.write('Atom Type Labels\n\n')
        f.write('1    A\n')
        f.write('2    U\n')
        f.write('3    M\n')
        f.write('4    Swi6\n')
        f.write('5    Swi6M\n\n')

        f.write('Bond Type Labels\n\n')
        f.write('1    Normal\n\n')
        
        f.write('Masses\n\n')
        f.write('1    1.00 \n')
        f.write('2    1.00 \n')
        f.write('3    1.00 \n')
        f.write('4    1.00 \n')
        f.write('5    1.00 \n\n')

        f.write('Atoms\n\n')

        # Assign monomer types for each polymer
        monomer_types = []
        for polymer_idx in range(n_polymers):
            if polymer_idx == 0:
                types = np.random.choice(["A", "U", "M"], n_monomers_per_polymer)
            else:
                types = ["M"] * n_monomers_per_polymer
            monomer_types.extend(types)

        # Generate positions for monomers in three distinct polymers
        for polymer_idx in range(n_polymers):
            start_idx = polymer_idx * n_monomers_per_polymer
            offset = polymer_idx * 15  # Adjusted offset to fit within 50x50x50 box
            molecule_idx = polymer_idx + 1  # Each polymer has a unique molecule index
            for i in range(n_monomers_per_polymer):
                x = (i % 20) * 0.6 + 5 + offset
                y = ((i // 20) % 20) * 0.6 + 5 + offset
                z = ((i // 400) % 20) * 0.6 + 5 + offset
                global_idx = start_idx + i + 1
                f.write(f'  {global_idx}  {molecule_idx}  {monomer_types[start_idx + i]}  {x}  {y}  {z}\n')

        # Generate positions for HP1
        hp1_positions = np.random.uniform(1, 50, (n_HP1, 3))
        for i, (x, y, z) in enumerate(hp1_positions):
            f.write(f'  {total_monomers + i + 1}  {total_monomers + i + 1}  Swi6  {x}  {y}  {z}\n')

        # Generate positions for ghost particles
        if n_ghost > 0:
            x, y, z = np.linspace(1, 50, 50), np.linspace(1, 50, 50), np.linspace(1, 50, 50)
            x, y, z = np.meshgrid(x, y, z)
            x, y, z = x.flatten(), y.flatten(), z.flatten()
            
            for i in range(n_ghost):
                f.write(f'  {total_monomers + n_HP1 + i + 1}  {total_monomers + n_HP1 + i + 1}  G  {x[i]}  {y[i]}  {z[i]}\n')

        f.write('\nBonds\n\n')
        for polymer_idx in range(n_polymers):
            start_idx = polymer_idx * n_monomers_per_polymer
            for i in range(n_chromatin_bonds):
                f.write(f'  {polymer_idx * n_chromatin_bonds + i + 1}  1  {start_idx + i + 1}  {start_idx + i + 2}\n')

# Define ranges for parameters
p2_values = [2.5e-4]
noise_values = [500]
swi6numbers_values = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

# List of files to copy to each simulation directory
files_to_copy = [
    "AM_post-reaction.template",
    "AM_pre-reaction.template",
    "analyse_ouputfile.ipynb",
    "analyzeDensityProfile.py",
    "ASwi6M_post-reaction.template",
    "ASwi6M_pre-reaction.template",
    "AU_post-reaction.template",
    "AU_pre-reaction copy.template",
    "AU_pre-reaction.template",
    "clusters.ipynb",
    "contactMatrix.ipynb",
    "create_initfile.ipynb",
    "density_profile_analysis.ipynb",
    "id_types.ipynb",
    "input.lammps",
    "log.lammps",
    "MA_post-reaction.template",
    "MA_pre-reaction.template",
    "MU_post-reaction.template",
    "MU_pre-reaction.template",
    "output.log",
    "restart.equil",
    "simple.map",
    "Swi6toSwi6M_post-reaction.template",
    "Swi6toSwi6M_pre-reaction.template",
    "Types.ipynb",
    "updated_prob_analysis,.ipynb"
]

base_dir = "3DModel/MorseSmoothLinear/VariationSwi6Concentration"  # Set this to your actual base directory

# Ensure base_dir exists
if not os.path.exists(base_dir):
    raise FileNotFoundError(f"Base directory {base_dir} does not exist.")

for noise in noise_values:
    for p2 in p2_values:
        for swi6number in swi6numbers_values:
            # Create a unique directory for each simulation
            dir_name = f"sim_p2_{p2}_noise_{noise}_swi6_{swi6number}"
            dir_path = os.path.join(base_dir, dir_name)
            os.makedirs(dir_path, exist_ok=True)

            # Copy necessary files to the directory
            for file in files_to_copy:
                src = os.path.join(base_dir, file)
                dst = os.path.join(dir_path, file)
                if os.path.exists(src):  # Ensure the source file exists
                    shutil.copy(src, dst)
                else:
                    print(f"Warning: Source file {src} does not exist. Skipping.")

            # Modify the input script with the current p2 and noise values
            input_file_path = os.path.join(dir_path, "input.lammps")
            if os.path.exists(input_file_path):  # Ensure the input file exists
                with open(input_file_path, 'r') as file:
                    data = file.read()
                data = data.replace("variable p2 equal 1e-4", f"variable p2 equal {p2}")
                data = data.replace("variable noise equal 500", f"variable noise equal {noise}")
                with open(input_file_path, 'w') as file:
                    file.write(data)
            else:
                print(f"Warning: Input file {input_file_path} does not exist. Skipping.")

            # Generate InitialFile.txt with varying n_HP1 for each simulation
            # Here, using swi6number as n_HP1
            write_initial_file(n_HP1=swi6number, dir_path=dir_path)