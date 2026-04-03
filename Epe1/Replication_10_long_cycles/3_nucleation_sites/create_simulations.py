import os
import shutil

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
    "InitialFile.txt",
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
