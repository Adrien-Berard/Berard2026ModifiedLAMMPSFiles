import numpy as np
import os
import shutil
import csv

base_dir = TO_CHANGE

# ---------------- PARAMETERS ----------------
swi6_values = [200, 400, 600]
nucleation_polymer_values = [40,80]
deltaK_values = [80]
nevery_values       = [1, 10]
TOTAL_STEPS         = 200_000_000 

sigma = 1
D = 1
dt = 0.001

# template input file
template_file = "input.lammps"

# --------------------------------------------

def rouse_time(N):
    return int((N**2 * sigma**2) / (dt * 3.0 * np.pi**2 * D))

def production_steps(N):
    rt = rouse_time(N)
    return max(TOTAL_STEPS , 10 * rt)


def write_initial_file(filename, N1, N2, n_Swi6):
    total_monomers = N1 + N2
    n_bonds = (N1 - 1) + (N2 - 1)

    with open(filename, 'w') as f:
        f.write("# Auto-generated file\n\n")
        f.write(f"{total_monomers + n_Swi6} atoms\n")
        f.write(f"{n_bonds} bonds\n\n")

        f.write("5 atom types\n")
        f.write("1 bond types\n\n")

        f.write("0 50 xlo xhi\n0 50 ylo yhi\n0 50 zlo zhi\n\n")

        f.write("Atom Type Labels\n\n")
        f.write("1 A\n2 U\n3 M\n4 Swi6\n5 Swi6M\n\n")

        f.write("Bond Type Labels\n\n1 Normal\n\n")

        f.write("Masses\n\n")
        for i in range(1, 6):
            f.write(f"{i} 1.0\n")

        f.write("\nAtoms\n\n")

        random_1 = np.random.choice(['A','U','M'],N1)
        # polymer 1
        for i in range(N1):
            x = 10 + i * 0.6
            f.write(f"{i+1} 1  {random_1[i]} {x} 10 10\n")

        # polymer 2
        offset = N1
        for i in range(N2):
            x = 30 + i * 0.6
            f.write(f"{offset+i+1} 2 M {x} 30 30\n")

        # Swi6
        for i in range(n_Swi6):
            x, y, z = np.random.uniform(0, 50, 3)
            f.write(f"{N1+N2+i+1} {N1+N2+i+1} Swi6 {x} {y} {z}\n")

        f.write("\nBonds\n\n")

        # bonds polymer 1
        for i in range(N1 - 1):
            f.write(f"{i+1} 1 {i+1} {i+2}\n")

        # bonds polymer 2
        for i in range(N2 - 1):
            f.write(f"{N1+i} 1 {N1+i+1} {N1+i+2}\n")


def modify_input(template, output, N1, N2,n_swi6, nevery, n_steps):
    with open(template, 'r') as f:
        content = f.read()

    content = content.replace("variable monomers_deltaK  equal 80",
                              f"variable monomers_deltaK equal {N1}")
    content = content.replace("variable monomers_2 equal 80",
                              f"variable monomers_2 equal {N2}")
    content = content.replace("variable number_swi6 equal 200",
                              f"variable number_swi6 equal {n_swi6}")

    content = content.replace(
        "variable nevery_swi6 equal 10", 
        f"variable nevery_swi6 equal {nevery}"  
    )
    content = content.replace(
        "variable total_steps equal 2000000",
        f"variable total_steps equal {n_steps}"
    )
    with open(output, 'w') as f:
        f.write(content)


# ---------------- MAIN LOOP ----------------
results = []

# List of files to copy to each simulation directory
files_to_copy = [
    "AM_post-reaction.template",
    "AM_pre-reaction.template",
    "ASwi6M_post-reaction.template",
    "ASwi6M_pre-reaction.template",
    "AU_post-reaction.template",
    "AU_pre-reaction copy.template",
    "AU_pre-reaction.template",
    "MA_post-reaction.template",
    "MA_pre-reaction.template",
    "MU_post-reaction.template",
    "MU_pre-reaction.template",
    "simple.map",
    "Swi6toSwi6M_post-reaction.template",
    "Swi6toSwi6M_pre-reaction.template",
    "Types.ipynb"
]

for swi6 in swi6_values:
    os.makedirs(f"Swi6_{swi6}", exist_ok=True)
    for N2 in nucleation_polymer_values:
        for N1 in deltaK_values:
            for nevery in nevery_values:
                folder = f"Swi6_{swi6}/sim_N1_{N1}_N2_{N2}_every_{nevery}"
                os.makedirs(folder, exist_ok=True)
                
                # Copy necessary files to the directory
                for file in files_to_copy:
                    src = os.path.join(base_dir, file)
                    dst = os.path.join(folder, file)
                    if os.path.exists(src):  # Ensure the source file exists
                        shutil.copy(src, dst)
                    else:
                        print(f"Warning: Source file {src} does not exist. Skipping.")

                # write data file
                write_initial_file(f"{folder}/InitialFile.txt", N1, N2, swi6)
                steps = production_steps(max(N1,N2))
                # write input file
                modify_input(template_file, f"{folder}/input.lammps", N1, N2,swi6,nevery,steps)



print("All simulations + CSV generated.")
