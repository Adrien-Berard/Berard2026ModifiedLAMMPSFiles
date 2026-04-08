"""
extract_lammps_github.py
=================
Extracts and saves scripts for LAMMPS simulation intput files and, if here, post-processing.
Tailored for chromatin/Swi6 simulations with bond/react fix.

Supported input files (auto-detected per folder):
─────────────────────────────────────────────────
  *.template        → necessary template files for fix bond/react
  
  *.map             → necessary map files for fix bond/react
  
  InitialFile.txt   → generally used for read_data function of any *.lammps file (here input.lammps)

  input.lammps      → save the motherboard of any lammps simulation

  r2.dat            → r2_evolution.csv

  dump.lammpstrj    → contact_matrix.npz  (mean, std, timeseries)
                    → dump_sub.lammpstrj   (subsampled trajectory)

  *.py              → python scripts

  *.ipynb           → python notebooks


Usage:
──────
  # Single simulation folder
  python extract_lammps_github.py --root /path/to/sim/

  # Nested parameter scan (recurses into all subfolders containing sim files)
  python extract_lammps_github.py --root /path/to/scan/ --recursive

  # Write all outputs to a separate directory (mirrors folder structure)
  python extract_lammps_github.py --root /path/to/scan/ --recursive --outdir /path/to/out/
"""

import argparse
import logging
import os
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# FILE LIST (safe: only copied if present)
# ─────────────────────────────────────────────────────────────

STATIC_FILES = [
    "AM_post-reaction.template",
    "AM_pre-reaction.template",
    "ASwi6M_post-reaction.template",
    "ASwi6M_pre-reaction.template",
    "AU_post-reaction.template",
    "AU_pre-reaction.template",
    "MA_post-reaction.template",
    "MA_pre-reaction.template",
    "MU_post-reaction.template",
    "MU_pre-reaction.template",
    "Swi6toSwi6M_post-reaction.template",
    "Swi6toSwi6M_pre-reaction.template",
    "Swi6toEpe1C_pre-reaction.template",
    "Swi6toEpe1C_post-reaction.template",
    "MtoUEpe1C_pre-reaction.template",
    "MtoUEpe1C_post-reaction.template",
    "Epe1CtoSwi6_pre-reaction.template",
    "Epe1CtoSwi6_post-reaction.template",
    "InitialFile.txt",
    "input.lammps",
    "simple.map",
]

# ─────────────────────────────────────────────────────────────
# DETECTION
# ─────────────────────────────────────────────────────────────

def is_sim_folder(folder: Path):
    """Loose check: contains at least one relevant file"""
    for f in STATIC_FILES:
        if (folder / f).exists():
            return True
    # also accept folders with scripts
    if list(folder.glob("*.py")) or list(folder.glob("*.ipynb")):
        return True
    return False


def find_folders(root: Path, recursive: bool):
    if not recursive:
        return [root] if is_sim_folder(root) else []

    folders = []
    for dirpath, _, _ in os.walk(root):
        p = Path(dirpath)
        if is_sim_folder(p):
            folders.append(p)
    return sorted(folders)


def get_outdir(sim_folder: Path, root: Path, outdir: Path):
    rel = sim_folder.relative_to(root)
    target = outdir / rel / "src"
    target.mkdir(parents=True, exist_ok=True)
    return target


# ─────────────────────────────────────────────────────────────
# COPY LOGIC
# ─────────────────────────────────────────────────────────────

def copy_files(sim_folder: Path, out_src: Path):
    copied = 0

    # 1. Copy static known files
    for fname in STATIC_FILES:
        src = sim_folder / fname
        if src.exists():
            shutil.copy2(src, out_src / fname)
            copied += 1

    # 2. Copy all .py and .ipynb
    for pattern in ("*.py", "*.ipynb"):
        for file in sim_folder.glob(pattern):
            shutil.copy2(file, out_src / file.name)
            copied += 1

    return copied


# ─────────────────────────────────────────────────────────────
# MAIN PROCESS
# ─────────────────────────────────────────────────────────────

def process_folder(folder: Path, root: Path, outdir: Path):
    out_src = get_outdir(folder, root, outdir)

    log.info(f"\nFolder : {folder.relative_to(root)}")
    log.info(f"Output → {out_src}")

    try:
        n = copy_files(folder, out_src)
        log.info(f"Copied {n} files")
    except Exception as e:
        log.error(f"Failed: {e}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare clean GitHub src/ folders from LAMMPS simulations"
    )
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--recursive", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.root.exists():
        log.error(f"Root not found: {args.root}")
        return

    folders = find_folders(args.root, args.recursive)

    if not folders:
        log.error("No valid folders found.")
        return

    log.info(f"Found {len(folders)} folders")

    for f in folders:
        process_folder(f, args.root, args.outdir)

    log.info("\nDone.")


if __name__ == "__main__":
    main()
