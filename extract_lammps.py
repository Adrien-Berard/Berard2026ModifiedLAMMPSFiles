"""
extract_lammps.py
=================
Extracts and saves processed data from LAMMPS simulation output files.
Tailored for chromatin/Swi6 simulations with bond/react fix.

Supported input files (auto-detected per folder):
─────────────────────────────────────────────────
  types1.dat        → types_evolution.csv
                       columns: step, typeA, typeU, typeM, typeSwi6, typeSwi6M [, typeEpe1C]

  reactions.dat     → reactions.csv
                       columns: step, reacAtoU, reacMtoU, reacUtoA, reacUtoM,
                                reacSwi6toSwi6M [, reacSwi6toEpe1c, reacMtoUEpe1C]

  output.log        → thermo.csv     (step, temp, Rg from YAML thermo lines)

  r2.dat            → r2_evolution.csv

  dump.lammpstrj    → contact_matrix.npz  (mean, std, timeseries)
                    → dump_sub.lammpstrj   (subsampled trajectory)

Usage:
──────
  # Single simulation folder
  python extract_lammps.py --root /path/to/sim/

  # Nested parameter scan (recurses into all subfolders containing sim files)
  python extract_lammps.py --root /path/to/scan/ --recursive

  # Only extract specific file types
  python extract_lammps.py --root /path/to/scan/ --recursive --only types reactions

  # Contact matrix: distance cutoff in LJ units (default 1.5 = bead diameter)
  python extract_lammps.py --root /path/to/sim/ --contact-cutoff 1.5

  # Subsampled trajectory: keep every Nth frame (default 10)
  python extract_lammps.py --root /path/to/sim/ --dump-stride 100

  # Write all outputs to a separate directory (mirrors folder structure)
  python extract_lammps.py --root /path/to/scan/ --recursive --outdir /path/to/out/
"""

import argparse
import csv
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# COLUMN NAME MAPS
# Edit these if your fix/react names differ
# ─────────────────────────────────────────────────────────────────────────────

# reactions.dat: f_reacter_chem[1..7] printed by fix myFixReactions
# With Epe1C (6-type system): 7 reactions
REACTION_NAMES_7 = [
    "reacAtoU", "reacMtoU", "reacUtoA", "reacUtoM",
    "reacSwi6toSwi6M", "reacSwi6toEpe1c", "reacMtoUEpe1C",
]
# Without Epe1C (5-type system): 5 reactions
REACTION_NAMES_5 = [
    "reacAtoU", "reacMtoU", "reacUtoA", "reacUtoM", "reacSwi6toSwi6M",
]

# types1.dat: c_myTypes[1..6] printed by fix myfixTypes
TYPE_NAMES_6 = ["typeA", "typeU", "typeM", "typeSwi6", "typeSwi6M", "typeEpe1C"]
TYPE_NAMES_5 = ["typeA", "typeU", "typeM", "typeSwi6", "typeSwi6M"]

# ─────────────────────────────────────────────────────────────────────────────
# FILE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

KNOWN_FILES = {
    "types":     ["types1.dat", "types.dat"],
    "reactions": ["reactions.dat", "reactions.csv"],
    "dump":      ["dump.lammpstrj", "dump.lammpstrj.gz"],
    "log":       ["output.log", "log.lammps", "log.out"],
    "r2":        ["r2.dat", "r2_1.dat"],
}


def detect_files(folder: Path) -> dict:
    found = {}
    for key, candidates in KNOWN_FILES.items():
        for name in candidates:
            p = folder / name
            if p.exists():
                found[key] = p
                break
    return found


def find_simulation_folders(root: Path, recursive: bool) -> list:
    if not recursive:
        return [root] if detect_files(root) else []
    folders = []
    for dirpath, _, _ in os.walk(root):
        folder = Path(dirpath)
        if detect_files(folder):
            folders.append(folder)
    return sorted(folders)


def get_out_dir(sim_folder: Path, root: Path, outdir) -> Path:
    if outdir is not None:
        rel = sim_folder.relative_to(root)
        target = outdir / rel
    else:
        target = sim_folder / "extracted"
    target.mkdir(parents=True, exist_ok=True)
    return target


# ─────────────────────────────────────────────────────────────────────────────
# TYPES1.DAT
# Produced by: fix myfixTypes all print 1000 "${typeA},${typeU},..." file types1.dat
# Format: comma-separated integers, no header, one row per 1000 steps
# ─────────────────────────────────────────────────────────────────────────────

def extract_types(types_path: Path, out_dir: Path):
    log.info(f"  [types] Reading {types_path.name}")
    rows = []
    with open(types_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = [v.strip() for v in re.split(r"[,\s]+", line) if v.strip()]
            try:
                rows.append(list(map(int, vals)))
            except ValueError:
                continue  # skip header-like lines

    if not rows:
        log.warning("  [types] File is empty, skipping.")
        return

    n_cols = len(rows[0])
    if n_cols == 6:
        col_names = TYPE_NAMES_6
    elif n_cols == 5:
        col_names = TYPE_NAMES_5
    else:
        col_names = [f"type_{i+1}" for i in range(n_cols)]
        log.warning(f"  [types] Unexpected column count ({n_cols}), using generic names.")

    # Step numbers reconstructed from print frequency (every 1000 steps)
    steps = [i * 1000 for i in range(len(rows))]

    out_path = out_dir / "types_evolution.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step"] + col_names)
        for step, row in zip(steps, rows):
            writer.writerow([step] + row)

    log.info(f"  [types] Wrote {out_path.name}  "
             f"({len(rows)} rows, columns: {col_names})")


# ─────────────────────────────────────────────────────────────────────────────
# REACTIONS.DAT
# Produced by: fix myFixReactions all print 10000 "${reactionsIDs1},..." file reactions.dat
# Format: comma-separated cumulative counts, trailing comma, no header
# Values are f_reacter_chem[N] = cumulative count since simulation start
# ─────────────────────────────────────────────────────────────────────────────

def extract_reactions(reactions_path: Path, out_dir: Path):
    log.info(f"  [reactions] Reading {reactions_path.name}")
    rows = []
    with open(reactions_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Split on comma or whitespace, filter empty strings (trailing comma)
            vals = [v.strip() for v in re.split(r"[,\s]+", line) if v.strip()]
            try:
                rows.append(list(map(float, vals)))
            except ValueError:
                continue

    if not rows:
        log.warning("  [reactions] File is empty, skipping.")
        return

    n_cols = len(rows[0])
    if n_cols == 7:
        col_names = REACTION_NAMES_7
    elif n_cols == 5:
        col_names = REACTION_NAMES_5
    else:
        col_names = [f"reaction_{i+1}" for i in range(n_cols)]
        log.warning(f"  [reactions] Unexpected column count ({n_cols}), "
                    f"using generic names. Expected 5 or 7.")

    # Step numbers from print frequency (every 10000 steps)
    steps = [i * 10000 for i in range(len(rows))]

    # Compute per-interval deltas (reaction events per window — useful for rate plots)
    cumulative = np.array(rows)
    deltas = np.diff(cumulative, axis=0, prepend=cumulative[[0]])
    deltas[0] = cumulative[0]

    out_path = out_dir / "reactions.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["step"] +
            [f"{n}_cumulative" for n in col_names] +
            [f"{n}_per_window" for n in col_names]
        )
        for i, (step, cum_row) in enumerate(zip(steps, rows)):
            writer.writerow(
                [step] +
                [int(v) for v in cum_row] +
                [int(v) for v in deltas[i]]
            )

    log.info(f"  [reactions] Wrote {out_path.name}  "
             f"({len(rows)} timepoints, {n_cols} reactions)")


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT.LOG  — YAML thermo format
# thermo_style custom step temp c_R2[0]
# thermo_modify line yaml
#
# Each logged block looks like:
#   ---
#   - keywords: step temp c_R2[0]
#     data: 10000 0.9987 12.345
# ─────────────────────────────────────────────────────────────────────────────

def extract_log(log_path: Path, out_dir: Path):
    log.info(f"  [log] Reading {log_path.name}")

    thermo_rows   = []
    thermo_header = None
    current_keys  = None

    keyword_re = re.compile(r"^\s*-\s*keywords:\s*(.+)$")
    data_re    = re.compile(r"^\s*data:\s*(.+)$")

    with open(log_path, errors="replace") as f:
        for line in f:
            m = keyword_re.match(line)
            if m:
                current_keys = m.group(1).split()
                if thermo_header is None:
                    thermo_header = current_keys
                continue

            m = data_re.match(line)
            if m and current_keys is not None:
                try:
                    vals = list(map(float, m.group(1).split()))
                    if len(vals) == len(current_keys):
                        thermo_rows.append(vals)
                except ValueError:
                    pass
                current_keys = None

    if not thermo_rows:
        log.warning("  [log] No YAML thermo data found. "
                    "Confirm 'thermo_modify line yaml' is in your input file.")
        return

    # Clean column names for CSV readability
    clean_header = []
    for col in thermo_header:
        col = col.replace("c_R2[0]", "Rg_gyration")
        col = re.sub(r"\[(\d+)\]", r"_\1", col)  # c_foo[2] → c_foo_2
        clean_header.append(col)

    thermo_path = out_dir / "thermo.csv"
    with open(thermo_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(clean_header)
        writer.writerows(thermo_rows)

    log.info(f"  [log] Wrote {thermo_path.name}  "
             f"({len(thermo_rows)} thermo entries, columns: {clean_header})")


# ─────────────────────────────────────────────────────────────────────────────
# R2.DAT
# Produced by: fix myfixR2 chromatin ave/time 10 100 1000 c_R2[0] file r2.dat
# Format: standard LAMMPS ave/time header (# lines) then two columns: step, Rg
# ─────────────────────────────────────────────────────────────────────────────

def extract_r2(r2_path: Path, out_dir: Path):
    log.info(f"  [r2] Reading {r2_path.name}")

    rows = []
    header = None

    with open(r2_path) as f:
        for line in f:
            line_s = line.strip()
            if not line_s:
                continue
            if line_s.startswith("#"):
                parts = line_s.lstrip("#").split()
                # LAMMPS ave/time header line 2: "TimeStep v_..."
                if parts and parts[0].lower() in ("timestep", "step"):
                    header = ["step", "Rg_gyration"]
                continue
            try:
                vals = list(map(float, line_s.split()))
                rows.append(vals)
            except ValueError:
                continue

    if not rows:
        log.warning("  [r2] File is empty, skipping.")
        return

    n_cols = len(rows[0])
    if header is None or len(header) != n_cols:
        header = ["step"] + [f"col_{i}" for i in range(n_cols - 1)]

    out_path = out_dir / "r2_evolution.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    log.info(f"  [r2] Wrote {out_path.name}  ({len(rows)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# DUMP.LAMMPSTRJ
# dump mydmp all atom 10000 dump.lammpstrj
# Atom style: id type xs ys zs (or xu yu zu for unwrapped)
# ─────────────────────────────────────────────────────────────────────────────

def iter_dump_frames(dump_path: Path):
    """Generator: yields (timestep, box, col_names, atoms_array) per frame."""
    import gzip
    open_fn = gzip.open if str(dump_path).endswith(".gz") else open

    with open_fn(dump_path, "rt") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if "TIMESTEP" not in line:
                continue
            timestep = int(f.readline().strip())

            f.readline()  # ITEM: NUMBER OF ATOMS
            n_atoms = int(f.readline().strip())

            f.readline()  # ITEM: BOX BOUNDS ...
            box = [list(map(float, f.readline().split())) for _ in range(3)]

            atoms_line = f.readline().strip()  # ITEM: ATOMS id type xs ys zs ...
            col_names = atoms_line.replace("ITEM: ATOMS", "").split()

            atoms = np.array([
                list(map(float, f.readline().split()))
                for _ in range(n_atoms)
            ])
            yield timestep, box, col_names, atoms


def extract_dump(dump_path: Path, out_dir: Path,
                 contact_cutoff: float = 1.5,
                 dump_stride: int = 10,
                 n_contact_frames: int = 52):
    """
    From dump.lammpstrj produce:
      1. contact_matrix.npz  — keys: mean (N,N), std (N,N),
                                      timeseries (T,N,N), frame_indices, timesteps, cutoff
      2. dump_sub.lammpstrj  — every dump_stride-th frame (for Ovito)
    """
    log.info(f"  [dump] Reading {dump_path.name}  (may take a while...)")

    frames = []
    for ts, box, col_names, atoms in iter_dump_frames(dump_path):
        frames.append((ts, box, col_names, atoms))

    n_frames = len(frames)
    if n_frames == 0:
        log.warning("  [dump] No frames found.")
        return

    n_atoms = frames[0][3].shape[0]
    log.info(f"  [dump] {n_frames} frames × {n_atoms} atoms")

    col_names = frames[0][2]
    try:
        id_idx   = col_names.index("id")
        type_idx = col_names.index("type")
        x_idx = col_names.index("xu") if "xu" in col_names else col_names.index("xs")
        y_idx = col_names.index("yu") if "yu" in col_names else col_names.index("ys")
        z_idx = col_names.index("zu") if "zu" in col_names else col_names.index("zs")
    except ValueError as e:
        log.error(f"  [dump] Missing column: {e}. Found: {col_names}")
        return

    # ── Contact matrix ────────────────────────────────────────────────────────
    contact_indices = np.linspace(
        0, n_frames - 1, min(n_contact_frames, n_frames), dtype=int
    )
    contact_ts = np.zeros((len(contact_indices), n_atoms, n_atoms), dtype=np.float32)

    for out_i, frame_i in enumerate(contact_indices):
        _, _, _, atoms = frames[frame_i]
        coords = atoms[:, [x_idx, y_idx, z_idx]]
        diff   = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist   = np.sqrt((diff ** 2).sum(axis=-1))
        contact_ts[out_i] = (dist < contact_cutoff).astype(np.float32)
        np.fill_diagonal(contact_ts[out_i], 0)

    npz_path = out_dir / "contact_matrix.npz"
    np.savez_compressed(
        npz_path,
        mean=contact_ts.mean(axis=0),
        std=contact_ts.std(axis=0),
        timeseries=contact_ts,
        frame_indices=contact_indices,
        timesteps=np.array([frames[i][0] for i in contact_indices]),
        cutoff=np.float32(contact_cutoff),
    )
    log.info(f"  [dump] Wrote {npz_path.name}  "
             f"(shape {contact_ts.shape}, cutoff={contact_cutoff})")
    log.info("  [dump]   Usage: d=np.load('contact_matrix.npz'); "
             "mean=d['mean']  # (N,N)")

    # ── Subsampled trajectory ─────────────────────────────────────────────────
    sub_path  = out_dir / "dump_sub.lammpstrj"
    sub_count = 0

    with open(sub_path, "w") as f:
        for i, (ts, box, col_names_frame, atoms) in enumerate(frames):
            if i % dump_stride != 0:
                continue
            sub_count += 1
            f.write("ITEM: TIMESTEP\n")
            f.write(f"{ts}\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{n_atoms}\n")
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            for lo, hi in box:
                f.write(f"{lo:.6f} {hi:.6f}\n")
            f.write("ITEM: ATOMS " + " ".join(col_names_frame) + "\n")
            for row in atoms:
                parts = [
                    str(int(v)) if j in (id_idx, type_idx) else f"{v:.6f}"
                    for j, v in enumerate(row)
                ]
                f.write(" ".join(parts) + "\n")

    log.info(f"  [dump] Wrote {sub_path.name}  "
             f"({sub_count} frames, stride={dump_stride})")


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────

def process_folder(sim_folder: Path, root: Path, args):
    files = detect_files(sim_folder)
    if not files:
        return

    out_dir = get_out_dir(sim_folder, root, args.outdir)
    label = sim_folder.relative_to(root) if sim_folder != root else sim_folder.name
    log.info(f"\n{'─'*60}")
    log.info(f"Folder : {label}")
    log.info(f"Found  : {', '.join(files.keys())}")
    log.info(f"Output → {out_dir}")

    only = set(args.only) if args.only else set(files.keys())

    tasks = [
        ("types",     extract_types,     [files.get("types"),     out_dir]),
        ("reactions", extract_reactions,  [files.get("reactions"), out_dir]),
        ("log",       extract_log,        [files.get("log"),       out_dir]),
        ("r2",        extract_r2,         [files.get("r2"),        out_dir]),
    ]
    for key, fn, fn_args in tasks:
        if key in files and key in only:
            try:
                fn(*fn_args)
            except Exception as e:
                log.error(f"  [{key}] Failed: {e}", exc_info=args.verbose)

    if "dump" in files and "dump" in only:
        try:
            extract_dump(
                files["dump"], out_dir,
                contact_cutoff=args.contact_cutoff,
                dump_stride=args.dump_stride,
                n_contact_frames=args.contact_frames,
            )
        except Exception as e:
            log.error(f"  [dump] Failed: {e}", exc_info=args.verbose)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract processed data from LAMMPS chromatin/Swi6 simulations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--root", type=Path, required=True,
                   help="Simulation folder or top of parameter scan tree")
    p.add_argument("--recursive", action="store_true",
                   help="Recurse into subfolders (parameter scan mode)")
    p.add_argument("--outdir", type=Path, default=None,
                   help="Root for outputs. Default: extracted/ inside each sim folder")
    p.add_argument("--only", nargs="+",
                   choices=["types", "reactions", "log", "r2", "dump"],
                   help="Only process these file types")
    p.add_argument("--dump-stride", type=int, default=10,
                   help="Keep every Nth dump frame in dump_sub (default: 10)")
    p.add_argument("--contact-frames", type=int, default=52,
                   help="Number of evenly spaced frames for contact matrix (default: 52)")
    p.add_argument("--contact-cutoff", type=float, default=1.5,
                   help="Contact distance cutoff in LJ units (default: 1.5)")
    p.add_argument("--verbose", action="store_true",
                   help="Show full tracebacks on errors")
    return p.parse_args()


def main():
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    if not args.root.exists():
  
