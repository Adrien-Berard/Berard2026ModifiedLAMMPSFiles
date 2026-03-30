"""
plot_polymers.py
----------------
Reads the LAMMPS custom dump file (id_and_type.dat) produced by:
    dump myDump chromatin custom 10000 id_and_type.dat id type

Generates a figure with:
  1. Two time-series panels (one per polymer) — count of each type vs. timestep
  2. One stacked kymograph — both polymers (bead index vs. timestep),
     separated by a black horizontal line between bead 80 and 81.

Atom types:  A=1 (blue)  |  U=2 (yellow)  |  M=3 (red)
Polymers:    Polymer 1 -> IDs  1-80
             Polymer 2 -> IDs 81-160

Usage:
    python plot_polymers.py                        # uses default filename
    python plot_polymers.py my_dump.dat            # custom filename
    python plot_polymers.py --out figure.png       # save instead of show
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

# -- CONFIG -------------------------------------------------------------------
DUMP_FILE  = "id_and_type.dat"

# Define polymers (list of tuples). You can give 1 or 2.
POLYMERS = [
    (1, 80)
    #(81, 160),   # remove this line if only one polymer
]

TS_STRIDE   = 1   # time series downsampling
KYMO_STRIDE = 1    # kymograph downsampling

TYPE_COLORS = {
    1: "#2166AC",   # A - blue
    2: "#F4C300",   # U - yellow
    3: "#D6001C",   # M - red
}
TYPE_LABELS = {1: "A (type 1)", 2: "U (type 2)", 3: "M (type 3)"}
TYPE_CMAP   = mcolors.ListedColormap([TYPE_COLORS[k] for k in sorted(TYPE_COLORS)])

# -- PARSER -------------------------------------------------------------------

def parse_dump(filepath):
    """
    Parse a LAMMPS custom dump with columns: id  type
    Returns:
        timesteps : list of int  (raw LAMMPS step numbers)
        frames    : list of dict {atom_id (int): atom_type (int)}
    """
    timesteps, frames = [], []
    current_ts = None
    n_atoms = 0

    with open(filepath) as fh:
        lines = fh.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line == "ITEM: TIMESTEP":
            current_ts = int(lines[i + 1].strip())
            i += 2

        elif line == "ITEM: NUMBER OF ATOMS":
            n_atoms = int(lines[i + 1].strip())
            i += 2

        elif line.startswith("ITEM: ATOMS"):
            header   = line.split()[2:]
            col_id   = header.index("id")
            col_type = header.index("type")
            frame = {}
            for _ in range(n_atoms):
                i += 1
                parts = lines[i].strip().split()
                frame[int(parts[col_id])] = int(parts[col_type])
            timesteps.append(current_ts)
            frames.append(frame)
            i += 1
        else:
            i += 1

    return timesteps, frames

# -- PROCESSING ---------------------------------------------------------------

def build_arrays(frames, polymers):
    """Build arrays for arbitrary number of polymers."""
    ids_list = [list(range(p[0], p[1] + 1)) for p in polymers]
    n = len(frames)

    arrays = [
        np.zeros((n, len(ids)), dtype=np.int32)
        for ids in ids_list
    ]

    for t, frame in enumerate(frames):
        for p_idx, ids in enumerate(ids_list):
            for j, aid in enumerate(ids):
                arrays[p_idx][t, j] = frame.get(aid, 0)

    return ids_list, arrays


def compute_counts(arr):
    """Return dict type -> count array of shape (n_frames,)."""
    return {t: np.sum(arr == t, axis=1) for t in [1, 2, 3]}

# -- PLOTTING -----------------------------------------------------------------

def plot_all(timesteps, ids_list, arrays, outfile=None):
    ts = np.array(timesteps)

    # ---- Downsampling ----
    ts_ts   = ts[::TS_STRIDE]
    ts_kymo = ts[::KYMO_STRIDE]

    arrays_ts   = [arr[::TS_STRIDE] for arr in arrays]
    arrays_kymo = [arr[::KYMO_STRIDE] for arr in arrays]

    counts_list = [compute_counts(arr) for arr in arrays_ts]

    bounds = [0.5, 1.5, 2.5, 3.5]
    norm   = mcolors.BoundaryNorm(bounds, TYPE_CMAP.N)

    n_polymers = len(arrays)

    fig = plt.figure(figsize=(15, 10 + 2*n_polymers))
    gs  = fig.add_gridspec(n_polymers + 1, 1,
                           height_ratios=[1]*n_polymers + [1.8],
                           hspace=0.45)

    # ---- Time series ----
    for i, (counts, ids) in enumerate(zip(counts_list, ids_list)):
        ax = fig.add_subplot(gs[i])
        for t in [1, 2, 3]:
            ax.plot(ts_ts, counts[t],
                    color=TYPE_COLORS[t],
                    label=TYPE_LABELS[t],
                    linewidth=1.6)

        ax.set_title(f"Polymer {i+1} (IDs {ids[0]}-{ids[-1]})", fontsize=11)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    # ---- Kymograph ----
    ax_kymo = fig.add_subplot(gs[-1])

    y_offset = 0
    for i, (arr, ids) in enumerate(zip(arrays_kymo, ids_list)):
        n_beads = arr.shape[1]

        ax_kymo.imshow(
            arr.T,
            aspect="auto",
            origin="upper",
            cmap=TYPE_CMAP,
            norm=norm,
            interpolation="nearest",
            extent=[ts_kymo[0], ts_kymo[-1],
                    y_offset + n_beads + 0.5, y_offset + 0.5]
        )

        # separator line (except last)
        if i < n_polymers - 1:
            ax_kymo.axhline(y=y_offset + n_beads + 0.5,
                            color="black", linewidth=2)

        y_offset += n_beads

    ax_kymo.set_ylim(y_offset + 0.5, 0.5)
    ax_kymo.set_xlabel("Timestep")
    ax_kymo.set_title("Kymograph")

    legend_patches = [Patch(color=TYPE_COLORS[t], label=TYPE_LABELS[t])
                      for t in [1, 2, 3]]
    ax_kymo.legend(handles=legend_patches, fontsize=8)

    plt.tight_layout()

    if outfile:
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {outfile}")
    else:
        plt.show()
# -- MAIN ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Plot LAMMPS polymer dump.")
    ap.add_argument("dump", nargs="?", default=DUMP_FILE,
                    help=f"Path to dump file (default: {DUMP_FILE})")
    ap.add_argument("--out", default=None,
                    help="Save figure to this path instead of displaying it")
    args = ap.parse_args()

    print(f"Reading {args.dump} ...")
    timesteps, frames = parse_dump(args.dump)
    print(f"  -> {len(timesteps)} frames, timesteps {timesteps[0]} - {timesteps[-1]}")

    ids_list, arrays = build_arrays(frames, POLYMERS)

    for i, ids in enumerate(ids_list):
        print(f"  -> Polymer {i+1}: {len(ids)} beads")

    plot_all(timesteps, ids_list, arrays, outfile=args.out)


if __name__ == "__main__":
    main()