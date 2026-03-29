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
POLY1_IDS  = (1,   80)
POLY2_IDS  = (81, 160)

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

def build_arrays(frames, poly1_ids, poly2_ids):
    """Build integer arrays (n_frames, n_beads) for each polymer."""
    ids1 = list(range(poly1_ids[0], poly1_ids[1] + 1))
    ids2 = list(range(poly2_ids[0], poly2_ids[1] + 1))
    n = len(frames)

    arr1 = np.zeros((n, len(ids1)), dtype=np.int32)
    arr2 = np.zeros((n, len(ids2)), dtype=np.int32)

    for t, frame in enumerate(frames):
        for j, aid in enumerate(ids1):
            arr1[t, j] = frame.get(aid, 0)
        for j, aid in enumerate(ids2):
            arr2[t, j] = frame.get(aid, 0)

    return ids1, arr1, ids2, arr2


def compute_counts(arr):
    """Return dict type -> count array of shape (n_frames,)."""
    return {t: np.sum(arr == t, axis=1) for t in [1, 2, 3]}

# -- PLOTTING -----------------------------------------------------------------

def plot_all(timesteps, ids1, arr1, ids2, arr2, outfile=None):
    ts = np.array(timesteps)

    counts1 = compute_counts(arr1)
    counts2 = compute_counts(arr2)

    bounds = [0.5, 1.5, 2.5, 3.5]
    norm   = mcolors.BoundaryNorm(bounds, TYPE_CMAP.N)

    fig = plt.figure(figsize=(15, 12))
    fig.suptitle("Chromatin polymer dynamics", fontsize=14, fontweight="bold")
    gs  = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.8], hspace=0.45)

    # -- Time series ----------------------------------------------------------
    for row, (counts, label) in enumerate([
            (counts1, "Polymer 1 (IDs 1-80)"),
            (counts2, "Polymer 2 (IDs 81-160)")]):
        ax = fig.add_subplot(gs[row])
        for t in [1, 2, 3]:
            ax.plot(ts, counts[t],
                    color=TYPE_COLORS[t], label=TYPE_LABELS[t], linewidth=1.6)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Count")
        ax.set_title(label, fontsize=11)
        ax.set_xlim(ts[0], ts[-1])
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)

    # -- Stacked kymograph ----------------------------------------------------
    ax_kymo = fig.add_subplot(gs[2])

    # Plot each polymer separately with explicit y extents so bead index
    # maps correctly: polymer 1 -> y 1..80 (top), polymer 2 -> y 81..160 (bottom)
    # arr shape: (n_frames, n_beads) -> transpose to (n_beads, n_frames) for imshow
    ax_kymo.imshow(
        arr1.T,                          # (80, n_frames)
        aspect="auto",
        origin="upper",
        cmap=TYPE_CMAP,
        norm=norm,
        interpolation="nearest",
        extent=[ts[0], ts[-1], 80.5, 0.5],   # y: bead 1 (top) to 80 (bottom)
    )
    ax_kymo.imshow(
        arr2.T,                          # (80, n_frames)
        aspect="auto",
        origin="upper",
        cmap=TYPE_CMAP,
        norm=norm,
        interpolation="nearest",
        extent=[ts[0], ts[-1], 160.5, 80.5],  # y: bead 81 (top) to 160 (bottom)
    )

    # Black separator between polymer 1 and polymer 2
    ax_kymo.axhline(y=80.5, color="black", linewidth=2.5, linestyle="-")

    # y-axis: label each polymer block
    ax_kymo.set_yticks([40, 120])
    ax_kymo.set_yticklabels(
        ["Polymer 1\n(beads 1-80)", "Polymer 2\n(beads 81-160)"],
        fontsize=9
    )
    ax_kymo.set_ylim(160.5, 0.5)
    ax_kymo.yaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax_kymo.tick_params(axis="y", which="minor", left=True, length=3)

    ax_kymo.set_xlabel("Timestep")
    ax_kymo.set_title("Kymograph - both polymers", fontsize=11)
    ax_kymo.set_xlim(ts[0], ts[-1])

    legend_patches = [Patch(color=TYPE_COLORS[t], label=TYPE_LABELS[t])
                      for t in [1, 2, 3]]
    ax_kymo.legend(handles=legend_patches, fontsize=8,
                   loc="upper right", framealpha=0.85)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

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

    ids1, arr1, ids2, arr2 = build_arrays(frames, POLY1_IDS, POLY2_IDS)
    print(f"  -> Polymer 1: {len(ids1)} beads | Polymer 2: {len(ids2)} beads")

    plot_all(timesteps, ids1, arr1, ids2, arr2, outfile=args.out)


if __name__ == "__main__":
    main()
