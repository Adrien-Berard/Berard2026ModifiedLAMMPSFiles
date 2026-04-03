"""
Chromatin state visualisation
- Kymograph (space-time plot)
- ChIP-seq-like averaged signal subplots

Input: LAMMPS dump file with columns  id  type
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap, BoundaryNorm
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
FILE         = "id_and_type.dat"
LENGTH_CHRO  = 500          # beads per frame
WINDOW_SIZE  = 100            # rolling-average window for ChIP-seq plot
FRAMES_CHIP  = [10,200,10000,100000]   # frames to show in ChIP panel
TYPE_LABELS  = {1: "A", 2: "U", 3: "M"}
COLORS       = {1: "#4e79a7", 2: "#f0c040", 3: "#e15759"}   # A=blue, U=yellow, M=red
OUT_KYMO     = "kymograph.pdf"
OUT_CHIP     = "ChIPseq_Like_Averaged_Signal_Subplots.pdf"
# ─────────────────────────────────────────────────────────────────────────────


def parse_lammps_dump(filepath: str, n_atoms: int) -> np.ndarray:
    """
    Parse a LAMMPS dump with  'ITEM: ATOMS id type'  blocks.
    Returns a 2-D int array of shape (n_atoms, n_frames) sorted by bead id.
    """
    types_all: list[np.ndarray] = []
    buf: list[list[str]] = []
    in_atoms = False

    with open(filepath) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith("ITEM: ATOMS"):
                # flush previous block
                if buf:
                    arr = np.array(buf, dtype=int)          # shape (n_atoms, 2)
                    arr = arr[arr[:, 0].argsort()]          # sort by id
                    types_all.append(arr[:, 1])
                    buf = []
                in_atoms = True
                continue
            if line.startswith("ITEM:"):
                in_atoms = False
                continue
            if in_atoms and line:
                buf.append(line.split())

    # flush last block
    if buf:
        arr = np.array(buf, dtype=int)
        arr = arr[arr[:, 0].argsort()]
        types_all.append(arr[:, 1])

    if not types_all:
        raise ValueError(f"No ATOMS blocks found in {filepath}")

    matrix = np.column_stack(types_all)   # (n_atoms, n_frames)
    return matrix


def plot_kymograph(matrix: np.ndarray, out: str) -> None:
    """Space-time (kymograph) plot of chromatin states."""
    n_types = len(TYPE_LABELS)
    cmap   = ListedColormap([COLORS[k] for k in sorted(COLORS)])
    bounds = [0.5, 1.5, 2.5, 3.5]
    norm   = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(
        matrix,
        cmap=cmap, norm=norm,
        aspect="auto", interpolation="none",
        origin="upper",
    )

    # colour-bar acting as legend
    cbar = fig.colorbar(im, ax=ax, ticks=[1, 2, 3], pad=0.02)
    cbar.ax.set_yticklabels([TYPE_LABELS[k] for k in sorted(TYPE_LABELS)], fontsize=10)
    cbar.set_label("Chromatin state", fontsize=10)

    ax.set_xlabel("Frame", fontsize=11)
    ax.set_ylabel("Bead index", fontsize=11)
    ax.set_title("Chromatin state kymograph", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_major_locator(ticker.AutoLocator())

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved  {out}")


def rolling_mean(signal: np.ndarray, w: int) -> np.ndarray:
    """Fast uniform rolling average (same length as input, edge-padded)."""
    kernel = np.ones(w) / w
    return np.convolve(signal, kernel, mode="same")


def plot_chipseq(matrix: np.ndarray, frames: list[int], window: int, out: str) -> None:
    """ChIP-seq-like signal of type-3 marks for selected frames."""
    n_beads, n_frames = matrix.shape
    valid = [f for f in frames if 1 <= f <= n_frames]
    if not valid:
        print("No valid frames for ChIP-seq plot — skipping.")
        return

    fig, axes = plt.subplots(
        len(valid), 1,
        figsize=(10, 2.2 * len(valid)),
        sharex=True, sharey=True,
        gridspec_kw={"hspace": 0.05},
    )
    if len(valid) == 1:
        axes = [axes]

    x = np.arange(n_beads)
    color = COLORS[3]

    for ax, frame in zip(axes, valid):
        raw    = (matrix[:, frame - 1] == 3).astype(float)
        signal = rolling_mean(raw, window)

        ax.fill_between(x, signal, color=color, alpha=0.35)
        ax.plot(x, signal, color=color, lw=1.2, alpha=0.85)
        ax.set_ylabel(f"f{frame}", fontsize=9, rotation=0, labelpad=28, va="center")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.grid(axis="y", lw=0.4, alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("Bead index", fontsize=11)
    fig.supylabel("Type-3 (M) mark density", fontsize=10, x=0.02)
    fig.suptitle(
        f"ChIP-seq-like signal  ·  M marks  ·  window = {window} beads",
        fontsize=12, fontweight="bold", y=1.01,
    )

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved  {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not Path(FILE).exists():
        raise FileNotFoundError(f"Cannot find {FILE!r}")

    print(f"Parsing {FILE} …")
    matrix = parse_lammps_dump(FILE, LENGTH_CHRO)
    print(f"  → {matrix.shape[0]} beads  ×  {matrix.shape[1]} frames")

    plot_kymograph(matrix, OUT_KYMO)
    plot_chipseq(matrix, FRAMES_CHIP, WINDOW_SIZE, OUT_CHIP)