import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg
from tqdm import tqdm
from pathlib import Path
import argparse
import base64
import io
import json
import sys
from pathlib import Path
try:
    from matplotlib.figure import Figure
    import matplotlib.svg as msvg
    SVG_AVAILABLE = True
except ImportError:
    SVG_AVAILABLE = False
from pdf2image import convert_from_path

# -- PLOTTING -----------------------------------------------------------------
# Row 0 (30%):   (a) Model 1 (first column that takes 50%)| (b) Model 2 (last column that takes 50%)
# Row 1 (55%):   (c) kymo chip 1| kymo chip 2| kymo chip 3| kymo chip 4| (e spans row 1–2) (last column that takes 50%)
# Row 2 (15%):   (d) HiC contact 1| HiC contact2| HiC contact3| HiC contact 4| (e continues) (last column that takes 50%)
# c) and d) is divided in 4 columns they all take 1/4 of the 50%
# ---------------------------------------------------------------------------
# PRX Life rcParams  —  tuned for A4 (170 mm column width)
# ---------------------------------------------------------------------------
# A4 usable width  ≈ 170 mm  ≈ 6.69 in
# A4 usable height ≈ 257 mm  ≈ 10.12 in  (with 2 cm margins top/bottom)

PRX_RC = {
    "font.family":        "serif",
    "font.size":          10.2,   # 8 × 1.4
    "axes.labelsize":     7.6,   # 9 × 1.4
    "axes.titlesize":     10.2,   # 8 × 1.4
    "xtick.labelsize":    10.2,   # 8 × 1.4
    "ytick.labelsize":    10.2,   # 8 × 1.4
    "legend.fontsize":    8.8,    # 7 × 1.4
    "legend.framealpha":  0.85,
    "legend.edgecolor":   "0.7",
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "xtick.minor.width":  0.5,
    "ytick.minor.width":  0.5,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.top":          True,
    "ytick.right":        True,
    "lines.linewidth":    1.2,
    "figure.dpi":         500,
    "savefig.dpi":        500,
    "savefig.bbox":       "tight",
}


A4_WIDTH  = 13    
A4_HEIGHT = 8   

# ── Config ────────────────────────────────────────────────────────────────────
LENGTH_CHRO  = 500
WINDOW_SIZE  = 100
TYPE_LABELS  = {1: "A", 2: "U", 3: "M"}
COLORS       = {1: "#4e79a7", 2: "#f0c040", 3: "#e15759"}
OUT_DEFAULT  = "combined_kymo_vertical.pdf"
# ─────────────────────────────────────────────
# USER SETTINGS
# ─────────────────────────────────────────────
TRAJ_FILE          = "dump.lammpstrj"
POLYMER_IDS        = set(range(1, 501))   # atom IDs 1–80
CONTACT_DIST       = 3.0                 # contact cutoff in σ
# ─────────────────────────────────────────────

SNAPSHOT_LIST = [1,2,3,4] # i will put name of paths later
MODEL_LIST = [1,2] # i will put name of paths later

# ══════════════════════════════════════════════
# FRUIT PUNCH COLORMAPS
# ══════════════════════════════════════════════
FRUIT_PUNCH = mcolors.LinearSegmentedColormap.from_list(
    "fruit_punch", ["#ffffff", "#ffd6e0", "#ff6b9d", "#c0392b", "#6b0000"])
FRUIT_PUNCH_STD = mcolors.LinearSegmentedColormap.from_list(
    "fruit_punch_std", ["#f8f0fb", "#e8c8f0", "#ff6b9d", "#c0392b", "#6b0000"])
FRUIT_PUNCH_HIC = mcolors.LinearSegmentedColormap.from_list(
    "fruit_punch_hic", ["#ffffff", "#ffe0ec", "#ff4d7d", "#b5001f", "#3d0010"])

# ─────────────────────────────────────────────────────────────────────────────

def index_trajectory(filepath):
    """
    Single fast pass over the file. For each frame records:
        timestep   : int
        offset     : byte position of the 'ITEM: TIMESTEP' line
        n_atoms    : number of atoms in this frame

    Also prints a summary: total frames, timestep interval, time span.

    The number of lines per frame is:
        1  ITEM: TIMESTEP
        1  <timestep value>
        1  ITEM: NUMBER OF ATOMS
        1  <n_atoms value>
        1  ITEM: BOX BOUNDS ...
        3  <box lines>
        1  ITEM: ATOMS ...
        n_atoms  <atom lines>
      = 9 + n_atoms  lines total per frame
    We do NOT need to count these manually — we just seek by byte offset.
    """
    path = Path(filepath)
    if not path.exists():
        sys.exit(f"[ERROR] File not found: {filepath}")

    index = []       # list of {"timestep": int, "offset": int, "n_atoms": int}
    reading_ts     = False
    reading_natoms = False
    current_ts     = None
    current_offset = None

    print(f"[1/3] Scanning trajectory index: {filepath}")
    file_size = path.stat().st_size

    with open(path, "rb") as fh:   # binary for reliable tell()
        with tqdm(total=file_size, desc="  Indexing",
                  unit="B", unit_scale=True, dynamic_ncols=True) as pbar:
            while True:
                offset = fh.tell()
                raw = fh.readline()
                if not raw:
                    break
                pbar.update(len(raw))
                line = raw.decode("ascii", errors="replace").strip()

                if line == "ITEM: TIMESTEP":
                    current_offset = offset
                    reading_ts = True
                    reading_natoms = False
                    continue

                if reading_ts:
                    current_ts = int(line)
                    reading_ts = False
                    continue

                if line == "ITEM: NUMBER OF ATOMS":
                    reading_natoms = True
                    continue

                if reading_natoms:
                    n_atoms = int(line)
                    reading_natoms = False
                    index.append({
                        "timestep": current_ts,
                        "offset":   current_offset,
                        "n_atoms":  n_atoms,
                    })

    if not index:
        sys.exit("[ERROR] No frames found in trajectory file.")

    # ── Summary
    n_total = len(index)
    ts_list = [f["timestep"] for f in index]
    dt = None
    if n_total > 1:
        diffs = [ts_list[i+1] - ts_list[i] for i in range(min(10, n_total - 1))]
        if len(set(diffs)) == 1:
            dt = diffs[0]
        else:
            dt = int(np.median(diffs))

    print(f"\n  Trajectory summary:")
    print(f"    Total frames   : {n_total}")
    print(f"    Atoms/frame    : {index[0]['n_atoms']}")
    if dt is not None:
        print(f"    Timestep freq  : every {dt} steps")
    print(f"    First timestep : {ts_list[0]}")
    print(f"    Last  timestep : {ts_list[-1]}")
    if dt and dt > 0:
        print(f"    Time span      : {ts_list[-1] - ts_list[0]} steps  "
              f"({(ts_list[-1] - ts_list[0]) // dt} intervals)")

    return index

def _parse_frame_at(fh, frame_info):
    """
    fh is already seeked to frame_info['offset'] (start of 'ITEM: TIMESTEP').
    Parses and returns (timestep, box, coords_dict).
    """
    fh.readline()                              # ITEM: TIMESTEP
    timestep = int(fh.readline().strip())
    fh.readline()                              # ITEM: NUMBER OF ATOMS
    n_atoms = int(fh.readline().strip())

    fh.readline()                              # ITEM: BOX BOUNDS ...
    box = []
    for _ in range(3):
        lo, hi = map(float, fh.readline().split())
        box.append([lo, hi])
    box = np.array(box)
    L = box[:, 1] - box[:, 0]

    header = fh.readline()                     # ITEM: ATOMS id type xs ys zs
    cols   = header.split()[2:]
    id_col = cols.index("id")
    try:
        x_col  = cols.index("xs"); scaled = True
    except ValueError:
        x_col  = cols.index("x");  scaled = False
    y_col, z_col = x_col + 1, x_col + 2

    coords = {}
    for _ in range(n_atoms):
        parts = fh.readline().split()
        aid = int(parts[id_col])
        x   = float(parts[x_col])
        y   = float(parts[y_col])
        z   = float(parts[z_col])
        if scaled:
            x = box[0, 0] + x * L[0]
            y = box[1, 0] + y * L[1]
            z = box[2, 0] + z * L[2]
        coords[aid] = np.array([x, y, z])

    return timestep, box, coords


def load_frames(filepath, index, needed_indices, polymer_ids, cutoff):
    """
    Seeks to each needed frame by byte offset; parses only those frames.
    Returns a dict  position → (timestep, contact_matrix)
    and the sorted polymer bead IDs (from the first frame).
    """
    needed_sorted = sorted(set(needed_indices))   # visit offsets in order
    results  = {}
    ids_out  = None

    print(f"\n[3/3] Loading {len(needed_sorted)} frames from disk …")
    with open(filepath, "r") as fh:
        for pos in tqdm(needed_sorted, desc="  Parsing frames",
                        unit="frame", dynamic_ncols=True):
            fh.seek(index[pos]["offset"])
            ts, box, coords = _parse_frame_at(fh, index[pos])
            mat, ids = _coords_to_contact_matrix(coords, polymer_ids, cutoff, box)
            results[pos] = (ts, mat)
            if ids_out is None:
                ids_out = ids

    print(f"  → done. Polymer beads: {len(ids_out)} "
          f"(IDs {ids_out[0]}–{ids_out[-1]})")
    return results, ids_out

def _coords_to_contact_matrix(coords, polymer_ids, cutoff, box):
    ids = sorted(pid for pid in polymer_ids if pid in coords)
    if not ids:
        raise ValueError("No polymer atoms found in this frame.")
    L   = box[:, 1] - box[:, 0]
    pos = np.array([coords[i] for i in ids])           # (N, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (N, N, 3)
    diff -= np.round(diff / L) * L                        # minimum image
    dist  = np.sqrt((diff ** 2).sum(axis=-1))             # (N, N)
    mat   = (dist <= cutoff).astype(np.float32)
    np.fill_diagonal(mat, 0)
    return mat, ids

def _ticks(ids, n=8):
    N = len(ids)
    t = np.arange(0, N, max(1, N // n))
    return t, [str(ids[i]) for i in t]

def plot_hic(avg_mat, ids, output_dir):
    N = len(ids)
    ticks, labels = _ticks(ids)
    log_mat = np.log10(avg_mat + 1e-3)

    fig = plt.figure(figsize=(9.5, 5.2))
    gs  = GridSpec(1, 2, width_ratios=[3, 1], wspace=0.38)
    ax_map = fig.add_subplot(gs[0])
    ax_sep = fig.add_subplot(gs[1])

    vmin, vmax = np.percentile(log_mat, [2, 98])
    im = ax_map.imshow(log_mat, cmap=FRUIT_PUNCH_HIC, vmin=vmin, vmax=vmax,
                       origin="lower", interpolation="nearest")
    ax_map.set_xlabel("Monomer index", fontsize=11)
    ax_map.set_ylabel("Monomer index", fontsize=11)
    ax_map.set_title("Hi-C Style Map  (log₁₀ contact probability)",
                     fontsize=11, fontweight="bold")
    ax_map.set_xticks(ticks); ax_map.set_xticklabels(labels)
    ax_map.set_yticks(ticks); ax_map.set_yticklabels(labels)
    fig.colorbar(im, ax=ax_map, label="log₁₀(P_contact)")

    sep_vals = [np.diagonal(avg_mat, offset=s).mean() for s in range(1, N)]
    ax_sep.plot(sep_vals, np.arange(1, N), color="#c0392b", lw=1.8)
    ax_sep.set_xscale("log")
    ax_sep.set_xlabel("P(contact)", fontsize=10)
    ax_sep.set_ylabel("|i − j|", fontsize=10)
    ax_sep.set_title("P vs separation", fontsize=10, fontweight="bold")
    ax_sep.grid(True, alpha=0.3)

    fig.suptitle(f"Hi-C Analysis  (N={N}, cutoff={CONTACT_DIST} σ)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = output_dir / "04_hic_style_map.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

def parse_lammps_dump(filepath: str, n_atoms: int) -> np.ndarray:
    """Parse a LAMMPS dump file and return a (n_beads × n_frames) integer matrix."""
    types_all: list[np.ndarray] = []
    buf: list[list[str]] = []
    in_atoms = False

    with open(filepath) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith("ITEM: ATOMS"):
                if buf:
                    arr = np.array(buf, dtype=int)
                    arr = arr[arr[:, 0].argsort()]
                    types_all.append(arr[:, 1])
                    buf = []
                in_atoms = True
                continue
            if line.startswith("ITEM:"):
                in_atoms = False
                continue
            if in_atoms and line:
                buf.append(line.split())

    if buf:
        arr = np.array(buf, dtype=int)
        arr = arr[arr[:, 0].argsort()]
        types_all.append(arr[:, 1])

    if not types_all:
        raise ValueError(f"No ATOMS blocks found in {filepath}")

    return np.column_stack(types_all)

def _try_load_image(path):
    """Return image array or None if file is missing."""
    try:
        return mpimg.imread(path)
    except FileNotFoundError:
        warnings.warn(f"Snapshot not found: {path} — showing placeholder.")
        return None
    
# currently wrong name but structure ok
def draw_snapshot_row(ax_PS, ax_NO_PS, path_PS, path_NO_PS,
                      inset_PS=None, inset_NO_PS=None):
    def process(ax, path, inset_path, inset_loc, inset_color):
        img = _try_load_image(path)
        ax.set_xticks([])
        ax.set_yticks([])
        if img is not None:
            img = crop_img(img, 0.103)
            ax.imshow(img, aspect="equal")
            if inset_path is not None:
                inset_img = _try_load_image(inset_path)
                add_inset(ax, inset_img, loc=inset_loc, edgecolor=inset_color)
        else:
            ax.set_facecolor("#e8e8e8")
            ax.text(
                0.5, 0.5,
                f"{path}\n(not found)",
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=9.8, color="#888888",
                style="italic",
            )

    process(ax_NO_PS, path_NO_PS, inset_NO_PS, inset_loc="upper right", inset_color=COL_NPS)
    process(ax_PS, path_PS, inset_PS, inset_loc="upper left",  inset_color=COL_PS)    
def plot_panels(
    matrices: list[np.ndarray],
    labels: list[str],
    out: str,
    max_frames: int | None = None,
) -> None:
    """
    Plot 1–4 kymograph panels vertically (time ↓, nucleosome position →).
    All panels share the same time (y) axis.
    Each panel has a ChIP-like strip appended below.

    Parameters
    ----------
    matrices   : list of (n_beads × n_frames) arrays
    labels     : panel titles (same length as matrices)
    out        : output file path
    max_frames : if given, truncate every matrix to this many frames
    """
    n_panels = len(matrices)
    if not (1 <= n_panels <= 4):
        raise ValueError("Pass between 1 and 4 files.")

    # ── Optionally limit time ─────────────────────────────────────────────────
    if max_frames is not None:
        matrices = [m[:, :max_frames] for m in matrices]

    n_beads  = matrices[0].shape[0]
    n_frames = matrices[0].shape[1]

    # Colour map
    cmap   = ListedColormap([COLORS[k] for k in sorted(COLORS)])
    bounds = [0.5, 1.5, 2.5, 3.5]
    norm   = BoundaryNorm(bounds, cmap.N)

    # ── Figure layout ─────────────────────────────────────────────────────────
    # Each panel: kymograph column + narrow ChIP strip
    # height ratio  kymograph : chip strip  ≈ 8:1
    kymo_h = 8
    chip_h = 1
    panel_w = 3.0          # inches per kymograph panel
    chip_w  = 0.45         # inches for the ChIP strip

    fig_w = n_panels * (panel_w + chip_w) + 0.8   # + left margin
    fig_h = 7.0

    fig = plt.figure(figsize=(fig_w, fig_h))

    # We build a gridspec: for each panel we need 2 columns (kymo + chip).
    # All panels share the y-axis (time).
    from matplotlib.gridspec import GridSpec

    # column widths: alternate panel_w and chip_w
    col_widths = []
    for _ in range(n_panels):
        col_widths += [panel_w, chip_w]

    gs = GridSpec(
        2, n_panels,                      # ← 1 column per panel
        figure=fig,
        height_ratios=[kymo_h, chip_h],  # ← controls vertical sizes
        wspace=0.15,
        left=0.10, right=0.97,
        top=0.93, bottom=0.08,
    )


    axes_kymo = []
    axes_chip = []

    for i, (matrix, label) in enumerate(zip(matrices, labels)):
        # ── Kymograph axes ────────────────────────────────────────────────────
        # First panel owns the y-axis; the rest share it.
        if i == 0:
            ax = fig.add_subplot(gs[0, i])
        else:
            ax = fig.add_subplot(gs[0, i], sharey=axes_kymo[0])

        # matrix is (n_beads × n_frames); we want time on y, bead on x
        # → transpose to (n_frames × n_beads), origin="upper" → time flows down
        im = ax.imshow(
            matrix.T,           # shape: (n_frames, n_beads)
            cmap=cmap,
            norm=norm,
            aspect="auto",
            interpolation="none",
            origin="upper",
        )


        # ax.set_title(label, fontsize=11, fontweight="bold")
        # ax.set_xlabel("Nucleosome position", fontsize=9)



        # y-ticks: time, formatted as e.g. "1e4" (multiply frame index by 1000)
        def fmt_time(x, pos, _scale=1000):
            return f"{(round(x / _scale,2))}e4"

        if i == 0:
            ax.set_ylabel(r"Time ($\tau_{LJ}$)", fontsize=11)
            ax.set_yticks([0, 1000, 1999])
            ax.set_yticklabels(['0e4','1e4','2e4'])
            ax.set_xticks([])
            ax.set_xlabel("")   # optional, ensures no leftover label
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.set_xticks([])
            ax.set_xlabel("")   # optional, ensures no leftover label

        axes_kymo.append(ax)

        # ── ChIP strip axes ───────────────────────────────────────────────────
        if i == 0:
            ax_chip = fig.add_subplot(gs[1, i])
            ax_chip.set_ylabel("H3K9me3", fontsize=11)
            ax_chip.yaxis.set_tick_params(labelsize=11)
            
        else:
            ax_chip = fig.add_subplot(gs[1, i], sharey=axes_chip[0])

        binary = (matrix == 3).astype(float)
        mean_signal = binary.mean(axis=1)   # (n_beads,)
        x = np.arange(n_beads)

        color = COLORS[3]

        # correct 1D profile plot
        ax_chip.plot(x, mean_signal, color=color, lw=1.2)
        ax_chip.fill_between(x, mean_signal, color=color, alpha=0.35)


        ax_chip.set_xlabel("Nucleosome position", fontsize=11)
        ax_chip.set_xlim(0, n_beads - 1)
        ax_chip.set_ylim(0, 1)
            # if i == 0:
            # ax_chip.set_title("H3K9me3", fontsize=8)
        
        
        ax_chip.tick_params(axis="x")
        ax_chip.grid(axis="y", lw=0.4, alpha=0.5)
        # x-ticks: nucleosome positions
        xticks = [0, 99, 199, 299, 399, 499]
        xtick_labels = [str(t + 1) for t in xticks]
        ax_chip.set_xticks(xticks)
        ax_chip.set_xticklabels(xtick_labels, fontsize=11)
        ax_chip.spines[["top", "right"]].set_visible(False)

        axes_chip.append(ax_chip)

    # # ── Shared colorbar (optional, placed at far right) ───────────────────────
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])
    # cbar = fig.colorbar(sm, ax=axes_kymo + axes_chip, ticks=[1, 2, 3],
    #                     shrink=0.5, pad=0.02, aspect=25)
    # cbar.ax.set_yticklabels([TYPE_LABELS[k] for k in sorted(TYPE_LABELS)])
    # cbar.set_label("Chromatin state", fontsize=9)

    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved → {out}")

def make_master_figure():
    fig = plt.figure(figsize=(A4_WIDTH, A4_HEIGHT))

    # Main grid: 3 rows × 2 columns
    gs = GridSpec(
        nrows=3,
        ncols=2,
        height_ratios=[0.30, 0.55, 0.15],
        width_ratios=[1, 1],
        figure=fig
    )

    # ── Row 0 ─────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])  # Model 1
    ax_b = fig.add_subplot(gs[0, 1])  # Model 2

    # ── Left block (rows 1–2, col 0) ──────────────────────
    gs_left = gs[1:, 0].subgridspec(
        2, 4,  # 2 rows (c, d), 4 columns
        height_ratios=[0.75, 0.25],
        wspace=0.05,
        hspace=0.05
    )

    ax_c = [fig.add_subplot(gs_left[0, i]) for i in range(4)]
    ax_d = [fig.add_subplot(gs_left[1, i]) for i in range(4)]

    # ── Right big panel (e spans rows 1–2) ────────────────
    ax_e = fig.add_subplot(gs[1:, 1])

    return fig, ax_a, ax_b, ax_c, ax_d, ax_e

# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Vertical kymograph plotter (1–4 panels, shared time axis)."
    )
    p.add_argument(
        "files",
        nargs="+",
        metavar="FILE.dat",
        help="1–4 LAMMPS dump files to plot side-by-side.",
    )
    p.add_argument(
        "--labels",
        nargs="*",
        metavar="LABEL",
        help="Panel titles (defaults to filenames). Must match number of files.",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        metavar="N",
        help="Truncate to the first N frames (time limit).",
    )
    p.add_argument(
        "--out",
        default=OUT_DEFAULT,
        metavar="OUTPUT.pdf",
        help=f"Output file (default: {OUT_DEFAULT}).",
    )
    p.add_argument(
        "--n-beads",
        type=int,
        default=LENGTH_CHRO,
        metavar="N",
        help=f"Number of beads / nucleosomes (default: {LENGTH_CHRO}).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    files = args.files
    if not (1 <= len(files) <= 4):
        raise SystemExit("ERROR: supply between 1 and 4 .dat files.")

    labels = args.labels or [Path(f).stem for f in files]
    if len(labels) != len(files):
        raise SystemExit("ERROR: --labels count must match number of files.")

    matrices = []
    for f in files:
        if not Path(f).exists():
            raise FileNotFoundError(f"Cannot find {f!r}")
        print(f"Parsing {f} …")
        m = parse_lammps_dump(f, args.n_beads)
        print(f"  → {m.shape[0]} beads × {m.shape[1]} frames")
        matrices.append(m)
        # 1. Fast index pass — only reads TIMESTEP + NUMBER OF ATOMS lines
        index = index_trajectory(TRAJ_FILE)

        # 2. Pick evenly distributed indices for both tasks
        evo_idx, avg_idx = pick_frame_indices(
            index, N_FRAMES_EVOLUTION, N_FRAMES_AVG
        )

        # 3. Load only the needed frames (seek by byte offset)
        needed = np.union1d(evo_idx, avg_idx)
        frame_data, ids = load_frames(
            TRAJ_FILE, index, needed, POLYMER_IDS, CONTACT_DIST
        )

        # Reassemble ordered lists for each task
        evo_frames = [frame_data[i] for i in evo_idx]   # list of (ts, mat)
        avg_frames = [frame_data[i] for i in avg_idx]

        # 4. Compute average and std from avg_frames
        stack   = np.stack([mat for _, mat in tqdm(avg_frames,
                            desc="  Stacking avg frames", unit="frame",
                            dynamic_ncols=True)], axis=0)
        avg_mat = stack.mean(axis=0)
        std_mat = stack.std(axis=0)
        plot_hic(avg_mat, ids, output_dir)

    plot_panels(matrices, labels, out=args.out, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
