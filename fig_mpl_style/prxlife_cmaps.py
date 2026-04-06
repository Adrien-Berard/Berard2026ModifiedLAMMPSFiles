"""
prxlife_cmaps.py
================
Custom colormaps and phase-diagram helpers for PRX Life figures.

Usage
-----
    import matplotlib.pyplot as plt
    from prxlife_cmaps import phase_cmap, phase_norm, PHASE_COLORS, phase_legend_patches

    plt.style.use('prxlife.mplstyle')

    # Discrete phase diagram (integer mask: 0=A, 1=bistable, 2=M)
    img = ax.imshow(phase_mask, cmap=phase_cmap, norm=phase_norm,
                    origin='lower', aspect='auto')
    ax.legend(handles=phase_legend_patches(), loc='upper right')

    # Continuous bistability score in [0, 1]
    img = ax.imshow(score, cmap=bistable_cmap, vmin=0, vmax=1)
    plt.colorbar(img, ax=ax, label='Bistability score')
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# ------------------------------------------------------------------
# Semantic colors (colorblind-safe Okabe-Ito anchors)
# ------------------------------------------------------------------
PHASE_COLORS = {
    "A":        "#0072B2",   # blue        — monostable A
    "bistable": "#2CA05A",   # vivid green — bistable / success
    "M":        "#C0392B",   # crimson red — monostable M
}

# ------------------------------------------------------------------
# 1. Discrete 3-state colormap  (values 0, 1, 2)
#    0 → monostable A  (blue)
#    1 → bistable      (green)
#    2 → monostable M  (red)
# ------------------------------------------------------------------
_discrete_colors = [
    PHASE_COLORS["A"],
    PHASE_COLORS["bistable"],
    PHASE_COLORS["M"],
]
phase_cmap = mcolors.ListedColormap(_discrete_colors, name="prx_phase")
phase_norm = mcolors.BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5], ncolors=3)


def phase_legend_patches(labels=("Monostable A", "Bistable", "Monostable M")):
    """Return a list of Patch handles for a legend."""
    return [
        mpatches.Patch(facecolor=PHASE_COLORS["A"],        label=labels[0]),
        mpatches.Patch(facecolor=PHASE_COLORS["bistable"], label=labels[1]),
        mpatches.Patch(facecolor=PHASE_COLORS["M"],        label=labels[2]),
    ]


# ------------------------------------------------------------------
# 2. Continuous bistability-score colormap
#    0 → pure blue (A-dominated)   via white centre
#    0.5 → rich green (fully bistable)
#    1 → pure red  (M-dominated)
#
#    Built as a smooth perceptual path:
#      blue → desaturated teal → vivid green → desaturated salmon → red
# ------------------------------------------------------------------
_score_nodes = [0.0, 0.25, 0.5, 0.75, 1.0]
_score_rgb = [
    mcolors.to_rgb("#0072B2"),   # blue
    mcolors.to_rgb("#66B8A0"),   # desaturated teal
    mcolors.to_rgb("#2CA05A"),   # vivid green (peak bistability)
    mcolors.to_rgb("#E07060"),   # desaturated salmon
    mcolors.to_rgb("#C0392B"),   # crimson red
]
bistable_cmap = mcolors.LinearSegmentedColormap.from_list(
    "prx_bistable",
    list(zip(_score_nodes, _score_rgb)),
    N=256,
)

# ------------------------------------------------------------------
# 3. Green-only gradient  (white → vivid green)
#    Useful for single-variable "success" quantity heat-maps.
# ------------------------------------------------------------------
green_cmap = mcolors.LinearSegmentedColormap.from_list(
    "prx_green",
    [(0.0, "#FFFFFF"), (0.15, "#C8EDD8"), (0.5, "#5DBF85"), (1.0, "#1A7A40")],
    N=256,
)

# ------------------------------------------------------------------
# Register all colormaps so plt.set_cmap('prx_phase') works after import
# ------------------------------------------------------------------
for _cm in (phase_cmap, bistable_cmap, green_cmap):
    try:
        plt.colormaps.register(_cm)
    except AttributeError:
        # matplotlib < 3.5 fallback
        plt.cm.register_cmap(cmap=_cm)


# ------------------------------------------------------------------
# Subplot helper
# ------------------------------------------------------------------
import string

# PRX column widths (inches)
_PRX_SINGLE = 3.35   # one column
_PRX_DOUBLE = 6.97   # two columns (full page width)
_PRX_ROW_H  = 2.4    # default height per row

def prx_subplots(
    nrows=1,
    ncols=1,
    figsize=None,
    sharex=False,
    sharey=False,
    label_type="paren",   # "paren" → (a), "bold_paren" → **(a)**, "plain" → a.
    label_pos=(0.02, 0.98),
    constrained=True,
    **kwargs,
):
    """
    Create PRX Life–style subplots with automatic panel labels.

    Parameters
    ----------
    nrows, ncols : int
        Grid size.
    figsize : tuple or None
        Override figure size. Default: single-column width for ncols=1,
        double-column width for ncols >= 2; height scales with nrows.
    sharex, sharey : bool or str
        Passed straight to plt.subplots.
    label_type : str
        "paren"       → (a), (b), …   [default]
        "plain"       → a, b, …
    label_pos : (x, y)
        Axes-fraction position of the label.  Default top-left (0.02, 0.98).
    constrained : bool
        Use constrained_layout (recommended for multi-panel PRX figures).
    **kwargs
        Forwarded to plt.subplots.

    Returns
    -------
    fig : Figure
    axes : list of Axes   (always 1-D, even for a single panel)
    """
    if figsize is None:
        width  = _PRX_DOUBLE if ncols >= 2 else _PRX_SINGLE
        height = _PRX_ROW_H * nrows
        figsize = (width, height)

    layout = "constrained" if constrained else None

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        layout=layout,
        **kwargs,
    )

    # Always return a flat list — callers never need to know the grid shape
    flat = np.array(axes).flatten().tolist() if not isinstance(axes, plt.Axes) else [axes]

    fmt = {"paren": "({letter})", "plain": "{letter}"}[label_type]

    for i, ax in enumerate(flat):
        letter = string.ascii_lowercase[i]
        label  = fmt.format(letter=letter)
        ax.text(
            *label_pos,
            label,
            transform=ax.transAxes,
            fontsize=ax.xaxis.label.get_fontsize(),   # matches axes.labelsize from style
            fontweight="bold",
            va="top",
            ha="left",
            zorder=10,
        )

    return fig, flat


# ------------------------------------------------------------------
# Quick demo  (run this file directly to preview the colormaps)
# ------------------------------------------------------------------
if __name__ == "__main__":
    plt.style.use("prxlife.mplstyle")
    fig, axes = prx_subplots(1, 3)

    # Panel 1 — discrete phase diagram
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 200)
    y = np.linspace(0, 1, 200)
    X, Y = np.meshgrid(x, y)
    phase = np.zeros_like(X, dtype=int)
    phase[Y > 0.35 + 0.2 * np.sin(3 * X)] = 1   # bistable band
    phase[Y > 0.65 + 0.15 * np.cos(2 * X)] = 2  # monostable M

    im0 = axes[0].imshow(phase, cmap=phase_cmap, norm=phase_norm,
                         origin="lower", aspect="auto",
                         extent=[0, 1, 0, 1])
    axes[0].set_title("Discrete phase")
    axes[0].legend(handles=phase_legend_patches(), fontsize=5,
                   loc="lower right", frameon=False)

    # Panel 2 — continuous bistability score
    score = np.clip(np.sin(np.pi * X) * np.sin(np.pi * Y) + 0.1 * rng.normal(size=X.shape), 0, 1)
    im1 = axes[1].imshow(score, cmap=bistable_cmap, vmin=0, vmax=1,
                         origin="lower", aspect="auto", extent=[0, 1, 0, 1])
    plt.colorbar(im1, ax=axes[1], label="Bistability score")
    axes[1].set_title("Continuous score")

    # Panel 3 — green-only gradient
    z = np.outer(np.linspace(0, 1, 200), np.ones(200))
    im2 = axes[2].imshow(z, cmap=green_cmap, vmin=0, vmax=1,
                         origin="lower", aspect="auto", extent=[0, 1, 0, 1])
    plt.colorbar(im2, ax=axes[2], label="Success metric")
    axes[2].set_title("Green gradient")

    for ax in axes:
        ax.set_xlabel("Parameter 1")
        ax.set_ylabel("Parameter 2")

    fig.savefig("prxlife_cmap_demo.pdf")
    plt.show()
    print("Saved prxlife_cmap_demo.pdf")
