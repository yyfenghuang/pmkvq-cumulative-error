"""Brand palette, rcParams, and colormaps. The single source for every asset.

No figure declares a color inline. Import ``PRIMARY`` .. ``BRAND_DIV`` from
here and use the role table below. See ``docs/pmkvq-h1-todo.md`` Section 4 for
the role assignment and the measured luminance ramp.
"""
from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# === BRAND PALETTE ===
PRIMARY = "#5C1D74"
SECONDARY = "#E5C300"
TERTIARY = "#1D7466"
TEXT_COLOR = "#25172A"
BG_COLOR = "#FCFAFF"

# Role assignment, fixed across all assets so a color means the same thing on
# every page.
#
#   Measured, feedback on   -> PRIMARY    (Arm B, free running)
#   Measured, feedback off  -> SECONDARY  (Arm A, teacher forced)
#   Measured, third series  -> TERTIARY   (Arm C, and non-arm right-axis series)
#   Analytic prediction     -> TEXT_COLOR at alpha=0.55, dashed
#   Canvas                  -> BG_COLOR
#   Ink                     -> TEXT_COLOR
ARM_B = PRIMARY
ARM_A = SECONDARY
ARM_C = TERTIARY
PREDICTION = TEXT_COLOR
PREDICTION_ALPHA = 0.55

# SECONDARY against BG_COLOR reaches only 1.67 contrast, so Arm A is drawn thick
# with a filled marker and never used for text or thin annotation lines.
ARM_A_LINEWIDTH = 2.0
ARM_A_MARKER = "o"
ARM_B_MARKER = "s"  # Arm A and Arm B separated by marker shape as well as color.


def apply_rcparams() -> None:
    """Install the brand rcParams. Call once at the top of every asset script."""
    mpl.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": BG_COLOR,
        "savefig.facecolor": BG_COLOR,
        "text.color": TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "axes.edgecolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "axes.prop_cycle": mpl.cycler(color=[PRIMARY, SECONDARY, TERTIARY]),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    })


# Sequential ramp for the A9 heatmap. Luminance falls monotonically along it,
# so BRAND_SEQ stays readable under LogNorm and survives grayscale conversion.
BRAND_SEQ = LinearSegmentedColormap.from_list(
    "brand_seq", [BG_COLOR, SECONDARY, TERTIARY, PRIMARY]
)

# Diverging map for signed quantities, zero on the canvas. Centre with
# TwoSlopeNorm(vcenter=0) wherever a signed quantity is drawn.
BRAND_DIV = LinearSegmentedColormap.from_list(
    "brand_div", [SECONDARY, BG_COLOR, PRIMARY]
)

# Gif frames must be handed this so they do not fall back to white on
# transparent: PillowWriter(savefig_kwargs=PILLOW_SAVEFIG_KWARGS).
PILLOW_SAVEFIG_KWARGS = {"facecolor": BG_COLOR}

# Log axes carry gridlines at this weight only.
GRID_KWARGS = {"which": "both", "alpha": 0.3}
