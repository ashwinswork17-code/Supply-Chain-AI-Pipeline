"""
============================================================================
viz_style.py — Shared publication-quality visual identity for the
Supply Chain AI Pipeline (DMAIC).
----------------------------------------------------------------------------
One house style applied to every figure so the ~25 charts read as a single
coherent deliverable (dissertation / executive deck / portfolio).

Design choices (deliberate, not matplotlib defaults):
  - Palette: a cool slate-navy base with a controlled accent set. Each DMAIC
    phase gets ONE signature accent so a reader can tell the phase from the
    colour alone. Risk/loss are always the same warm red; good/positive is
    always the same green — semantic colour, used consistently.
  - Typography: DejaVu Sans (ships with matplotlib, so it renders anywhere)
    with a clear weight hierarchy — heavy titles, medium labels, light notes.
  - Every figure carries a title, a subtitle stating the business question,
    and a footnote with the data source + an interpretation line.
============================================================================
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import FuncFormatter
import numpy as np

# ---- Core palette -----------------------------------------------------------
INK        = "#1d2733"   # near-black slate for text
INK_SOFT   = "#5b6b7b"   # secondary text
GRID       = "#e3e8ee"   # hairline grid
PANEL      = "#ffffff"   # panel background
PAGE       = "#f7f9fb"   # figure background

# Semantic colours (meaning is fixed across the whole deck)
GOOD       = "#1a9d6f"   # positive / on-time / profit
BAD        = "#d64545"   # risk / late / loss / fraud
WARN       = "#e8a33d"   # caution / mid
NEUTRAL    = "#3d6a99"   # generic primary series

# One signature accent per DMAIC phase
PHASE_ACCENT = {
    "define":  "#3d6a99",  # steel blue
    "measure": "#2e8b8b",  # teal
    "analyze": "#6a5acd",  # slate violet
    "improve": "#c2603e",  # burnt sienna
    "control": "#4a7c59",  # pine green
}

# A categorical sequence for multi-series charts (markets, segments, modes)
CATEGORICAL = ["#3d6a99", "#c2603e", "#2e8b8b", "#e8a33d",
               "#6a5acd", "#9d4e4e", "#5b8c5a", "#b0758a"]

# Sequential / diverging maps for heatmaps
SEQ_CMAP  = "GnBu"
DIV_CMAP  = "RdBu_r"


def set_house_style():
    """Apply global rcParams so every figure inherits the identity."""
    plt.rcParams.update({
        "figure.facecolor":  PAGE,
        "axes.facecolor":    PANEL,
        "savefig.facecolor": PAGE,
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "text.color":        INK,
        "axes.edgecolor":    GRID,
        "axes.labelcolor":   INK,
        "axes.titlecolor":   INK,
        "axes.linewidth":    1.0,
        "axes.grid":         True,
        "grid.color":        GRID,
        "grid.linewidth":    0.8,
        "xtick.color":       INK_SOFT,
        "ytick.color":       INK_SOFT,
        "xtick.labelsize":   9.5,
        "ytick.labelsize":   9.5,
        "legend.frameon":    False,
        "legend.fontsize":   9.5,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


def titled(fig, title, subtitle=None, accent=NEUTRAL):
    """Heavy title + business-question subtitle in a reserved header band.

    Call AFTER fig.tight_layout(rect=[0, 0.06, 1, 0.88]) so the header band
    (top 12%) and footnote band (bottom 6%) are already reserved.
    """
    fig.suptitle(title, fontsize=15, fontweight="bold", color=INK,
                 x=0.012, ha="left", y=0.985, va="top")
    if subtitle:
        fig.text(0.012, 0.945, subtitle, fontsize=10.3, color=INK_SOFT,
                 ha="left", va="top", style="italic")
    fig.add_artist(plt.Line2D([0.012, 0.16], [0.905, 0.905],
                   color=accent, linewidth=3.2,
                   transform=fig.transFigure, solid_capstyle="round"))


def footnote(fig, text):
    """Source + interpretation footnote, bottom-left."""
    fig.text(0.012, 0.012, text, fontsize=8.3, color=INK_SOFT, ha="left")


def annotate_insight(ax, text, xy=(0.98, 0.03), ha="right", va="bottom",
                     accent=NEUTRAL):
    """A small boxed 'business insight' callout inside an axis."""
    ax.text(*xy, text, transform=ax.transAxes, fontsize=9, color=INK,
            ha=ha, va=va,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff",
                      edgecolor=accent, linewidth=1.2, alpha=0.95))


def style_ax(ax, title=None, xlabel=None, ylabel=None):
    if title:  ax.set_title(title, fontsize=11.5, fontweight="bold", pad=8, loc="left")
    if xlabel: ax.set_xlabel(xlabel, fontsize=10, fontweight="medium")
    if ylabel: ax.set_ylabel(ylabel, fontsize=10, fontweight="medium")
    ax.tick_params(length=0)
    return ax


def thousands(x, pos):
    if abs(x) >= 1e6: return f"{x/1e6:.1f}M"
    if abs(x) >= 1e3: return f"{x/1e3:.0f}K"
    return f"{x:.0f}"

K_FMT = FuncFormatter(thousands)


def save(fig, path, dpi=150):
    # Reserve header (top) and footnote (bottom) bands without the coordinate
    # recomputation that bbox_inches='tight' would apply to suptitle/text.
    try:
        fig.tight_layout(rect=[0, 0.055, 1, 0.875])
    except Exception:
        fig.subplots_adjust(top=0.875, bottom=0.075)
    fig.savefig(path, dpi=dpi, facecolor=PAGE)
    plt.close(fig)
    return path