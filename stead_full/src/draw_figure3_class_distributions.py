# draw_figure3_class_distributions.py
# ------------------------------------------------------------
# Draw Figure 3:
# Class distributions before and after balancing
#
# - Top row: original binary, magnitude, and distance distributions
# - Bottom row: balanced binary, magnitude, and distance distributions
#
# The value labels are horizontal (not vertical), larger, and placed
# slightly above the bars.
#
# The figure is saved at high resolution (800 dpi).
# ------------------------------------------------------------

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# =========================
# Paths
# =========================
CSV_PATH = os.path.expanduser("~/STEAD_FULL/merge.csv")   # <-- change if needed
OUT_DIR = os.path.expanduser("~/DLHW2Par8/stead_full/results/figures")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PATH = os.path.join(OUT_DIR, "figure3_class_distributions.png")

# =========================
# Plot settings
# =========================
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["axes.titlesize"] = 16
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12
mpl.rcParams["figure.dpi"] = 100

# Colors
c_bin = ["#8fb3c9", "#eea564"]
c_mag = ["#b0c4d8", "#b5d3ab", "#bebecf", "#ecc493"]
c_dist = ["#b0c4d8", "#b5d3ab", "#ecc493", "#e8ab8f"]

# =========================
# Helper function
# =========================
def add_bar_labels(ax, bars, fontsize=12, extra_pad_ratio=0.02):
    """
    Add horizontal value labels above bars.
    """
    ymax = ax.get_ylim()[1]
    pad = ymax * extra_pad_ratio

    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + pad,
            f"{int(h)}",
            ha="center",
            va="bottom",
            rotation=0,          # horizontal labels
            fontsize=fontsize
        )

def plot_bar(ax, labels, values, colors, title, ylabel=None, rotate=0):
    bars = ax.bar(labels, values, color=colors, edgecolor="0.25", linewidth=0.8)

    ax.set_title(title, pad=8)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.tick_params(axis="x", rotation=rotate)
    ax.grid(False)

    # Make room for labels
    ymax = max(values)
    ax.set_ylim(0, ymax * 1.15)

    add_bar_labels(ax, bars, fontsize=11, extra_pad_ratio=0.015)

    return bars

# =========================
# Read data
# =========================
df = pd.read_csv(CSV_PATH, low_memory=False)

# Keep only needed categories
df = df[df["trace_category"].isin(["noise", "earthquake_local"])].copy()

# =========================
# Original binary counts
# =========================
orig_binary = df["trace_category"].value_counts()
orig_binary = orig_binary.reindex(["noise", "earthquake_local"])

orig_binary_labels = ["Noise", "Earthquake"]
orig_binary_values = orig_binary.values.tolist()

# =========================
# Original magnitude counts
# Use only earthquake records with valid magnitudes
# =========================
eq = df[df["trace_category"] == "earthquake_local"].copy()
eq = eq[pd.to_numeric(eq["source_magnitude"], errors="coerce").notna()].copy()
eq["source_magnitude"] = pd.to_numeric(eq["source_magnitude"], errors="coerce")

mag0 = (eq["source_magnitude"] < 2).sum()
mag1 = ((eq["source_magnitude"] >= 2) & (eq["source_magnitude"] < 3)).sum()
mag2 = ((eq["source_magnitude"] >= 3) & (eq["source_magnitude"] < 4)).sum()
mag3 = (eq["source_magnitude"] >= 4).sum()

orig_mag_labels = [r"$M<2$", r"$2\leq M<3$", r"$3\leq M<4$", r"$M\geq4$"]
orig_mag_values = [mag0, mag1, mag2, mag3]

# =========================
# Original distance counts
# Use only earthquake records with valid distances
# =========================
eq = eq[pd.to_numeric(eq["source_distance_km"], errors="coerce").notna()].copy()
eq["source_distance_km"] = pd.to_numeric(eq["source_distance_km"], errors="coerce")

dist0 = (eq["source_distance_km"] < 50).sum()
dist1 = ((eq["source_distance_km"] >= 50) & (eq["source_distance_km"] < 100)).sum()
dist2 = ((eq["source_distance_km"] >= 100) & (eq["source_distance_km"] < 200)).sum()
dist3 = (eq["source_distance_km"] >= 200).sum()

orig_dist_labels = [r"$d<50$", r"$50\leq d<100$", r"$100\leq d<200$", r"$d\geq200$"]
orig_dist_values = [dist0, dist1, dist2, dist3]

# =========================
# Balanced binary counts
# =========================
n_noise = (df["trace_category"] == "noise").sum()
n_eq = (df["trace_category"] == "earthquake_local").sum()
n_bin = min(n_noise, n_eq)

bal_binary_labels = ["Noise", "Earthquake"]
bal_binary_values = [n_bin, n_bin]

# =========================
# Balanced magnitude counts
# =========================
n_mag = min(orig_mag_values)
bal_mag_labels = orig_mag_labels
bal_mag_values = [n_mag, n_mag, n_mag, n_mag]

# =========================
# Balanced distance counts
# =========================
n_dist = min(orig_dist_values)
bal_dist_labels = orig_dist_labels
bal_dist_values = [n_dist, n_dist, n_dist, n_dist]

# =========================
# Draw figure
# =========================
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# Top row: original
plot_bar(
    axes[0, 0],
    orig_binary_labels,
    orig_binary_values,
    c_bin,
    "Original Binary Distribution",
    ylabel="Number of samples",
    rotate=0
)

plot_bar(
    axes[0, 1],
    orig_mag_labels,
    orig_mag_values,
    c_mag,
    "Original Magnitude Distribution",
    rotate=15
)

plot_bar(
    axes[0, 2],
    orig_dist_labels,
    orig_dist_values,
    c_dist,
    "Original Distance Distribution",
    rotate=15
)

# Bottom row: balanced
plot_bar(
    axes[1, 0],
    bal_binary_labels,
    bal_binary_values,
    c_bin,
    "Balanced Binary Distribution",
    ylabel="Number of samples",
    rotate=0
)

plot_bar(
    axes[1, 1],
    bal_mag_labels,
    bal_mag_values,
    c_mag,
    "Balanced Magnitude Distribution",
    rotate=15
)

plot_bar(
    axes[1, 2],
    bal_dist_labels,
    bal_dist_values,
    c_dist,
    "Balanced Distance Distribution",
    rotate=15
)

# Tidy layout
plt.tight_layout()

# Save figure at high resolution
plt.savefig(OUT_PATH, dpi=800, bbox_inches="tight")
plt.show()

print("Figure saved to:")
print(OUT_PATH)