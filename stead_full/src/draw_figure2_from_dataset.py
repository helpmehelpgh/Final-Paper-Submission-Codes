# draw_figure2_from_dataset.py
# Draw Figure 2 directly from the STEAD dataset at 800 dpi

import os
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl

# -----------------------------
# Paths
# -----------------------------
CSV_PATH = os.path.expanduser("~/STEAD_FULL/merge.csv")
HDF5_PATH = os.path.expanduser("~/STEAD_FULL/merge.hdf5")
OUT_PATH = os.path.expanduser(
    "~/DLHW2Par8/stead_full/results/figures/figure2_representative_waveforms.png"
)

# -----------------------------
# Plot style
# -----------------------------
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["axes.unicode_minus"] = False

TITLE_FS = 18
LABEL_FS = 16
TICK_FS = 13
COMP_FS = 16

# -----------------------------
# Utilities
# -----------------------------
def pad_or_trim_to_70s(x, fs=100):
    n_target = int(70 * fs)
    n = x.shape[0]
    if n >= n_target:
        return x[:n_target, :]
    out = np.zeros((n_target, x.shape[1]), dtype=x.dtype)
    out[:n, :] = x
    return out

def normalize_channel(y):
    m = np.max(np.abs(y))
    if m > 0:
        return y / m
    return y

def get_trace_from_hdf5(h5, trace_name):
    x = h5["data"][trace_name][()]
    x = np.asarray(x)

    if x.ndim != 2:
        raise ValueError(f"Trace {trace_name} does not have 2 dimensions: got {x.shape}")

    # If stored as (3, n), transpose to (n, 3)
    if x.shape[0] == 3 and x.shape[1] != 3:
        x = x.T

    if x.shape[1] != 3:
        raise ValueError(f"Trace {trace_name} does not have 3 components after adjustment: got {x.shape}")

    return x

def plot_three_component(ax, x, title, colors=("royalblue", "forestgreen", "red"),
                         labels=("N", "W", "E"), fs=100, show_xlabel=True,
                         show_component_labels=True):
    x = pad_or_trim_to_70s(x, fs=fs)

    x_norm = np.zeros_like(x, dtype=float)
    for i in range(3):
        x_norm[:, i] = normalize_channel(x[:, i])

    t = np.arange(x_norm.shape[0]) / fs

    offsets = np.array([2.0, 1.0, 0.0])
    amp_scale = 0.45

    for i, (c, lab, off) in enumerate(zip(colors, labels, offsets)):
        y = x_norm[:, i] * amp_scale + off
        ax.plot(t, y, color=c, lw=0.8)

        if show_component_labels:
            # slightly above the line
            ax.text(
                0.8, off + 0.12, lab,
                color=c, fontsize=COMP_FS, fontweight="bold",
                ha="left", va="center"
            )

    ax.set_title(title, fontsize=TITLE_FS, pad=8)
    ax.set_xlim(0, 70)
    ax.set_ylim(-0.25, 2.65)
    ax.set_yticks([])

    if show_xlabel:
        ax.set_xlabel("Time (s)", fontsize=LABEL_FS)
        ax.tick_params(axis="x", labelsize=TICK_FS)
    else:
        ax.set_xlabel("")
        ax.set_xticks([])

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

# -----------------------------
# Select representative records
# -----------------------------
df = pd.read_csv(CSV_PATH)

noise_df = df[df["trace_category"] == "noise"].copy()
eq_df = df[df["trace_category"] == "earthquake_local"].copy()

m1 = eq_df[eq_df["source_magnitude"] < 2].copy()
m2 = eq_df[(eq_df["source_magnitude"] >= 2) & (eq_df["source_magnitude"] < 3)].copy()
m4 = eq_df[eq_df["source_magnitude"] >= 4].copy()

d1 = eq_df[eq_df["source_distance_km"] < 50].copy()
d2 = eq_df[(eq_df["source_distance_km"] >= 50) & (eq_df["source_distance_km"] < 100)].copy()
d3 = eq_df[(eq_df["source_distance_km"] >= 100) & (eq_df["source_distance_km"] < 200)].copy()
d4 = eq_df[eq_df["source_distance_km"] >= 200].copy()

def choose_record(g):
    if len(g) == 0:
        return None
    return g.iloc[len(g) // 2]

noise_rec = choose_record(noise_df.reset_index(drop=True))
m1_rec    = choose_record(m1.sort_values("source_magnitude").reset_index(drop=True))
m2_rec    = choose_record(m2.sort_values("source_magnitude").reset_index(drop=True))
m4_rec    = choose_record(m4.sort_values("source_magnitude").reset_index(drop=True))

d1_rec    = choose_record(d1.sort_values("source_distance_km").reset_index(drop=True))
d2_rec    = choose_record(d2.sort_values("source_distance_km").reset_index(drop=True))
d3_rec    = choose_record(d3.sort_values("source_distance_km").reset_index(drop=True))
d4_rec    = choose_record(d4.sort_values("source_distance_km").reset_index(drop=True))

# -----------------------------
# Draw figure
# -----------------------------
fig, axes = plt.subplots(2, 4, figsize=(16, 9))
plt.subplots_adjust(wspace=0.10, hspace=0.14)

titles_top = ["Noise", r"$M<2$", r"$2\leq M<3$", r"$M\geq 4$"]
titles_bottom = [
    r"$d<50\ \mathrm{km}$",
    r"$50\leq d<100\ \mathrm{km}$",
    r"$100\leq d<200\ \mathrm{km}$",
    r"$d\geq 200\ \mathrm{km}$"
]

records_top = [noise_rec, m1_rec, m2_rec, m4_rec]
records_bottom = [d1_rec, d2_rec, d3_rec, d4_rec]

with h5py.File(HDF5_PATH, "r") as h5:
    for j, (rec, title) in enumerate(zip(records_top, titles_top)):
        trace_name = rec["trace_name"]
        x = get_trace_from_hdf5(h5, trace_name)
        plot_three_component(
            axes[0, j], x, title,
            labels=("N", "W", "E"),
            show_xlabel=False,
            show_component_labels=True
        )

    for j, (rec, title) in enumerate(zip(records_bottom, titles_bottom)):
        trace_name = rec["trace_name"]
        x = get_trace_from_hdf5(h5, trace_name)
        plot_three_component(
            axes[1, j], x, title,
            labels=("N", "W", "E"),
            show_xlabel=True,
            show_component_labels=True
        )

fig.tight_layout(pad=0.8)
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fig.savefig(OUT_PATH, dpi=800, bbox_inches="tight")
plt.show()

print(f"Figure saved to:\n{OUT_PATH}")