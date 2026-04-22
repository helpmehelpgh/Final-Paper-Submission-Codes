import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# -------------------------------------------------
# Font settings
# Use installed Times-like font to avoid warnings
# -------------------------------------------------
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['mathtext.fontset'] = 'stix'

# -------------------------------------------------
# Confusion matrices (counts)
# -------------------------------------------------
cm_binary = np.array([
    [47167,   91],
    [ 1001, 45912]
])

cm_magnitude = np.array([
    [4001,  732,   37,   12],
    [ 856, 2814,  872,  167],
    [  49,  952, 2166, 1553],
    [  10,  212, 1126, 3598]
])

cm_distance = np.array([
    [3939,  570,  130,   29],
    [ 861, 2775, 1041,  101],
    [ 106,  990, 2658,  919],
    [  30,   43,  555, 4083]
])

# -------------------------------------------------
# Row-normalized confusion matrices (%)
# -------------------------------------------------
def normalize_rows(cm):
    return cm / cm.sum(axis=1, keepdims=True) * 100.0

cm_binary_norm = normalize_rows(cm_binary)
cm_magnitude_norm = normalize_rows(cm_magnitude)
cm_distance_norm = normalize_rows(cm_distance)

# -------------------------------------------------
# Labels
# -------------------------------------------------
binary_labels = ['Noise', 'Earthquake']
mag_labels = [r'$M<2$', r'$2\leq M<3$', r'$3\leq M<4$', r'$M\geq 4$']
dist_labels = [r'$d<50$', r'$50\leq d<100$', r'$100\leq d<200$', r'$d\geq 200$']

# -------------------------------------------------
# Plot helper
# -------------------------------------------------
def plot_cm(ax, cm, labels, title, is_percent=False, cmap='viridis'):
    if is_percent:
        im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=100)
    else:
        im = ax.imshow(cm, cmap=cmap)

    ax.set_title(title, fontsize=13, pad=8)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=10, rotation=25, ha='right')
    ax.set_yticklabels(labels, fontsize=10)

    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)

    if is_percent:
        threshold = 50
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                ax.text(
                    j, i, f'{val:.1f}%',
                    ha='center', va='center',
                    color='white' if val < threshold else 'black',
                    fontsize=10
                )
    else:
        threshold = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                ax.text(
                    j, i, f'{val:d}',
                    ha='center', va='center',
                    color='white' if val < threshold else 'black',
                    fontsize=10
                )

    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.8)
    ax.tick_params(which='minor', bottom=False, left=False)

    return im

# -------------------------------------------------
# Create figure
# -------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=800)

im1 = plot_cm(axes[0, 0], cm_binary, binary_labels, 'Binary (Counts)', is_percent=False)
im2 = plot_cm(axes[0, 1], cm_magnitude, mag_labels, 'Magnitude (Counts)', is_percent=False)
im3 = plot_cm(axes[0, 2], cm_distance, dist_labels, 'Distance (Counts)', is_percent=False)

im4 = plot_cm(axes[1, 0], cm_binary_norm, binary_labels, 'Binary (Normalized %)', is_percent=True)
im5 = plot_cm(axes[1, 1], cm_magnitude_norm, mag_labels, 'Magnitude (Normalized %)', is_percent=True)
im6 = plot_cm(axes[1, 2], cm_distance_norm, dist_labels, 'Distance (Normalized %)', is_percent=True)

cbar1 = fig.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
cbar2 = fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
cbar3 = fig.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
cbar4 = fig.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)
cbar5 = fig.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.04)
cbar6 = fig.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)

for cbar in [cbar1, cbar2, cbar3, cbar4, cbar5, cbar6]:
    cbar.ax.tick_params(labelsize=9)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.95,
                    wspace=0.45, hspace=0.45)

out_dir = 'stead_full/results/figures'
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'combined_confusion_matrices.png')

plt.savefig(out_path, dpi=800, bbox_inches='tight')
plt.show()

print(f'Figure saved to:\n{out_path}')