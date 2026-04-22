import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# -----------------------------
# Confusion matrix values
# -----------------------------
cm = np.array([
    [47167,   91],
    [ 1001, 45912]
])

classes = ['Noise', 'Earthquake']

# -----------------------------
# Font settings
# -----------------------------
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
mpl.rcParams['mathtext.fontset'] = 'stix'

# -----------------------------
# Output path
# -----------------------------
out_dir = os.path.expanduser("~/DLHW2Par8/stead_full/results/figures")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "figure5_binary_confusion_matrix.png")

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=800)

im = ax.imshow(cm, cmap='viridis')

# Axis labels and ticks
ax.set_xlabel('Predicted Label', fontsize=14)
ax.set_ylabel('True Label', fontsize=14)
ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(classes, fontsize=12)
ax.set_yticklabels(classes, fontsize=12)

# Annotate values inside cells
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j, i, f'{cm[i, j]}',
            ha='center', va='center',
            color='black', fontsize=12
        )

# Colorbar
cbar = fig.colorbar(im, ax=ax)
cbar.ax.tick_params(labelsize=11)

plt.tight_layout()
plt.savefig(out_path, dpi=800, bbox_inches='tight')
print("Saved to:", out_path)
plt.show()