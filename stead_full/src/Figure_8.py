import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# -----------------------------
# Data
# -----------------------------
tasks = ['Binary', 'Magnitude', 'Distance']
accuracy = [0.9884, 0.6566, 0.7146]
macro_f1 = [0.9884, 0.6544, 0.7145]

x = np.arange(len(tasks))
width = 0.35

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
out_path = os.path.join(out_dir, "figure8_performance_summary.png")

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=800)

bars1 = ax.bar(x - width/2, accuracy, width, label='Validation Accuracy', color='tab:blue')
bars2 = ax.bar(x + width/2, macro_f1, width, label='Macro F1-score', color='tab:orange')

ax.set_ylabel('Score', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=12)
ax.set_ylim(0, 1.08)
ax.tick_params(axis='y', labelsize=12)
ax.legend(frameon=True, fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.4)

# Value labels
for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.015,
                f'{h:.4f}', ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig(out_path, dpi=800, bbox_inches='tight')
print("Saved to:", out_path)
plt.show()