# draw_figure4_training_history.py

import os
import matplotlib.pyplot as plt

# -------------------------------
# Data from your training outputs
# -------------------------------

# Binary classification (5 epochs)
epochs_bin = [1, 2, 3, 4, 5]
train_loss_bin = [0.0993, 0.0664, 0.0545, 0.0441, 0.0320]
val_acc_bin    = [0.9815, 0.9771, 0.9876, 0.9870, 0.9884]

# Magnitude classification (10 epochs)
epochs_mag = [1,2,3,4,5,6,7,8,9,10]
train_loss_mag = [0.9557, 0.8907, 0.8633, 0.8441, 0.8333, 0.8150, 0.8054, 0.7969, 0.7877, 0.7790]
val_acc_mag    = [0.5745, 0.6090, 0.6251, 0.6371, 0.6105, 0.6332, 0.6437, 0.6499, 0.6440, 0.6566]

# Distance classification (10 epochs)
epochs_dist = [1,2,3,4,5,6,7,8,9,10]
train_loss_dist = [0.8764, 0.8008, 0.7747, 0.7547, 0.7415, 0.7275, 0.7168, 0.7008, 0.6942, 0.6868]
val_acc_dist    = [0.6520, 0.6556, 0.6737, 0.6192, 0.6889, 0.7064, 0.7073, 0.7146, 0.6787, 0.7010]

# -------------------------------
# Figure settings
# -------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 13
})

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=800)

# -------------------------------
# Left: Training Loss
# -------------------------------
ax = axes[0]
ax.plot(epochs_bin,  train_loss_bin,  marker='o', linewidth=2, markersize=5, label='Binary',     color='tab:blue')
ax.plot(epochs_mag,  train_loss_mag,  marker='s', linewidth=2, markersize=5, label='Magnitude',  color='tab:orange')
ax.plot(epochs_dist, train_loss_dist, marker='^', linewidth=2, markersize=5, label='Distance',   color='tab:green')

ax.set_xlabel('Epoch')
ax.set_ylabel('Training Loss')
ax.set_title('Training Loss History')
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend(frameon=True, fontsize=11)

# -------------------------------
# Right: Validation Accuracy
# -------------------------------
ax = axes[1]
ax.plot(epochs_bin,  val_acc_bin,  marker='o', linewidth=2, markersize=5, label='Binary',     color='tab:blue')
ax.plot(epochs_mag,  val_acc_mag,  marker='s', linewidth=2, markersize=5, label='Magnitude',  color='tab:orange')
ax.plot(epochs_dist, val_acc_dist, marker='^', linewidth=2, markersize=5, label='Distance',   color='tab:green')

ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Accuracy')
ax.set_title('Validation Accuracy History')
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend(frameon=True, fontsize=11)

# -------------------------------
# Layout and save
# -------------------------------
plt.tight_layout()

out_dir = os.path.join('stead_full', 'results', 'figures')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'figure4_training_history.png')

plt.savefig(out_path, dpi=800, bbox_inches='tight')
plt.show()

print(f'Figure saved to:\n{os.path.abspath(out_path)}')