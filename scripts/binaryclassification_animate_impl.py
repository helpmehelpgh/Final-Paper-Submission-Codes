import os
import torch

from mchnpkg.deepl import binary_classification
from mchnpkg.animation import animate_weight_heatmap, animate_large_heatmap


def main():
    # -----------------------------
    # Suggested parameters from HW
    # -----------------------------
    dt = 0.04
    epochs = 5000
    eta = 0.01
    d = 200
    n = 40000
    seed = 0

    # -----------------------------
    # Train and collect weight history
    # -----------------------------
    out = binary_classification(d=d, n=n, epochs=epochs, eta=eta, seed=seed)

    # unpack (must match your updated return order)
    W1, W2, W3, W4, losses, W1_hist, W2_hist, W3_hist, W4_hist = out

    print("Training done.")
    print(f"Final loss: {losses[-1]:.6f}")
    print("Hist shapes:")
    print("W1_hist:", tuple(W1_hist.shape))
    print("W2_hist:", tuple(W2_hist.shape))
    print("W3_hist:", tuple(W3_hist.shape))
    print("W4_hist:", tuple(W4_hist.shape))

    # -----------------------------
    # Decide which animation function to use
    # - animate_weight_heatmap is fine for smaller matrices
    # - animate_large_heatmap is for very large matrices (e.g., 1000x1000)
    # -----------------------------
    # Your matrices are:
    # W1: (d,48)   = (200,48)   OK for animate_weight_heatmap
    # W2: (48,16)  OK
    # W3: (16,32)  OK
    # W4: (32,1)   OK

    # Make sure history is float32 torch tensors (already is)
    W1_stack = W1_hist
    W2_stack = W2_hist
    W3_stack = W3_hist
    W4_stack = W4_hist

    # -----------------------------
    # Create 4 animations (4 movie files)
    # -----------------------------
    animate_weight_heatmap(
        W1_stack,
        dt=dt,
        file_name="W1_weights_animation",
        title_str="Weight Evolution: W1"
    )

    animate_weight_heatmap(
        W2_stack,
        dt=dt,
        file_name="W2_weights_animation",
        title_str="Weight Evolution: W2"
    )

    animate_weight_heatmap(
        W3_stack,
        dt=dt,
        file_name="W3_weights_animation",
        title_str="Weight Evolution: W3"
    )

    animate_weight_heatmap(
        W4_stack,
        dt=dt,
        file_name="W4_weights_animation",
        title_str="Weight Evolution: W4"
    )

    print("All animations requested. Check the Manim output under the media/ folder.")


if __name__ == "__main__":
    main()
