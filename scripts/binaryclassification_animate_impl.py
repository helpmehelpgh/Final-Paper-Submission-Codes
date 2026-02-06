from mchnpkg.deepl import binary_classification
from mchnpkg.animation import animate_large_heatmap


def main():
    # Assignment suggested params
    dt = 0.04
    epochs = 5000
    eta = 0.01
    d = 200
    n = 40000
    seed = 0

    # binary_classification returns:
    # W1, W2, W3, W4, losses, W1_hist, W2_hist, W3_hist, W4_hist
    _, _, _, _, losses, W1_hist, W2_hist, W3_hist, W4_hist = binary_classification(
        d=d,
        n=n,
        epochs=epochs,
        eta=eta,
        seed=seed,
    )

    # For large matrices, use animate_large_heatmap (fast, uses ImageObject)
    animate_large_heatmap(W1_hist, dt=dt, file_name="W1_evolution", title_str="W1 Evolution")
    animate_large_heatmap(W2_hist, dt=dt, file_name="W2_evolution", title_str="W2 Evolution")
    animate_large_heatmap(W3_hist, dt=dt, file_name="W3_evolution", title_str="W3 Evolution")
    animate_large_heatmap(W4_hist, dt=dt, file_name="W4_evolution", title_str="W4 Evolution")

    # optional: print final loss
    print(f"Final loss: {losses[-1]}")


if __name__ == "__main__":
    main()
