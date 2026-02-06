# DLHW2 — CPE 487/587 Homework 2 (Training a Neural Network)

This repository contains the Homework 2 implementation for CPE 487/587.
It extends the UV-based package structure from Homework 01 and adds weight-matrix visualization
animations for the learned weights.

---

## Project Structure

```text
DLHW2/
├─ pyproject.toml
├─ README.md
├─ uv.lock
├─ scripts/
│  ├─ binaryclassification_impl.py
│  ├─ binaryclassification_animate_impl.py
├─ src/
│  └─ mchnpkg/
│     ├─ __init__.py
│     ├─ deepl/
│     │  ├─ __init__.py
│     │  └─ two_layer_binary_classification.py
│     └─ animation/
│        ├─ __init__.py
│        ├─ weight_animation.py
│        └─ largewt_animation.py
└─ .gitignore

Install (UV)

From the project root:

uv sync

Why do we need clone() when saving weight history?

When we store weights at each epoch, we must freeze a snapshot of the tensor at that moment.
If we save without clone(), the saved tensor can still share underlying storage or be affected by later updates.
That can cause earlier history entries to accidentally reflect newer weight values.

Using:

W.detach().cpu().clone()

creates an independent copy in new memory, so each W*_hist[i] truly contains the weight values from epoch i.
This is required to build a correct animation over time.

How to execute HW02Q7 and generate .mp4 animations

This part generates 4 animations (one per weight matrix W1–W4) using the 3D weight histories
returned by binary_classification().

1) Run the animation script

From the project root:

uv run python scripts/binaryclassification_animate_impl.py


The script trains the model and then creates 4 videos using:

animate_weight_heatmap(W1_hist, ...)

animate_weight_heatmap(W2_hist, ...)

animate_weight_heatmap(W3_hist, ...)

animate_weight_heatmap(W4_hist, ...)

2) Where the .mp4 files are saved

Manim writes outputs under:

./media/videos/
