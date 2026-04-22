# Final Paper Submission Codes

## Project Title
**Deep Learning-Based Engineering Characterization of Seismic Waveforms for Ground-Motion Screening**

## Overview
This repository contains the Python codes developed for the paper on engineering-oriented seismic waveform screening using the **STanford EArthquake Dataset (STEAD)**. The study uses a **one-dimensional convolutional neural network (1D-CNN)** to classify raw three-component seismic waveforms for three tasks:

1. **Binary earthquake-versus-noise classification**
2. **Balanced four-class magnitude classification**
3. **Balanced four-class source-to-station distance classification**

The main objective is to demonstrate that raw seismic waveforms can be used not only for signal discrimination but also for broader engineering-oriented source characterization before more computationally expensive structural or seismic-response analyses are performed.

---

## Repository Structure

```text
DLHW2Par8/
│
├── stead_full/
│   ├── src/
│   │   ├── config_full.py
│   │   ├── dataset_full.py
│   │   ├── model.py
│   │   ├── model_shallow.py
│   │   ├── inspect_full.py
│   │   ├── prepare_full_metadata.py
│   │   ├── train_binary_full.py
│   │   ├── evaluate_binary_full.py
│   │   ├── train_magnitude_full_balanced.py
│   │   ├── evaluate_magnitude_full_balanced.py
│   │   ├── train_distance_full_balanced.py
│   │   ├── evaluate_distance_full_balanced.py
│   │   ├── draw_figure2_from_dataset.py
│   │   ├── draw_figure3_class_distributions.py
│   │   ├── draw_figure4_training_history.py
│   │   ├── draw_combined_confusion_matrices.py
│   │   ├── Binary_full.py
│   │   ├── Figure_6.py
│   │   ├── Figure_7.py
│   │   └── Figure_8.py
│   │
│   └── results/
│       ├── metadata/
│       ├── models/
│       ├── figures/
│       └── tables/
│
└── README.md
```

---

## Dataset

This project uses the **merged STEAD dataset**, which contains:

- **1,265,657** total waveform records
- **1,030,231** local-earthquake traces
- **235,426** noise traces
- **3 waveform components**
- **6000 samples per waveform**
- **100 Hz sampling rate**

### Main metadata fields used
- `trace_category`
- `trace_name`
- `source_magnitude`
- `source_distance_km`
- `p_arrival_sample`
- `s_arrival_sample`
- `snr_db`

### Dataset location used in the project
The codes assume the STEAD files are stored on Lovelace in:

```text
~/STEAD_FULL/
```

with:
- `merge.csv`
- `merge.hdf5`

---

## Tasks Implemented

### 1. Binary Classification
Classifies a waveform as:
- **Noise**
- **Earthquake**

### 2. Magnitude Classification
Balanced four-class magnitude classification:
- `M < 2`
- `2 ≤ M < 3`
- `3 ≤ M < 4`
- `M ≥ 4`

### 3. Distance Classification
Balanced four-class source-to-station distance classification:
- `d < 50 km`
- `50 ≤ d < 100 km`
- `100 ≤ d < 200 km`
- `d ≥ 200 km`

---

## Main Model

The primary model used in the study is a **1D-CNN** that takes raw three-component seismic waveforms as input. The model is trained separately for the three tasks above using the same general architecture, with the output layer adapted to the corresponding number of classes.

A shallower baseline model is also included in:

```text
stead_full/src/model_shallow.py
```

---

## Environment and Dependencies

The project was developed in **Python 3.12** and uses:

- `numpy`
- `pandas`
- `matplotlib`
- `h5py`
- `torch`
- `scikit-learn`

Install the main dependencies with:

```bash
pip install numpy pandas matplotlib h5py torch scikit-learn
```

---

## How to Run

### Step 1: Prepare metadata
Generate the metadata files needed for training and evaluation:

```bash
python stead_full/src/prepare_full_metadata.py
```

This creates balanced and task-specific metadata files inside:

```text
stead_full/results/metadata/
```

---

### Step 2: Inspect the dataset
Optional check of the merged STEAD dataset:

```bash
python stead_full/src/inspect_full.py
```

---

### Step 3: Train the models

#### Binary classification
```bash
python stead_full/src/train_binary_full.py
```

#### Magnitude classification
```bash
python stead_full/src/train_magnitude_full_balanced.py
```

#### Distance classification
```bash
python stead_full/src/train_distance_full_balanced.py
```

Saved trained models are written to:

```text
stead_full/results/models/
```

---

### Step 4: Evaluate the models

#### Binary evaluation
```bash
python stead_full/src/evaluate_binary_full.py
```

#### Magnitude evaluation
```bash
python stead_full/src/evaluate_magnitude_full_balanced.py
```

#### Distance evaluation
```bash
python stead_full/src/evaluate_distance_full_balanced.py
```

These scripts generate:
- classification reports
- confusion matrices
- performance summaries

Outputs are saved in:

```text
stead_full/results/figures/
stead_full/results/tables/
```

---

## Figure Generation

The repository also includes scripts used to generate the figures reported in the manuscript.


Generated figures are saved in:

```text
stead_full/results/figures/
```

---

## Results Summary

Main reported results from the paper:

- **Binary earthquake-versus-noise classification**  
  Validation accuracy: **98.84%**

- **Balanced magnitude classification**  
  Validation accuracy: **65.66%**

- **Balanced distance classification**  
  Validation accuracy: **71.46%**

These results indicate a clear hierarchy of task difficulty:
- binary classification is easiest,
- distance classification is moderately difficult,
- magnitude classification is the most challenging.

The results also show that **balanced learning is important**, since imbalanced datasets can produce misleadingly strong overall accuracy while masking weak performance in less frequent classes.

---

## Where the Main Outputs Exist

### Metadata files
```text
stead_full/results/metadata/
```

### Trained models
```text
stead_full/results/models/
```

### Figures
```text
stead_full/results/figures/
```

### Tables / summaries
```text
stead_full/results/tables/
```