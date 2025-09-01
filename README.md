# HEVC Intra-Prediction Partitioning: Visualization and Model Inference

## Overview

This project provides a comprehensive suite of Python scripts for analyzing, visualizing, and running AI-based predictions on a specialized dataset for HEVC (High Efficiency Video Coding) intra-prediction complexity reduction. The tools allow for a deep dive into the ground truth CU (Coding Unit) partition data and a direct comparison against the outputs of pre-trained TensorFlow (CNN) and PyTorch (Vision Transformer) models.

## Understanding the Dataset

The core of this project is a large dataset stored in `.dat` binary files. [cite_start]The structure of these files was determined by reverse-engineering the provided data loading scripts (`input_data.py`, `net_CTU64.py`)[cite: 54].

### Sample Structure

[cite_start]The `.dat` file is a binary file where data is concatenated without any separators[cite: 43]. [cite_start]Each sample, representing a 64x64 video block, is stored in a **4992-byte chunk**[cite: 44].

[cite_start]A single 4992-byte sample is broken down as follows[cite: 45]:
* [cite_start]**Bytes 0–4095 (4096 bytes):** The 64x64 pixel data for the image block, with each pixel being a single unsigned 8-bit integer (`uint8`)[cite: 46, 47].
* [cite_start]**Bytes 4096–4159 (64 bytes):** An auxiliary data block that is skipped by the data loading scripts[cite: 48, 49].
* [cite_start]**Bytes 4160–4991 (832 bytes):** The label data, which contains 52 consecutive 16-byte labels[cite: 50]. [cite_start]Each label corresponds to a specific Quantization Parameter (QP) from 0 to 51[cite: 51].

### Label Data (CU Depth)

[cite_start]The 16 numbers in a label array correspond to the sixteen 16x16 sub-regions of the 64x64 image[cite: 105]. [cite_start]The value indicates the final CU split depth[cite: 110]:
* [cite_start]**`3`**: The 16x16 region is split into **8x8** CUs (high detail)[cite: 111, 112].
* [cite_start]**`2`**: The region is a final **16x16** CU[cite: 113].
* [cite_start]**`1`**: The region is part of a larger **32x32** CU (simple area)[cite: 114, 115].
* [cite_start]**`0`**: The region is part of a single **64x64** CU (very simple area)[cite: 116].

---
## Scripts and Usage

This repository contains several Python scripts for analysis and prediction. To use them, configure the path and parameter variables at the top of the desired file and run it from your terminal.

* **`visualize_partitions_on_sample.py`**: Reads a single 64x64 sample from a `.dat` file and visualizes its ground truth CU partitions.
* **`save_partition_map.py`**: Extracts and saves the full partition map for a single frame as a 2D text file.
* **`visualize_full_frame.py`**: Visualizes the complete ground truth partition map for an entire frame from a YUV file.
* **`predict_single_sample_vit.py` / `predict_single_sample_cnn.py`**: Loads a pre-trained model to predict partitions for a single 64x64 sample and generates a side-by-side comparison with the ground truth.
* **`predict_full_frame_vit.py` / `predict_full_frame_cnn.py`**: Runs a pre-trained model over an entire frame and generates a full side-by-side comparison of the predicted vs. ground truth partition maps.

---
## Results Showcase

### 1. Ground Truth Visualization

#### Single Sample Partitions
[cite_start]Visualizing the ground truth for a single 64x64 sample from the dataset[cite: 137].
* **Script:** `visualize_partitions_on_sample.py`
* [cite_start]**Output for Sample #50, QP=22:** [cite: 140, 141, 142]

![Ground Truth for Sample 50](https://i.imgur.com/L3oY1h0.png)

#### Full Frame Partitions
[cite_start]Visualizing the complete ground truth partition map for a full frame[cite: 168].
* **Script:** `visualize_full_frame.py`
* [cite_start]**Output for `IntraTrain_768x512`, Frame 3, QP=22:** [cite: 202]

![Full Frame Ground Truth Partitions](https://i.imgur.com/Y2E3dJc.png)

### 2. Model Prediction vs. Ground Truth

#### Single Sample Comparison
Running inference on a single 64x64 block and comparing the model's prediction to the ground truth.

[cite_start]**ViT Model Prediction (Sample #51, QP=22):** [cite: 153, 156]
* [cite_start]**Ground Truth:** `[1 1 2 2 1 1 3 3 3 2 3 3 3 3 3 3]` [cite: 154]
* [cite_start]**Prediction:** `[2 2 2 2 2 2 2 3 2 2 2 3 2 2 3 3]` [cite: 155]

![ViT Prediction vs. Ground Truth for Sample 51](https://i.imgur.com/qE4J3qM.png)

[cite_start]**CNN Model Prediction (Sample #51, QP=22):** [cite: 164, 167]
* [cite_start]**Ground Truth:** `[1 1 2 2 1 1 3 3 3 2 3 3 3 3 3 3]` [cite: 165]
* [cite_start]**Prediction:** `[3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3]` [cite: 166]

![CNN Prediction vs. Ground Truth for Sample 51](https://i.imgur.com/gK9f8vW.png)

#### Full Frame Comparison
Running inference on a full frame and comparing the complete predicted partition map to the ground truth.

[cite_start]**ViT Model Full Frame Prediction:** [cite: 203, 235]
![ViT Full Frame Prediction vs. Ground Truth](https://i.imgur.com/P4w5a9N.png)

[cite_start]**CNN Model Full Frame Prediction:** [cite: 236, 269]
![CNN Full Frame Prediction vs. Ground Truth](https://i.imgur.com/N6d7h4f.png)

---
## Repository

For the complete code and data, please visit the GitHub repository:
[cite_start][https://github.com/Krishna737Sharma/HEVC_CPH_Data_Visualization](https://github.com/Krishna737Sharma/HEVC_CPH_Data_Visualization) [cite: 136]
