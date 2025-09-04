# HEVC Intra-Prediction Partitioning: Visualization and Model Inference

## Overview

This project provides a comprehensive suite of Python scripts for analyzing, visualizing, and running AI-based predictions on a specialized dataset for HEVC (High Efficiency Video Coding) intra-prediction complexity reduction. The tools allow for a deep dive into the ground truth CU (Coding Unit) partition data and a direct comparison against the outputs of pre-trained TensorFlow (CNN) and PyTorch (Vision Transformer) models.

## Understanding the Dataset

The core of this project is a large dataset stored in `.dat` binary files.The structure of these files was determined by reverse-engineering the provided data loading scripts (`input_data.py`, `net_CTU64.py`).

### Sample Structure

The `.dat` file is a binary file where data is concatenated without any separators.Each sample, representing a 64x64 video block, is stored in a **4992-byte chunk**.

A single 4992-byte sample is broken down as follows:
* **Bytes 0–4095 (4096 bytes):** The 64x64 pixel data for the image block, with each pixel being a single unsigned 8-bit integer (`uint8`).
* **Bytes 4096–4159 (64 bytes):** An auxiliary data block that is skipped by the data loading scripts.
* **Bytes 4160–4991 (832 bytes):** The label data, which contains 52 consecutive 16-byte labels.Each label corresponds to a specific Quantization Parameter (QP) from 0 to 51.

### Label Data (CU Depth)

The 16 numbers in a label array correspond to the sixteen 16x16 sub-regions of the 64x64 image.The value indicates the final CU split depth:
* **`3`**: The 16x16 region is split into **8x8** CUs (high detail).
* **`2`**: The region is a final **16x16** CU.
* **`1`**: The region is part of a larger **32x32** CU (simple area).
* **`0`**: The region is part of a single **64x64** CU (very simple area).

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
Visualizing the ground truth for a single 64x64 sample from the dataset.
* **Script:** `visualize_partitions_on_sample.py`
* **Output for Sample #50, QP=22:** 
<img width="389" height="411" alt="partition_sample_51_qp_22" src="https://github.com/user-attachments/assets/957bb3a1-8f95-46f5-9e43-97f4ac7e2285" />


#### Full Frame Partitions
Visualizing the complete ground truth partition map for a full frame.
* **Script:** `visualize_full_frame.py`
* **Output for `IntraTrain_768x512`, Frame 3, QP=22:**.
  <img width="887" height="591" alt="frame_3_IntraTrain_768x512_qp_22" src="https://github.com/user-attachments/assets/e18e364d-9a6f-4979-993f-dea54139752d" />

### 2. Model Prediction vs. Ground Truth

#### Single Sample Comparison
Running inference on a single 64x64 block and comparing the model's prediction to the ground truth.

**ViT Model Prediction (Sample #51, QP=22):**
* **Ground Truth:** `[1 1 2 2 1 1 3 3 3 2 3 3 3 3 3 3]` 
* **Prediction:** `[2 2 2 2 2 2 2 3 2 2 2 3 2 2 3 3]` 
<img width="950" height="465" alt="vit_pred_sample_51_qp_22" src="https://github.com/user-attachments/assets/8a5e455f-90ff-489b-9b4c-d4ab9f38bbc3" />

**CNN Model Prediction (Sample #51, QP=22):**
* **Ground Truth:** `[1 1 2 2 1 1 3 3 3 2 3 3 3 3 3 3]`
* **Prediction:** `[3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3]`
<img width="950" height="465" alt="cnn_pred_sample_51_qp_22" src="https://github.com/user-attachments/assets/67ad6783-5153-4915-8324-f98cda10f513" />

#### Full Frame Comparison
Running inference on a full frame and comparing the complete predicted partition map to the ground truth.

**ViT Model Full Frame Prediction:**
<img width="2304" height="1536" alt="full_frame_comparison_qp22" src="https://github.com/user-attachments/assets/f983945d-e4a6-43af-9032-9dba059dc470" />

**CNN Model Full Frame Prediction:**
<img width="2304" height="1536" alt="cnn_full_frame_comparison_qp22" src="https://github.com/user-attachments/assets/5abafb4d-779e-42fe-9a16-ee84fe351e6c" />

---
## Repository

For the complete code and data, please visit the GitHub repository:
[https://github.com/Krishna737Sharma/HEVC_CPH_Data_Visualization](https://github.com/Krishna737Sharma/HEVC_CPH_Data_Visualization)
