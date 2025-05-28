# PcBD: A Novel Point Cloud Processing Flow for Boundary Detecting and De-Noising

![Graphical Abstract](./Graphical%20Abstract.png)

##  Overview

**PcBD** is a novel point cloud preprocessing pipeline designed for detecting projection boundaries and removing outliers in real-world 3D scans. The flow includes three main steps:  
- **Outlier Removal**  
- **Projection Boundary Detection**  
- **Boundary Smoothing**

This modular system provides a flexible preprocessing toolchain that enhances point cloud quality and structure for downstream tasks.

## Featured Application

The **PcBD** model can be flexibly integrated as a preprocessing module in applications such as:

- Wind tunnel testing  
- Depth camera-based object detection  
- High-fidelity point cloud modeling in robotics or SLAM

Researchers can:

- Use one or more of the provided modules as needed  
- Benchmark directly on the **Bound57 dataset**  
- Or generate new datasets based on the construction logic and source code of Bound57 for customized evaluation

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/desperadossy/PcBD.git
cd PcBD
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
cd PcBD
```

### 3. Prepare the dataset

#### Option 1: Download the provided Bound57 dataset

Run the following script to automatically download and extract the Bound57 dataset (`.tar.gz` format):

```bash
python data/download_bound57.py
```

#### Option 2: Generate your own dataset based on ShapeNetCore.v1

You can also generate your own customized dataset using ShapeNet as source:

1. Download the archive ShapeNetCore.v1.zip from Hugging Face:
  https://huggingface.co/datasets/ShapeNet/ShapeNetCore-archive/tree/main

2. Unzip the dataset into the following path:
 ./data/ShapeNetCore.v1/

3. Then, refer to the instructions in DATASET.md to convert ShapeNet models into PcBD-compatible point cloud format and label structure.
 This allows you to flexibly construct datasets tailored to your application scenarios, including boundary and outlier labels for any ShapeNet category.
