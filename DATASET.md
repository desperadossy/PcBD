#  Bound57 Dataset

![Dataset Overview](dataset.png)

## Introduction

**Bound57** is a point cloud dataset designed for a variety of 3D point cloud processing tasks, including:

- ✅ Outlier Removal  
- ✅ Boundary Detection  
- ✅ Boundary Smoothing  
- ✅ Overall Denoising  
- ✅ Point Cloud Upsampling  
- ✅ Point Cloud Completion

It is suitable for research in wind tunnel experiments, depth sensor modeling, and general 3D perception tasks.

---

##  Included Data
Generated dataset available at: https://ssy-pcdatasets.oss-cn-hangzhou.aliyuncs.com/Bound57  
The downloaded Bound57 dataset contains the following files:

- `input.pcd`: Noisy point cloud, typically from a simulated Time-of-Flight (ToF) camera  
- `gt.pcd`: Ground truth clean boundary, generated from orthographic projection  
- `label.npy`: Per-point labels including **outlier** and **projection boundary** annotations

> Currently, only labels for outliers, projected boundaries, and smoothed boundaries are included.

---

##  Generate Custom Datasets

To support tasks such as completion, denoising, and upsampling, you can generate your own dataset using ShapeNetCore.v1 models.

### Step 1: Download ShapeNetCore.v1

Please follow instructions in the [`README.md`](README.md) to download and extract:

- [https://huggingface.co/datasets/ShapeNet/ShapeNetCore-archive/tree/main](https://huggingface.co/datasets/ShapeNet/ShapeNetCore-archive/tree/main)

The expected structure after extraction:
