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
