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
   **`./data/ShapeNetCore/`** 
 To generate your own dataset from ShapeNetCore.v1, please ensure the unzipped directory follows this structure:  
 data/ShapeNetCore/  
 ├── 02691156/  
 │ ├── 1a6ad7a24bb89733f412783097373bdc/  
 │ │ └── model.obj  
 │ ├── ...  
 ├── 02958343/  
 │ ├── <model_id>/   
 │ │ └── model.obj
 ...

4. Then, refer to the instructions in DATASET.md to convert the ShapeNet model to various point cloud data and true labels provided by the Bound57 model. In this way, you can flexibly build a dataset suitable for your application scenario, including point cloud reconstruction, outlier filtering, denoising, point cloud upsampling, and boundary extraction.

5. After downloading and extracting the Bound57 dataset, or manually generating your own dataset, the directory structure is as follows:  
  data/Bound57/  
├── Bound57.json # Dataset metadata (split info, etc.)  
├── train/  
│ └── 02691156/ # Category ID  
│ │ └── 1a6ad7a24bb89733.../ # Model ID  
│ │ │ ├── 00/  
│ │ │ │ ├── input.pcd  
│ │ │ │ ├── gt.pcd  
│ │ │ │ └── label.npy  
│ │ │ ├── 01/  
│ │ │ └── ...  
│ │ └── ...  
│ └── ...  
├── val/  
│ └── 02691156/  
│ │ └── xxx.../  
│ │ │ └── 00/  
│ │ │ ├── input.pcd  
│ │ │ ├── gt.pcd  
│ │ │ └── label.npy  
│ │ └── ...  
│ └── ...  
└── test/  
│ └── 02691156/  
│ │ └── xxx.../  
│ │ │ └── 00/  
│ │ │ ├── input.pcd  
│ │ │ ├── gt.pcd  
│ │ │ └── label.npy  
│ │ └── ...  
│ └── ...  

### 4. Modify configuration files

You can configure both the dataset and model settings by editing the `.yaml` files under the following directory:  
cfgs/  
├── Bound57_models/  
│ └── xxx.yaml # Network architecture and training parameters  
├── dataset_configs/  
│ ├── Bound57.yaml # Full dataset configuration  
│ └── Bound57SingleCategory.yaml# For training on one category only  

### 5. Train the model

You can start training using the provided shell script and YAML configuration:

```bash
bash ./scripts/train.sh 0 --config ./cfgs/Bound57_models/PcBD.yaml --exp_name example
```

0 refers to the GPU ID. You can change it to 1, 2, etc. depending on your hardware.

### 6. Test the model

To evaluate a trained model, use the following command:

```bash
bash ./scripts/test.sh 0 --ckpts /root/PcBD/checkpoints/PcBD.pth --config /root/PcBD/cfgs/Bound57_models/PcBD.yaml --exp_name example
```

For models with a trained checkpoint, replace both the .pth and .yaml:
```bash
--ckpts /your/path/to/ModelName.pth
--config ./cfgs/Bound57_models/ModelName.yaml
```

For models that do not require training (e.g., SOR):  
keep --ckpts unchanged (e.g., a dummy path). The system will automatically skip model loading.  
Just modify the YAML config file to select the correct method and dataset.
