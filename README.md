
# point-transformer-v3-installation
“Installation guide for Point Transformer v3 on Ubuntu 24.04 with CUDA 12.8 and PyTorch 2.8.0”
=======
# Point Transformer v3 Installation Guide

This repository provides a step-by-step guide to install the dependencies for the Point Transformer v3 model on **Ubuntu 24.04** with **CUDA 12.8** and **PyTorch 2.8.0 (nightly)**. The instructions are tailored for NVIDIA GPUs, such as the RTX 5070 Ti with compute capability 12.0, and have been tested to ensure compatibility.

## Prerequisites

- **Operating System**: Ubuntu 24.04
- **GPU**: NVIDIA GPU with CUDA support (e.g., RTX 5070 Ti, compute capability 12.0)
- **CUDA Toolkit**: 12.8
- **NVIDIA Driver**: Version 570.144 or later
- **Conda**: Anaconda or Miniconda installed
- **Internet Access**: Required for downloading packages

Ensure the CUDA Toolkit 12.8 and NVIDIA drivers are installed before proceeding. You can verify this with:

```bash
nvcc --version
nvidia-smi
```

## Installation

Follow the detailed instructions in INSTALL.md to set up the environment and install all dependencies, including PyTorch, PyTorch Geometric modules, `spconv`, `pointops`, and `open3d`.

## Verification

After installation, verify that all modules work with CUDA:

```bash
python -c "import torch; import torch_scatter; import torch_cluster; import torch_sparse; import torch_geometric; import spconv; import pointops; import SharedArray; import tensorboard; import open3d; print('All modules imported successfully')"
```

## Usage

Once installed, navigate to your Point Transformer v3 repository (e.g., `~/Models/Pointcept`) and follow its documentation to prepare datasets and run training or inference scripts.


