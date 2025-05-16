# Installation Guide for Point Transformer v3

This guide details how to install the dependencies for the Point Transformer v3 model on Ubuntu 24.04 with CUDA 12.8 and PyTorch 2.8.0 (nightly build). The setup is tailored for NVIDIA GPUs with compute capability 12.0 (e.g., RTX 5070 Ti).

## Prerequisites

- **Ubuntu**: 24.04
- **CUDA Toolkit**: 12.8
- **NVIDIA Driver**: 570.144 or later
- **GPU**: NVIDIA GPU with compute capability 12.0
- **Conda**: Anaconda or Miniconda
- **Python**: 3.9

Verify CUDA and driver installation:

```bash
nvcc --version
nvidia-smi
```

## Installation Steps

### 1. Create and Activate Conda Environment

```bash
conda create -n ptv3 python=3.9 -y
conda activate ptv3
```

### 2. Install GCC and G++ (Version 14)

CUDA 12.8 requires GCC 14 or earlier for compiling extensions:

```bash
conda install -c conda-forge gcc_linux-64=14 gxx_linux-64=14 -y
```

Verify GCC version:

```bash
~/anaconda3/envs/ptv3/bin/x86_64-conda-linux-gnu-gcc --version
```

Expected output: `gcc (conda-forge linux-64) 14.2.0`.

### 3. Install PyTorch Nightly Build

Install PyTorch with CUDA 12.8 support:

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall
```

Verify PyTorch installation:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Expected output: `2.8.0.dev20250516+cu128` (or similar) and `True`.

### 4. Set CUDA Architecture

Set the CUDA architecture for your GPU (compute capability 12.0 for RTX 5070 Ti):

```bash
export TORCH_CUDA_ARCH_LIST="12.0"
mkdir -p ~/anaconda3/envs/ptv3/etc/conda/activate.d
echo 'export TORCH_CUDA_ARCH_LIST="12.0"' >> ~/anaconda3/envs/ptv3/etc/conda/activate.d/env_vars.sh
```

Verify compute capability:

```bash
python -c "import torch; print(torch.cuda.get_device_capability())"
```

Expected output: `(12, 0)`.

### 5. Install PyTorch Geometric Modules

Install from source to ensure compatibility with PyTorch nightly:

```bash
pip install git+https://github.com/rusty1s/pytorch_scatter.git
pip install git+https://github.com/rusty1s/pytorch_cluster.git
pip install git+https://github.com/rusty1s/pytorch_sparse.git
pip install torch-geometric
```

### 6. Install Additional Dependencies

Install remaining dependencies:

```bash
pip install h5py pyyaml sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm
```

Note: Import `sharedarray` as `SharedArray` (case-sensitive) in your code.

### 7. Verify Core Dependencies

Test that core modules work:

```bash
python -c "import torch; import torch_scatter; import torch_cluster; import torch_sparse; import torch_geometric; import SharedArray; import tensorboard; print(torch.__version__); print(torch.cuda.is_available())"
```

Expected output: PyTorch version and `True`.

### 8. Install `spconv` for SparseUNet

The original instructions specify `spconv-cu113`, but for CUDA 12.8, build from source:

```bash
git clone https://github.com/traveller59/spconv.git
cd spconv
export TORCH_CUDA_ARCH_LIST="12.0"
python setup.py bdist_wheel
pip install dist/*.whl
cd ..
```

Alternatively, check for a pre-built wheel:

```bash
pip install spconv-cu118
```

### 9. Build `pointops` Extension

Build the `pointops` extension from the Pointcept repository:

```bash
cd ~/Models/PTV3/Pointcept-main/libs/pointops
export TORCH_CUDA_ARCH_LIST="12.0"
python setup.py install
cd ../../..
```

Replace `~/Models/PTV3/Pointcept-main` with your actual repository path.

### 10. Install `open3d` (Optional)

For visualization:

```bash
pip install open3d
```

### 11. Final Verification

Confirm all modules work:

```bash
python -c "import torch; import torch_scatter; import torch_cluster; import torch_sparse; import torch_geometric; import spconv; import pointops; import SharedArray; import tensorboard; import open3d; print('All modules imported successfully')"
```

Expected output: `All modules imported successfully`.

## Troubleshooting

- **CUDA Errors**: Ensure CUDA 12.8 and NVIDIA drivers are correctly installed (`nvcc --version`, `nvidia-smi`).
- **Build Failures**: Verify `TORCH_CUDA_ARCH_LIST="12.0"` is set and GCC is version 14.
- **Module Import Issues**: Check installation with `pip list | grep <module>`.
- **Pointcept Issues**: Refer to the Pointcept repository for model-specific issues.

## Next Steps

Navigate to your Point Transformer v3 repository (e.g., `~/Models/Pointcept`), prepare the dataset, and run training or inference scripts as per the project’s documentation.