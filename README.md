# Flow Matching for Image Generation

A streamlined flow matching codebase focused specifically on image generation. This repository contains complete implementations for training flow matching models on image datasets using both continuous and discrete flows.

## Features

- ✅ **Continuous Flow Matching**: Standard flow matching for RGB images
- ✅ **Discrete Flow Matching**: Flow matching for categorical/discrete image representations  
- ✅ **Multiple Datasets**: Support for ImageNet and CIFAR-10
- ✅ **Advanced Training**: Distributed training, EMA, classifier-free guidance
- ✅ **Pre-built Models**: UNet and Discrete UNet architectures
- ✅ **Complete Pipeline**: Training scripts, evaluation, and inference notebook

## Quick Start

### 1. Setup Environment

```bash
# Create conda environment
conda env create -f environment.yml
conda activate flow_matching

# Install package
pip install -e .

# Install additional requirements
pip install -r requirements.txt
```

### 2. Prepare Data

Download and prepare ImageNet data:

```bash
# Set data directory
export IMAGENET_DIR=~/flow_matching/data/
export IMAGENET_RES=64

# Download blurred ImageNet from official website
tar -xf ~/Downloads/train_blurred.tar.gz -C $IMAGENET_DIR

# Downsample to desired resolution  
git clone git@github.com:PatrykChrabaszcz/Imagenet32_Scripts.git
python Imagenet32_Scripts/image_resizer_imagent.py -i ${IMAGENET_DIR}train_blurred -o ${IMAGENET_DIR}train_blurred_$IMAGENET_RES -s $IMAGENET_RES -a box -r -j 10
```

### 3. Training

**Test run locally:**
```bash
python train.py --data_path=${IMAGENET_DIR}train_blurred_$IMAGENET_RES/box/ --test_run
```

**Full training on SLURM cluster:**
```bash
python submitit_train.py --data_path=${IMAGENET_DIR}train_blurred_$IMAGENET_RES/box/
```

**Discrete flow matching:**
```bash
python submitit_train.py --dataset=cifar10 --discrete_flow_matching --batch_size=32 --epochs=3000
```

### 4. Evaluation & Inference

```bash
# Evaluate trained model
python submitit_train.py --data_path=${IMAGENET_DIR}train_blurred_$IMAGENET_RES/box/ --resume=./output_dir/checkpoint-899.pth --compute_fid --eval_only

# Use Jupyter notebook for interactive inference
jupyter notebook load_model_checkpoint.ipynb
```

## Repository Structure

```
.
├── flow_matching/              # Core flow matching library
│   ├── path/                   # Probability paths and schedulers
│   ├── solver/                 # ODE and discrete solvers
│   └── utils/                  # Utilities and model wrappers
├── models/                     # Model architectures
│   ├── unet.py                 # UNet for continuous flows
│   ├── discrete_unet.py        # UNet for discrete flows
│   └── model_configs.py        # Model configurations
├── training/                   # Training utilities
│   ├── train_loop.py           # Training loop implementation
│   ├── eval_loop.py            # Evaluation loop
│   └── ...                     # Data transforms, distributed mode, etc.
├── train.py                    # Main training script
├── submitit_train.py           # SLURM submission script
├── train_arg_parser.py         # Command line arguments
└── load_model_checkpoint.ipynb # Inference notebook
```

## Results

| Dataset | Model | Epochs | FID | Command |
|---------|-------|--------|-----|---------|
| ImageNet64 (Blurred) | Class conditional UNet | 900 | 1.64 | `python submitit_train.py --data_path=${IMAGENET_DIR}train_blurred_64/box/ --batch_size=32 --nodes=8` |
| CIFAR10 (Discrete) | Unconditional UNet | 2500 | 3.58 | `python submitit_train.py --dataset=cifar10 --discrete_flow_matching --batch_size=32 --epochs=3000` |

## Core Components

### Flow Matching Paths
- `CondOTProbPath`: Conditional Optimal Transport probability paths
- `MixtureDiscreteProbPath`: Discrete probability paths for categorical data

### Solvers  
- `ODESolver`: Continuous ODE solver for flow sampling
- `MixtureDiscreteEulerSolver`: Discrete Euler solver for categorical flows

### Model Architectures
- `UNetModel`: Standard UNet for continuous flows
- `DiscreteUNetModel`: UNet adapted for discrete/categorical data

## Requirements

- Python 3.9+
- PyTorch 2.0+
- torchvision
- torchdiffeq
- submitit (for cluster training)
- torchmetrics[image] (for evaluation)

## License

This project is licensed under the CC-by-NC license. See the LICENSE file for details.
