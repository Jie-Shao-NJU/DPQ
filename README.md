# Memory-Efficient Generative Models via Product Quantization

Official implementation of **"Memory-Efficient Generative Models via Product Quantization"** (ICCV 2025 Submission).

## Overview

This repository contains the implementation of an extreme compression method for diffusion models using **Product Quantization (PQ)**. Our approach enables compressing generative models to as low as **1-bit precision**, achieving up to **13× compression ratio** for class-to-image models and **8× compression** for text-to-image models while maintaining high generation quality.

### Key Features

- **Extreme Compression**: Compress diffusion models to 1-2 bits with minimal quality loss
- **High Compression Ratio**: Up to 13× reduction in model size compared to FP16 baseline
- **Product Quantization**: Extended PQ method with exponentially increased codebook capacity
- **Codebook Compression**: Novel approach to compress codebooks based on usage frequency
- **EM Calibration**: EM-based calibration with re-initialization for optimal performance
- **Broad Applicability**: Works with DiT, LDM, LLMs, and 3D Gaussian Splatting

### Performance Highlights

| Model | Bit-width | Method | Size (MB) | FID ↓ | IS ↑ |
|-------|-----------|--------|-----------|-------|------|
| DiT-XL/2 (250 steps) | 16 | FP16 | 1349.4 | 5.33 | 275.13 |
| | 2 | GPTQ | 172.8 | 280.11 | 3.97 |
| | 2 | Q-DiT | 168.3 | 313.63 | 1.28 |
| | 2 | **Ours** | **211.3** | **6.84** | **190.99** |
| | 1 | VQ | 89.1 | 91.71 | 13.51 |
| | 1 | **Ours** | **103.9** | **14.03** | **110.23** |

## Method

### Why Product Quantization?

Traditional post-training quantization (PTQ) methods suffer from severe degradation at extremely low bit-widths (1-2 bits). Vector quantization (VQ), while better, still struggles with diffusion models due to:
- **Low capacity**: Fixed-length shared codebook limits representation accuracy
- **Error accumulation**: Iterative nature of diffusion amplifies cumulative errors

### Our Approach

We extend product quantization for compressing diffusion models through:

1. **Product Quantization**: Decompose vector space into multiple low-dimensional subspaces, each with independent codebooks, achieving exponential capacity growth (k^N codewords with O(kN) storage)

2. **Codebook Compression**: Compress codebooks based on centroid usage frequency and similarity, keeping size within 1/4 of assignment storage

3. **EM Calibration**:
   - **E-step**: Update assignments to minimize output reconstruction error
   - **M-step**: Optimize codebooks via gradient descent using DDPM loss

4. **Codebook Re-initialization**: Prevent codebook collapse by maintaining a reservoir of past vectors and reinitializing dead centroids

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/DiT-pq.git
cd DiT-pq

# Create conda environment
conda create -n dit-pq python=3.10 -y
conda activate dit-pq

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Compress a Pre-trained DiT Model

```python
from pq.pq_getmodel import compress_model

# Load and compress DiT-XL/2 to 2-bits
compressed_model = compress_model(
    model_name='DiT-XL/2',
    bit_width=2,  # 1, 2, 3, or 4 bits
    image_size=256
)
```

### Sample from Compressed Model

```bash
# Sample from 2-bit compressed DiT-XL/2
python sample.py --model DiT-XL/2 --bits 2 --num-samples 10

# Sample from 1-bit compressed model
python sample.py --model DiT-XL/2 --bits 1 --num-samples 10 --cfg-scale 1.5
```

### Calibration (Optional)

For better performance, calibrate the compressed model:

```bash
# Calibrate compressed model (requires ImageNet training data)
python train.py --model DiT-XL/2 --bits 2 --data-path /path/to/imagenet --epochs 1
```

## Project Structure

```
DiT-pq/
├── pq/                      # Product quantization implementation
│   ├── pq_getmodel.py      # Model compression
│   ├── utils_model.py      # Model utilities
│   ├── utils_traineval.py  # Training and evaluation
│   └── tools/              # Helper tools
├── pqf/                     # PQ framework
│   ├── compressed_layers/   # Compressed layer implementations
│   └── permutation/         # Permutation optimization
├── diffusion/               # Diffusion model utilities
├── models.py                # DiT model definitions
├── train.py                 # Training script
├── sample.py                # Sampling script
├── sample_ddp.py            # Distributed sampling
└── evaluator.py             # Evaluation metrics
```

## Experiments

### Class-to-Image Generation (ImageNet 256×256)

Compression results on DiT-XL/2:

| Bit-width | FID ↓ | sFID ↓ | IS ↑ | Precision ↑ | Recall ↑ |
|-----------|-------|--------|------|-------------|----------|
| 16 (FP16) | 5.41 | 17.85 | 269.84 | 0.8197 | 0.5712 |
| 4 | 6.25 | - | 215.12 | 0.7576 | 0.6842 |
| 3 | 7.01 | - | 188.52 | 0.7403 | 0.6868 |
| 2 | 6.84 | 18.75 | 190.99 | 0.7685 | 0.6588 |
| 1 | 14.03 | - | 110.23 | 0.6863 | 0.6652 |

### Text-to-Image Generation

Results on PixArt-α and FLUX.1-dev at 1024×1024:

| Model | Method | Size (MB) | GenEval Overall | T2I-Compbench Overall |
|-------|--------|-----------|-----------------|----------------------|
| PixArt-α (FP16) | - | ~2400 | 0.48 | 0.65 |
| PixArt-α | SVDQuant (2-bit) | ~300 | 0.17 | 0.24 |
| PixArt-α | **Ours (2-bit)** | **~300** | **0.46** | **0.64** |
| FLUX.1-dev (FP16) | - | ~24000 | 0.66 | 0.63 |
| FLUX.1-dev | SVDQuant (2-bit) | ~3000 | 0.32 | 0.33 |
| FLUX.1-dev | **Ours (2-bit)** | **~3000** | **0.57** | **0.61** |

### Evaluation

```bash
# Generate 50K samples for FID calculation
torchrun --nproc_per_node=8 sample_ddp.py \
    --model DiT-XL/2 \
    --bits 2 \
    --num-fid-samples 50000

# Evaluate with ADM's evaluation suite
python evaluator.py --ref-batch path/to/imagenet --sample-batch path/to/samples
```

## Comparison with Other Methods

| Method | Compression Ratio | Device Compatibility | Training Support | Deployment |
|--------|------------------|---------------------|-----------------|------------|
| Pruning | ~2× | ✓ All devices | ✓ Yes | ✓ Yes |
| Low-rank | ~2× | ✓ All devices | ✓ Yes | ✓ Yes |
| Quantization (PTQ) | ~4× | ⚠ Specific (H100) | ✗ No | ✓ Yes |
| **Ours (PQ)** | **~13×** | **✓ All devices** | **✓ Yes** | **✓ Yes** |

## Implementation Details

### Hyperparameters

- **Subvector dimension (d)**: 2 (4-bit), 3 (3-bit), 4 (2-bit), 8 (1-bit)
- **Codebook size (k)**: 256 (INT8)
- **Similarity threshold (τc)**: 0.05
- **Dead threshold (τdead)**: 256
- **Learning rate**: 1e-4 (AdamW)
- **Calibration epochs**: 1 (default)

### Compression Stages

**Stage 1: Product Quantization**
- Partition weights into subvectors
- Initialize codebooks using K-means (20 iterations)
- Assign each subvector to nearest centroid
- Compress codebooks based on usage and similarity

**Stage 2: EM Calibration**
- E-step: Update assignments using activation-aware reconstruction
- M-step: Optimize codebooks via DDPM loss backpropagation
- Monitor and reinitialize dead centroids

## Applications

Our method has been validated across multiple domains:

- **Diffusion Models**: DiT, LDM (both transformer and U-Net architectures)
- **Large Language Models**: LLAMA-2 7B with Quip#
- **3D Gaussian Splatting**: Compact3D

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{dit-pq2025,
  title={Memory-Efficient Generative Models via Product Quantization},
  author={Anonymous},
  booktitle={ICCV},
  year={2025}
}
```

## License

This project is licensed under CC-BY-NC. See LICENSE for details.

## Acknowledgments

This codebase builds upon:
- [DiT](https://github.com/facebookresearch/DiT) - Diffusion Transformers
- [Stable Diffusion](https://github.com/Stability-AI/stablediffusion) - Latent Diffusion Models
- [ADM](https://github.com/openai/guided-diffusion) - Evaluation metrics

## Contact

For questions or issues, please open an issue on GitHub.
