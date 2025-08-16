# Diffusion SuctionNet Model

[![PyPI version](https://badge.fury.io/py/diffusion-suctionnet-model.svg)](https://badge.fury.io/py/diffusion-suctionnet-model)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A PyTorch implementation of Diffusion SuctionNet for 6DoF suction grasping using diffusion models.

## Features

- 🚀 **Diffusion-based Suction Grasping**: Advanced 6DoF pose estimation for suction grasping
- 🎯 **PointNet2 Backbone**: Efficient point cloud feature extraction
- 🔧 **DDIM Scheduler**: Denoising Diffusion Implicit Models for fast sampling
- 📊 **Multi-task Learning**: Simultaneous prediction of suction scores and object properties
- 💫 **Attention Mechanisms**: Channel and spatial attention for improved performance

## Installation

### From PyPI (Recommended)

```bash
pip install diffusion-suctionnet-model
```

### From Source

```bash
git clone https://github.com/btsd321/diffusion_suctionnet_model.git
cd diffusion_suctionnet_model
pip install -e .
```

### Dependencies

- PyTorch >= 1.7.0
- torchvision >= 0.8.0
- numpy >= 1.19.0
- scipy >= 1.5.0
- tqdm >= 4.50.0
- h5py >= 2.10.0

## Quick Start

```python
import torch
from diffusion_suctionnet_model import dsnet

# Create model
model = dsnet(use_vis_branch=True, return_loss=False)
model.eval()

# Prepare input data
batch_size = 1
num_points = 16384

inputs = {
    'point_clouds': torch.randn(batch_size, num_points, 3),
    'labels': {
        'suction_or': torch.randn(batch_size, num_points, 3),
        'suction_seal_scores': torch.randn(batch_size, num_points),
        'suction_wrench_scores': torch.randn(batch_size, num_points),
        'suction_feasibility_scores': torch.randn(batch_size, num_points),
        'individual_object_size_lable': torch.randn(batch_size, num_points),
    }
}

# Forward pass
with torch.no_grad():
    pred_results, ddim_loss = model(inputs)
    print(f"Output shape: {pred_results.shape}")
```

## Advanced Usage

### Custom Scheduling

```python
from diffusion_suctionnet_model import DDIMScheduler, ScheduledCNNRefine

# Create custom scheduler
scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02,
    beta_schedule="linear"
)

# Create refine model
refine_model = ScheduledCNNRefine(
    channels_in=128,
    channels_noise=4
)
```

### Training Mode

```python
# Enable training mode
model = dsnet(use_vis_branch=True, return_loss=True)
model.train()

# Forward pass returns loss for training
pred_results, ddim_loss = model(inputs)
loss = ddim_loss[0] + ddim_loss[1]  # Combined loss
```

## Model Architecture

The model consists of several key components:

1. **PointNet2 Backbone**: Extracts hierarchical features from point clouds
2. **Diffusion Module**: Uses DDIM for denoising prediction
3. **Attention Mechanisms**: Channel and spatial attention for feature enhancement
4. **Multi-task Heads**: Predicts various suction-related scores

## API Reference

### Main Classes

- `dsnet`: Main network class
- `ScheduledCNNRefine`: Diffusion refinement module
- `CNNDDIMPipiline`: DDIM sampling pipeline
- `DDIMScheduler`: Denoising diffusion scheduler

### Utility Functions

- `load_checkpoint()`: Load model from checkpoint
- `save_checkpoint()`: Save model checkpoint
- `save_pth()`: Save model as .pth file

## Development

### Building from Source

```bash
# Clone repository
git clone https://github.com/btsd321/diffusion_suctionnet_model.git
cd diffusion_suctionnet_model

# Install in development mode
pip install -e .

# Run tests
python test_package.py
```

### Building Distribution

```bash
# Run build script
./build.sh

# Or manually
python setup.py sdist bdist_wheel
```

## Citation

If you use this work in your research, please cite:

```bibtex
@article{huang2025diffusion,
  title={Diffusion Suction Grasping with Large-Scale Parcel Dataset},
  author={Huang, Ding-Tao and He, Xinyi and Hua, Debei and Yu, Dongfang and Lin, En-Te and Zeng, Long},
  journal={arXiv preprint arXiv:2502.07238},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

If you encounter any issues or have questions, please:

1. Check the [documentation](https://github.com/btsd321/diffusion_suctionnet_model/blob/main/README.md)
2. Search existing [issues](https://github.com/btsd321/diffusion_suctionnet_model/issues)
3. Create a new issue if needed
