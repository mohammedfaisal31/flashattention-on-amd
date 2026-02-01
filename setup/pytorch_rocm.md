# PyTorch with ROCm Support

To use PyTorch on AMD GPUs, you must install the specific ROCm-enabled wheels.

## Direct Installation
Replace `rocm6.0` with your installed ROCm version if different.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```

## Verification Script
Run the following Python snippet to check if PyTorch detects your AMD GPU:

```python
import torch
print(f"Is ROCm available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
```

## Environment Variables
In some cases, you may need to force the architecture if it's not detected correctly:
```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # For RX 6800 (gfx1030)
```

## Common Issues
- **MIOpen errors**: Ensure the MIOpen cache is properly initialized.
- **Symbol lookup errors**: Usually caused by a mismatch between the ROCm driver version and the PyTorch wheel version.
