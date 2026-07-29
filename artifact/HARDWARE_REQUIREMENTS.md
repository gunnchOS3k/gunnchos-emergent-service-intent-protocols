# Hardware requirements

## Minimum (CPU smoke)

- Apple Silicon or x86_64 CPU
- ≥8 GB RAM recommended
- Python 3.11+
- PyTorch CPU build

## GPU (optional)

- NVIDIA GPU with CUDA build of PyTorch
- If `torch.cuda.is_available()` is false, `make gate4-gpu` must report `BLOCKED_HARDWARE`
- Do not paste CUDA numbers from CPU hosts
