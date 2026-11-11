# System requirements

The formal revision campaign used Python 3.10.20, PyTorch 2.6.0+cu124, and
NVIDIA RTX 4090 GPUs. A current NVIDIA driver compatible with CUDA 12.4 is
required for GPU training and inference. The PyTorch wheel supplies the CUDA
runtime; a system CUDA toolkit is not required for ordinary inference.

Additional command-line tools are needed only for specific workflows:

- MMseqs2: rebuild or audit the 60%-identity clustered splits.
- HMMER 3.4 (`hmmscan`): ANARCI/NanoBodyBuilder2 structure controls.
- `zstd`: create or extract the `.tar.zst` release archive.

The main model, NbBench probes, and structure predictors use separate environments.
Do not install all optional baseline packages into the main training environment.
