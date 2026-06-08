# Lab 2: GPU Video Processing

Variant 5 implements color quantization for video frames. The script contains four GPU kernel variants:

- CUDA-A: 2D grid blocks and 2D threads.
- CUDA-B: 3D grid blocks and 2D threads.
- OpenCL-A: 1D work groups and 1D work items.
- OpenCL-B: 2D work groups and 2D work items.

Install dependencies:

```bash
python -m pip install -r requirements-gpu.txt
```

Run benchmark:

```bash
python run_lab2_gpu.py
```

The script generates synthetic input videos and writes benchmark artifacts to `lab2_results/`. Generated media and results are ignored by git.
