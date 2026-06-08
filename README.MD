# Parallel and High Performance Computing Labs

This repository contains source code for laboratory works.

## Structure

- `lab1` - CPU multithreaded video processing in Go.
- `lab2` - GPU video processing benchmark with CUDA and OpenCL kernels.
- `lab3` - distributed MPI web parsing for Wikipedia element pages.

Generated videos, benchmark outputs, profiling files, reports and task documents are intentionally excluded from the repository.

## Requirements

Lab 1:

- Go
- `ffmpeg`
- `ffprobe`

Lab 2:

- Python 3.11+
- NVIDIA GPU with CUDA-capable driver
- OpenCL runtime
- Python packages from `lab2/requirements-gpu.txt`

Lab 3:

- MPI runtime
- Python 3.11+
- Python packages from `lab3/requirements.txt`

See each lab directory for run commands.
