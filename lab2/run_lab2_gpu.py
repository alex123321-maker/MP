"""Run Lab 2 GPU benchmark for variant 5.

The script implements the four required kernels:
- CUDA-A: 2-D grid blocks and 2-D threads, one frame per launch
- CUDA-B: 3-D grid blocks and 2-D threads, whole video per launch
- OpenCL-A: 1-D work groups and 1-D work items, whole video per launch
- OpenCL-B: 2-D work groups and 2-D work items, one frame per launch
"""

from __future__ import annotations

import gc
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import cv2
import cupy as cp
import matplotlib
import numpy as np
import pandas as pd
import pyopencl as cl

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT_DIR = Path("lab2_results")
VIDEO_DIR = OUT_DIR / "videos"
N_QUANTS = 6
RUNS = 3
BLOCK_2D = [(8, 8), (16, 16), (32, 32)]
BLOCK_1D = [64, 128, 256, 512]
VIDEOS = [
    ("v1_small.mp4", 320, 240, 15),
    ("v2_med.mp4", 480, 360, 30),
    ("v3_large.mp4", 640, 480, 60),
]


CUDA_SRC = r"""
extern "C" __global__
void quantize_2d2d(unsigned char *img, int W, int H, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float bw = 256.0f / (float)N;
    int idx = (y * W + x) * 3;
    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        int b = (int)(img[idx+c] / bw);
        if (b >= N) b = N - 1;
        int v = (int)(b * bw + bw * 0.5f);
        if (v > 255) v = 255;
        img[idx+c] = (unsigned char)v;
    }
}

extern "C" __global__
void quantize_3d2d(unsigned char *vid, int W, int H, int Nf, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int f = blockIdx.z;
    if (x >= W || y >= H || f >= Nf) return;
    float bw = 256.0f / (float)N;
    int idx = ((f * H + y) * W + x) * 3;
    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        int b = (int)(vid[idx+c] / bw);
        if (b >= N) b = N - 1;
        int v = (int)(b * bw + bw * 0.5f);
        if (v > 255) v = 255;
        vid[idx+c] = (unsigned char)v;
    }
}
"""


OCL_SRC = r"""
__kernel void quantize_1d(__global uchar *img, int total_pixels, int N) {
    int gid = get_global_id(0);
    if (gid >= total_pixels) return;
    float bw = 256.0f / (float)N;
    int idx = gid * 3;
    for (int c = 0; c < 3; ++c) {
        int b = (int)(img[idx+c] / bw);
        if (b >= N) b = N - 1;
        int v = (int)(b * bw + bw * 0.5f);
        if (v > 255) v = 255;
        img[idx+c] = (uchar)v;
    }
}

__kernel void quantize_2d(__global uchar *img, int W, int H, int N) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= W || y >= H) return;
    float bw = 256.0f / (float)N;
    int idx = (y * W + x) * 3;
    for (int c = 0; c < 3; ++c) {
        int b = (int)(img[idx+c] / bw);
        if (b >= N) b = N - 1;
        int v = (int)(b * bw + bw * 0.5f);
        if (v > 255) v = 255;
        img[idx+c] = (uchar)v;
    }
}
"""


def shell(cmd: list[str]) -> str:
    return subprocess.getoutput(" ".join(cmd))


def make_video(path: Path, w: int, h: int, duration_s: int, fps: int = 15) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    n_frames = duration_s * fps
    n_scenes = 3
    frames_per_scene = n_frames // n_scenes
    rng = np.random.default_rng(42)
    palettes = [
        tuple(int(x) for x in rng.integers(40, 200, size=3))
        for _ in range(n_scenes)
    ]
    gx = np.linspace(0, 60, w, dtype=np.uint8)
    gy = np.linspace(0, 60, h, dtype=np.uint8)

    for i in range(n_frames):
        scene = min(i // frames_per_scene, n_scenes - 1)
        base = np.full((h, w, 3), palettes[scene], dtype=np.uint8)
        base[:, :, 0] = np.clip(base[:, :, 0].astype(int) + gx[None, :], 0, 255)
        base[:, :, 1] = np.clip(base[:, :, 1].astype(int) + gy[:, None], 0, 255)
        cx = int((i / n_frames) * w)
        cy = h // 2 + int(40 * np.sin(i / 6))
        cv2.circle(base, (cx, cy), min(w, h) // 8, (255, 255, 255), -1)
        cv2.putText(
            base,
            f"S{scene + 1} f{i}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )
        writer.write(base)
    writer.release()


def load_frames(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"Cannot read frames from {path}")
    return np.stack(frames).astype(np.uint8)


def save_video(path: Path, frames: np.ndarray, fps: int = 15) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _, h, w, _ = frames.shape
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


def round_up(n: int, mul: int) -> int:
    return ((n + mul - 1) // mul) * mul


class CudaRunner:
    def __init__(self) -> None:
        self.k_2d2d = cp.RawKernel(CUDA_SRC, "quantize_2d2d")
        self.k_3d2d = cp.RawKernel(CUDA_SRC, "quantize_3d2d")
        cp.cuda.Device().synchronize()

    def run_2d2d(self, frames: np.ndarray, n_quants: int, block: tuple[int, int]):
        n_frames, h, w, _ = frames.shape
        out = frames.copy()
        bx, by = block
        grid = ((w + bx - 1) // bx, (h + by - 1) // by, 1)
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        for i in range(n_frames):
            dev = cp.asarray(out[i])
            self.k_2d2d(grid, (bx, by, 1), (dev, np.int32(w), np.int32(h), np.int32(n_quants)))
            out[i] = cp.asnumpy(dev)
        end.record()
        end.synchronize()
        return out, cp.cuda.get_elapsed_time(start, end) / 1000.0

    def run_3d2d(self, frames: np.ndarray, n_quants: int, block: tuple[int, int]):
        n_frames, h, w, _ = frames.shape
        bx, by = block
        grid = ((w + bx - 1) // bx, (h + by - 1) // by, n_frames)
        dev = cp.asarray(frames.copy())
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        self.k_3d2d(grid, (bx, by, 1), (dev, np.int32(w), np.int32(h), np.int32(n_frames), np.int32(n_quants)))
        out = cp.asnumpy(dev)
        end.record()
        end.synchronize()
        return out, cp.cuda.get_elapsed_time(start, end) / 1000.0


class OpenCLRunner:
    def __init__(self) -> None:
        self.ctx, self.dev = self.pick_ctx()
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
        self.prog = cl.Program(self.ctx, OCL_SRC).build()

    @staticmethod
    def pick_ctx():
        for platform_ in cl.get_platforms():
            gpus = platform_.get_devices(cl.device_type.GPU)
            if gpus:
                return cl.Context([gpus[0]]), gpus[0]
        for platform_ in cl.get_platforms():
            cpus = platform_.get_devices(cl.device_type.CPU)
            if cpus:
                return cl.Context([cpus[0]]), cpus[0]
        raise RuntimeError("No OpenCL devices")

    def run_1d(self, frames: np.ndarray, n_quants: int, lws: int):
        n_frames, h, w, _ = frames.shape
        out = frames.copy()
        total = n_frames * h * w
        buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=out)
        evt = self.prog.quantize_1d(
            self.queue,
            (round_up(total, lws),),
            (lws,),
            buf,
            np.int32(total),
            np.int32(n_quants),
        )
        evt.wait()
        cl.enqueue_copy(self.queue, out, buf).wait()
        return out, (evt.profile.end - evt.profile.start) * 1e-9

    def run_2d(self, frames: np.ndarray, n_quants: int, lws: tuple[int, int]):
        n_frames, h, w, _ = frames.shape
        out = frames.copy()
        lx, ly = lws
        gws = (round_up(w, lx), round_up(h, ly))
        buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=h * w * 3)
        total_t = 0.0
        for i in range(n_frames):
            cl.enqueue_copy(self.queue, buf, out[i]).wait()
            evt = self.prog.quantize_2d(
                self.queue,
                gws,
                (lx, ly),
                buf,
                np.int32(w),
                np.int32(h),
                np.int32(n_quants),
            )
            evt.wait()
            cl.enqueue_copy(self.queue, out[i], buf).wait()
            total_t += (evt.profile.end - evt.profile.start) * 1e-9
        return out, total_t


def save_example_image(original: np.ndarray, quantized: np.ndarray) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(cv2.cvtColor(original[0], cv2.COLOR_BGR2RGB))
    ax[0].set_title("Original")
    ax[0].axis("off")
    ax[1].imshow(cv2.cvtColor(quantized[0], cv2.COLOR_BGR2RGB))
    ax[1].set_title(f"Quantized n={N_QUANTS}")
    ax[1].axis("off")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "example_frame.png", dpi=150)
    plt.close(fig)


def save_plots(df: pd.DataFrame) -> None:
    videos = list(df["video"].unique())
    fig, axes = plt.subplots(1, len(videos), figsize=(5 * len(videos), 4), sharey=False)
    if len(videos) == 1:
        axes = [axes]
    for ax, video in zip(axes, videos):
        sub = df[df["video"] == video]
        for kernel in sub["kernel"].unique():
            series = sub[sub["kernel"] == kernel].sort_values("threads_per_block")
            ax.plot(series["threads_per_block"], series["avg_s"] * 1000, marker="o", label=kernel)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Threads per block / work-items per group")
        ax.set_ylabel("Average time, ms")
        ax.set_title(video)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "benchmark.png", dpi=150)
    plt.close(fig)


def collect_hardware(opencl_dev: cl.Device) -> dict:
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": shell(["lscpu", "|", "grep", "-E", "'Model name|Socket|Thread|Core|CPU\\(s\\)'"]),
        "ram_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 2),
        "gpu": shell(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv,noheader"]),
        "cuda_runtime": str(cp.cuda.runtime.runtimeGetVersion()),
        "opencl_device": f"{opencl_dev.name} | {opencl_dev.platform.name}",
    }


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    VIDEO_DIR.mkdir(exist_ok=True)

    print("Generating test videos")
    for name, w, h, duration in VIDEOS:
        path = VIDEO_DIR / name
        if not path.exists():
            make_video(path, w, h, duration)
        print(f"  {path}: {w}x{h}, {duration}s, {path.stat().st_size / 1e6:.1f} MB")

    cuda_runner = CudaRunner()
    ocl_runner = OpenCLRunner()
    hardware = collect_hardware(ocl_runner.dev)
    (OUT_DIR / "hardware.json").write_text(json.dumps(hardware, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(hardware, ensure_ascii=False, indent=2))

    frames0 = load_frames(VIDEO_DIR / VIDEOS[0][0])
    c1, t1 = cuda_runner.run_2d2d(frames0, N_QUANTS, (16, 16))
    c2, t2 = cuda_runner.run_3d2d(frames0, N_QUANTS, (16, 16))
    o1, t3 = ocl_runner.run_1d(frames0, N_QUANTS, 256)
    o2, t4 = ocl_runner.run_2d(frames0, N_QUANTS, (16, 16))
    sanity = {
        "cuda_a_ms": t1 * 1000,
        "cuda_b_ms": t2 * 1000,
        "opencl_a_ms": t3 * 1000,
        "opencl_b_ms": t4 * 1000,
        "identical": bool(np.array_equal(c1, c2) and np.array_equal(c1, o1) and np.array_equal(c1, o2)),
    }
    if not sanity["identical"]:
        raise RuntimeError(f"Kernel outputs differ: {sanity}")
    (OUT_DIR / "sanity.json").write_text(json.dumps(sanity, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Sanity:", sanity)
    save_video(OUT_DIR / "v1_quantized_n6.mp4", c1)
    save_example_image(frames0, c1)
    del frames0, c1, c2, o1, o2
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

    rows = []
    for name, w, h, duration in VIDEOS:
        path = VIDEO_DIR / name
        print(f"\n=== {name} ({w}x{h}, {duration}s) ===", flush=True)
        frames = load_frames(path)
        for block in BLOCK_2D:
            for kernel, func in [
                ("CUDA-A 2D+2D", cuda_runner.run_2d2d),
                ("CUDA-B 3D+2D", cuda_runner.run_3d2d),
                ("OpenCL-B 2D+2D", ocl_runner.run_2d),
            ]:
                runs = []
                for _ in range(RUNS):
                    _, elapsed = func(frames, N_QUANTS, block)
                    runs.append(elapsed)
                    cp.get_default_memory_pool().free_all_blocks()
                avg = sum(runs) / len(runs)
                rows.append(
                    {
                        "video": name,
                        "width": w,
                        "height": h,
                        "duration_s": duration,
                        "kernel": kernel,
                        "block": f"{block[0]}x{block[1]}",
                        "threads_per_block": block[0] * block[1],
                        "avg_s": avg,
                        "runs_s": json.dumps(runs),
                    }
                )
                print(f"  {kernel:16s} {block}: avg={avg * 1000:.2f} ms runs={[round(x * 1000, 2) for x in runs]}", flush=True)

        for lws in BLOCK_1D:
            runs = []
            for _ in range(RUNS):
                _, elapsed = ocl_runner.run_1d(frames, N_QUANTS, lws)
                runs.append(elapsed)
            avg = sum(runs) / len(runs)
            rows.append(
                {
                    "video": name,
                    "width": w,
                    "height": h,
                    "duration_s": duration,
                    "kernel": "OpenCL-A 1D+1D",
                    "block": str(lws),
                    "threads_per_block": lws,
                    "avg_s": avg,
                    "runs_s": json.dumps(runs),
                }
            )
            print(f"  OpenCL-A 1D+1D  {lws}: avg={avg * 1000:.2f} ms runs={[round(x * 1000, 2) for x in runs]}", flush=True)
        del frames
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "benchmark_results.csv", index=False)
    save_plots(df)
    print(f"\nSaved results to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
