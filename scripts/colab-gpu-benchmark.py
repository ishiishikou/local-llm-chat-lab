#!/usr/bin/env python3
"""Colab T4 benchmark for the repository's verified Qwen3.5-4B GGUF setup."""

import hashlib
import subprocess
import threading
import time
from pathlib import Path

MODEL_URL = (
    "https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF/"
    "resolve/main/Qwen_Qwen3.5-4B-Q4_K_M.gguf?download=true"
)
MODEL_SHA256 = "13c16f426047e2de38cd075bdade4a7bcbc8c774384876f677740cda65f8a983"
MODEL = Path("/content/Qwen_Qwen3.5-4B-Q4_K_M.gguf")
LLAMA = Path("/content/llama.cpp")
BUILD = LLAMA / "build"
BENCH = BUILD / "bin" / "llama-bench"
CLI = BUILD / "bin" / "llama-cli"


def run(cmd, *, check=True, cwd=None):
    print("\n$", " ".join(map(str, cmd)), flush=True)
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(proc.stdout, flush=True)
    if check and proc.returncode:
        raise SystemExit(proc.returncode)
    return proc


def gpu_memory_mib():
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if proc.returncode:
        return None
    used, total = [int(x.strip()) for x in proc.stdout.strip().split(",")]
    return used, total


def run_with_peak(cmd):
    peak = 0
    stop = False

    def poll():
        nonlocal peak
        while not stop:
            memory = gpu_memory_mib()
            if memory:
                peak = max(peak, memory[0])
            time.sleep(0.2)

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    started = time.time()
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = time.time() - started
    stop = True
    thread.join(timeout=1)

    print(proc.stdout, flush=True)
    print(f"PEAK_VRAM_MIB={peak}")
    print(f"WALL_SECONDS={elapsed:.3f}")

    if proc.returncode:
        raise SystemExit(proc.returncode)
    return proc.stdout, peak, elapsed


print("=== ENVIRONMENT ===")
run(["nvidia-smi"])
run(["python3", "--version"])
run(["nvcc", "--version"], check=False)
run(["cmake", "--version"])

if not LLAMA.exists():
    run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            "b10809",
            "https://github.com/ggml-org/llama.cpp.git",
            str(LLAMA),
        ]
    )

if not BENCH.exists():
    run(
        [
            "cmake",
            "-S",
            str(LLAMA),
            "-B",
            str(BUILD),
            "-DGGML_CUDA=ON",
            "-DLLAMA_CURL=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
    )
    run(
        [
            "cmake",
            "--build",
            str(BUILD),
            "--target",
            "llama-bench",
            "llama-cli",
            "-j2",
        ]
    )

if not MODEL.exists():
    run(["wget", "-q", "-O", str(MODEL), MODEL_URL])

print("=== MODEL CHECK ===")
digest = hashlib.sha256()
with MODEL.open("rb") as file:
    for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
        digest.update(chunk)

actual_sha256 = digest.hexdigest()
print("MODEL_SIZE_BYTES=", MODEL.stat().st_size)
print("MODEL_SHA256=", actual_sha256)
if actual_sha256 != MODEL_SHA256:
    raise SystemExit("SHA256 mismatch")

print("=== CPU BENCHMARK (ngl=0) ===")
_, cpu_peak, cpu_wall = run_with_peak(
    [
        str(BENCH),
        "-m",
        str(MODEL),
        "-ngl",
        "0",
        "-p",
        "512",
        "-n",
        "64",
        "-r",
        "3",
    ]
)

print("=== GPU BENCHMARK (ngl=99 / T4) ===")
_, gpu_peak, gpu_wall = run_with_peak(
    [
        str(BENCH),
        "-m",
        str(MODEL),
        "-ngl",
        "99",
        "-p",
        "512",
        "-n",
        "64",
        "-r",
        "3",
    ]
)

print("=== CHAT-LIKE GENERATION (GPU) ===")
# Qwen's chat template auto-enables conversation mode in llama-cli.
# --single-turn prevents the CLI from waiting for a second interactive user turn.
prompt = "ローカルLLMをGPUで動かすメリットを3点、簡潔に説明してください。"
_, chat_peak, chat_wall = run_with_peak(
    [
        str(CLI),
        "-m",
        str(MODEL),
        "-ngl",
        "99",
        "-c",
        "2048",
        "-n",
        "128",
        "--temp",
        "0",
        "--conversation",
        "--single-turn",
        "--simple-io",
        "--no-display-prompt",
        "-p",
        prompt,
    ]
)

print("=== SUMMARY MARKERS ===")
print(f"CPU_PEAK_VRAM_MIB={cpu_peak}")
print(f"GPU_PEAK_VRAM_MIB={gpu_peak}")
print(f"CHAT_PEAK_VRAM_MIB={chat_peak}")
print(f"CPU_WALL_SECONDS={cpu_wall:.3f}")
print(f"GPU_WALL_SECONDS={gpu_wall:.3f}")
print(f"CHAT_WALL_SECONDS={chat_wall:.3f}")
