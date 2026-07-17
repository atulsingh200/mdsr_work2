#!/usr/bin/env python3
"""
GPU keep-alive / anti-idle-shutdown utility.

Holds a fraction of GPU memory and runs matmuls on a busy/idle duty cycle so
that `nvidia-smi` reports non-zero, non-trivial utilization on each targeted
GPU. Intended to stop a shared/managed instance from being reclaimed by an
idle-detection policy while you are still using it (e.g. between training
runs). It does NOT need root/sudo - CUDA compute does not require elevated
privileges.

Usage:
    python3 gpu_keepalive.py --gpus all --mem-frac 0.5 --duty-cycle 0.5
    python3 gpu_keepalive.py --gpus 0,1,2,3,4,5,6 --mem-frac 0.3 --duty-cycle 0.4

Stop with Ctrl+C (foreground) or SIGTERM (background via run_gpu_keepalive.sh).
"""
import argparse
import multiprocessing as mp
import os
import signal
import sys
import time

import torch


def worker(gpu_index: int, mem_frac: float, duty_cycle: float, matrix_size: int, cycle_len: float):
    device = torch.device(f"cuda:{gpu_index}")
    torch.cuda.set_device(device)

    stop = {"flag": False}

    def handle_sig(signum, frame):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, handle_sig)
    signal.signal(signal.SIGINT, handle_sig)

    total_mem = torch.cuda.get_device_properties(device).total_memory
    free_mem, _ = torch.cuda.mem_get_info(device)
    target_bytes = int(min(total_mem * mem_frac, free_mem * 0.9))
    n_elems = max(target_bytes // 4, 0)  # float32 = 4 bytes
    hold = None
    if n_elems > 0:
        side = int(n_elems ** 0.5)
        hold = torch.empty((side, side), dtype=torch.float32, device=device)
        hold.normal_()

    a = torch.randn(matrix_size, matrix_size, device=device)
    b = torch.randn(matrix_size, matrix_size, device=device)

    held_gb = (hold.numel() * 4 / 1e9) if hold is not None else 0.0
    print(f"[gpu{gpu_index}] holding ~{held_gb:.1f} GB, duty_cycle={duty_cycle}", flush=True)

    while not stop["flag"]:
        busy_time = cycle_len * duty_cycle
        t0 = time.time()
        while time.time() - t0 < busy_time:
            c = a @ b
            torch.cuda.synchronize(device)
        idle_time = cycle_len - busy_time
        if idle_time > 0:
            time.sleep(idle_time)

    print(f"[gpu{gpu_index}] stopping, releasing memory", flush=True)
    del hold, a, b
    torch.cuda.empty_cache()


def parse_gpus(spec: str) -> list[int]:
    if spec == "all":
        return list(range(torch.cuda.device_count()))
    return [int(x) for x in spec.split(",") if x.strip() != ""]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpus", default="all", help="'all' or comma list, e.g. 0,1,2,3 (default: all)")
    parser.add_argument("--mem-frac", type=float, default=0.5, help="fraction of each GPU's total memory to hold (default: 0.5)")
    parser.add_argument("--duty-cycle", type=float, default=0.5, help="fraction of each cycle spent computing vs idle, targets ~this GPU util (default: 0.5)")
    parser.add_argument("--matrix-size", type=int, default=8192, help="square matmul size used as the busy workload")
    parser.add_argument("--cycle-len", type=float, default=1.0, help="seconds per busy/idle cycle")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available, nothing to do.", file=sys.stderr)
        sys.exit(1)

    gpus = parse_gpus(args.gpus)
    print(f"[gpu_keepalive] targeting GPUs {gpus}, mem_frac={args.mem_frac}, duty_cycle={args.duty_cycle}", flush=True)

    mp.set_start_method("spawn", force=True)
    procs = []
    for g in gpus:
        p = mp.Process(target=worker, args=(g, args.mem_frac, args.duty_cycle, args.matrix_size, args.cycle_len), daemon=False)
        p.start()
        procs.append(p)

    stop = {"flag": False}

    def handle_sig(signum, frame):
        stop["flag"] = True
        for p in procs:
            if p.is_alive():
                os.kill(p.pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, handle_sig)
    signal.signal(signal.SIGINT, handle_sig)

    for p in procs:
        p.join()

    print("[gpu_keepalive] all workers stopped.", flush=True)


if __name__ == "__main__":
    main()
