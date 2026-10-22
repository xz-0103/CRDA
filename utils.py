from __future__ import annotations
import os, sys, time, random
from pathlib import Path
import numpy as np
import torch


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()


def unique_path(path):
    path = Path(path)
    if not path.exists():
        return path
    i = 1
    while True:
        p = path.with_name(f"{path.stem}_{i}{path.suffix}")
        if not p.exists():
            return p
        i += 1


def setup_logging(log_dir, stem):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    p = unique_path(Path(log_dir) / f"{stem}.log")
    f = open(p, "a", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.__stdout__, f)
    sys.stderr = Tee(sys.__stderr__, f)
    print("=" * 100)
    print("COMMAND:", " ".join(sys.argv))
    print("LOG:", p)
    print("=" * 100)
    return p, f


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fmt_mean_std(values):
    a = np.asarray(values, dtype=float)
    return f"{a.mean()*100:.2f}/{a.std(ddof=0)*100:.2f}"
