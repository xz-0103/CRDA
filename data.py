from __future__ import annotations
import os, re, glob
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly
from torch.utils.data import Dataset


DATASET_DEFAULTS = {
    "stroop":  dict(path="", channels=16, classes=3, fs=200),
    "DREAMER": dict(path="", channels=14, classes=2, fs=128),
    "SEED":    dict(path="", channels=62, classes=3, fs=200),
    "EEGMAT":  dict(path="", channels=19, classes=2, fs=200),
}

SEED_LABELS = [2,1,0,0,1,2,0,1,2,2,1,0,1,2,0]
SEED_IV_LABELS = {
    1: [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
    2: [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
    3: [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0],
}


def _zscore_channels(x):
    x = np.asarray(x, dtype=np.float32)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    return (x - mu) / np.maximum(sd, 1e-6)


def _resample(x, orig_fs, target_fs):
    if orig_fs == target_fs:
        return x.astype(np.float32, copy=False)
    from math import gcd
    g = gcd(int(orig_fs), int(target_fs))
    return resample_poly(x, target_fs // g, orig_fs // g, axis=1).astype(np.float32)


def _slice_2s(x, fs=200, time_segments=400):
    # User-requested non-overlapping 2-second windows.
    expected = int(round(2.0 * fs))
    if time_segments != expected:
        raise ValueError(
            f"time_segments={time_segments}, but 2 s at target_fs={fs} requires {expected}. "
            "Use --time_segments equal to 2*target_fs."
        )
    n = x.shape[1] // time_segments
    return [x[:, i*time_segments:(i+1)*time_segments].copy() for i in range(n)]


def _normalize_labels(y):
    y = np.asarray(y).reshape(-1).astype(int)
    uniq = sorted(np.unique(y).tolist())

    if uniq == list(range(len(uniq))):
        return y
    mp = {v:i for i,v in enumerate(uniq)}
    return np.asarray([mp[int(v)] for v in y], dtype=np.int64)


def _loadmat(path):
    try:
        return sio.loadmat(path)
    except NotImplementedError:
        import h5py
        out = {}
        with h5py.File(path, "r") as f:
            for k in f.keys():
                out[k] = np.array(f[k])
        return out


def _trial_to_windows(trial, orig_fs, target_fs, channels, time_segments):
    x = np.asarray(trial)
    if x.ndim != 2:
        raise ValueError(f"EEG trial must be 2D, got {x.shape}")
    # Convert to [C,T].
    if x.shape[0] != channels and x.shape[1] == channels:
        x = x.T
    if x.shape[0] != channels:
        raise ValueError(f"Expected {channels} EEG channels, got {x.shape}")
    # z-score per channel within the continuous trial, then resample and cut.
    x = _zscore_channels(x)
    x = _resample(x, orig_fs, target_fs)
    return _slice_2s(x, target_fs, time_segments)


@dataclass
class Sample:
    x: np.ndarray
    y: int
    subject: int


class EEGWindowDataset(Dataset):
    def __init__(self, samples: List[Sample]):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        import torch
        s = self.samples[idx]
        return torch.from_numpy(s.x), torch.tensor(s.y, dtype=torch.long), torch.tensor(s.subject, dtype=torch.long)


def _subject_id_from_name(name):
    nums = re.findall(r"\d+", os.path.basename(name))
    if not nums:
        raise ValueError(f"Cannot parse subject id from {name}")
    return int(nums[0])


def _eeg_fields(mat, n):
    fields = []
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        m = re.search(r"_eeg(\d+)$", k, flags=re.I)
        if m:
            fields.append((int(m.group(1)), v))
    fields.sort(key=lambda t: t[0])
    if len(fields) < n:
        raise ValueError(f"Expected >= {n} EEG trial fields, found {len(fields)}")
    return [v for _, v in fields[:n]]


def load_dataset(name, data_path=None, target_fs=200, time_segments=400, num_channels=0, target="valence"):
    if name not in DATASET_DEFAULTS:
        raise ValueError(f"Unknown dataset {name}; choose from {list(DATASET_DEFAULTS)}")
    cfg = DATASET_DEFAULTS[name]
    path = data_path or cfg["path"]
    channels = int(num_channels or cfg["channels"])
    samples = []

    if name == "DREAMER":
        files = sorted(glob.glob(os.path.join(path, "subject*.mat")))
        for fp in files:
            sid = _subject_id_from_name(fp)
            m = _loadmat(fp)
            eeg_obj = m["EEG"]
            score_key = {"valence":"ScoreValence","arousal":"ScoreArousal","dominance":"ScoreDominance"}[target]
            scores = np.asarray(m[score_key]).reshape(-1)
            for i in range(18):
                # scipy loadmat object array is commonly [1,18].
                trial = eeg_obj[0, i] if eeg_obj.ndim == 2 else eeg_obj[i]
                y = int(scores[i] >= 3)
                for w in _trial_to_windows(trial, 128, target_fs, channels, time_segments):
                    samples.append(Sample(w, y, sid))

    elif name == "DEAP":
        files = sorted(glob.glob(os.path.join(path, "subject*.mat")))
        col = {"valence":0, "arousal":1, "dominance":2, "liking":3}[target]
        for fp in files:
            sid = _subject_id_from_name(fp)
            m = _loadmat(fp)
            data = np.asarray(m["data"])
            labels = np.asarray(m["labels"])
            for i in range(data.shape[0]):
                trial = data[i, :32, :]
                y = int(labels[i, col] >= 5)
                for w in _trial_to_windows(trial, 128, target_fs, channels, time_segments):
                    samples.append(Sample(w, y, sid))

    elif name == "stroop":
        task_map = {"1back":0, "stroop1":0, "2back":1, "stroop2":1, "3back":2, "stroop3":2}
        files = sorted(glob.glob(os.path.join(path, "*.mat")))
        for fp in files:
            base = os.path.splitext(os.path.basename(fp))[0]
            sid = _subject_id_from_name(base)
            task = base.split("_", 1)[1]
            if task not in task_map:
                continue
            m = _loadmat(fp)
            for w in _trial_to_windows(m["eeg"], 200, target_fs, channels, time_segments):
                samples.append(Sample(w, task_map[task], sid))

    elif name == "EEGMAT":
        files = sorted(glob.glob(os.path.join(path, "*.mat")))
        for fp in files:
            base = os.path.splitext(os.path.basename(fp))[0]
            sid = _subject_id_from_name(base)
            label_raw = int(base.rsplit("_", 1)[1])
            y = label_raw - 1
            m = _loadmat(fp)
            for w in _trial_to_windows(m["eeg"], 200, target_fs, channels, time_segments):
                samples.append(Sample(w, y, sid))

    elif name == "SEED":
        files = sorted(glob.glob(os.path.join(path, "*.mat")))
        for fp in files:
            sid = _subject_id_from_name(fp)
            m = _loadmat(fp)
            trials = _eeg_fields(m, 15)
            for trial, y in zip(trials, SEED_LABELS):
                for w in _trial_to_windows(trial, 200, target_fs, channels, time_segments):
                    samples.append(Sample(w, int(y), sid))

    elif name == "SEED_IV":
        # session folders 1,2,3; same subject is grouped across all sessions.
        for sess in (1,2,3):
            sess_dir = os.path.join(path, str(sess))
            for fp in sorted(glob.glob(os.path.join(sess_dir, "*.mat"))):
                sid = _subject_id_from_name(fp)
                m = _loadmat(fp)
                trials = _eeg_fields(m, 24)
                for trial, y in zip(trials, SEED_IV_LABELS[sess]):
                    for w in _trial_to_windows(trial, 200, target_fs, channels, time_segments):
                        samples.append(Sample(w, int(y), sid))

    elif name in ("BCI2a", "BCI2b"):
        xfiles = sorted(glob.glob(os.path.join(path, f"{name}_sub*_X.npy")))
        for xfp in xfiles:
            sid = _subject_id_from_name(os.path.basename(xfp).replace(name, ""))
            yfp = xfp.replace("_X.npy", "_y.npy")
            X = np.load(xfp, allow_pickle=True)
            y = _normalize_labels(np.load(yfp, allow_pickle=True))
            for trial, label in zip(X, y):
                for w in _trial_to_windows(trial, 200, target_fs, channels, time_segments):
                    samples.append(Sample(w, int(label), sid))

    if not samples:
        raise RuntimeError(f"No samples loaded from {path} for dataset {name}")
    subjects = sorted({s.subject for s in samples})
    labels = sorted({s.y for s in samples})
    return samples, subjects, len(labels), channels, path


def split_loso(samples, test_subject):
    train = [s for s in samples if s.subject != test_subject]
    target = [s for s in samples if s.subject == test_subject]
    return train, target


def split_fewshot(target_samples, n_shot, num_classes, seed):

    rng = np.random.default_rng(seed)
    support_idx = []
    for c in range(num_classes):
        idx = np.asarray([i for i,s in enumerate(target_samples) if s.y == c], dtype=int)
        if len(idx) < n_shot:
            raise RuntimeError(f"Class {c} has only {len(idx)} target windows, < n_shot={n_shot}")
        chosen = rng.choice(idx, size=n_shot, replace=False)
        support_idx.extend(chosen.tolist())
    support_set = set(support_idx)
    support = [target_samples[i] for i in support_idx]
    query = [s for i,s in enumerate(target_samples) if i not in support_set]
    return support, query
