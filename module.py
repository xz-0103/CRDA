from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import torch
import torch.nn.functional as F


@dataclass
class SourceStatistics:
    # class_mean: [C,D]
    class_mean: torch.Tensor
    # class_scale: [C], mean source-subject sigma for each class
    class_scale: torch.Tensor
    # reference class-similarity matrix: [C,C]
    reference_similarity: torch.Tensor
    # valid pairs in reference matrix: [C,C] bool
    valid_pairs: torch.Tensor


@torch.no_grad()
def build_source_statistics(model, loader, num_classes, device):
    model.eval()
    sums, counts = {}, {}
    norm_sums, norm_counts = {}, {}

    for x, y, subject in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device)
        z = model.encode(x)
        zn = F.normalize(z, dim=1)
        subject = subject.cpu().tolist()
        for i, sid in enumerate(subject):
            c = int(y[i].item())
            key = (int(sid), c)
            sums[key] = sums.get(key, torch.zeros_like(z[i])) + z[i]
            counts[key] = counts.get(key, 0) + 1
            norm_sums[key] = norm_sums.get(key, torch.zeros_like(zn[i])) + zn[i]
            norm_counts[key] = norm_counts.get(key, 0) + 1

    subject_ids = sorted({s for s, _ in sums})
    embed_dim = next(iter(sums.values())).numel()
    class_means = torch.zeros(num_classes, embed_dim, device=device)
    class_scales = torch.ones(num_classes, device=device)

    for c in range(num_classes):
        zbars = [sums[(s,c)] / counts[(s,c)] for s in subject_ids if (s,c) in sums]
        if not zbars:
            raise RuntimeError(f"No source embedding found for class {c}")
        stack = torch.stack(zbars)
        class_means[c] = stack.mean(dim=0)
        sigmas = torch.sqrt(stack.var(dim=1, unbiased=False).clamp_min(1e-8))
        class_scales[c] = sigmas.mean().clamp_min(1e-4)

    sim_sum = torch.zeros(num_classes, num_classes, device=device)
    sim_count = torch.zeros(num_classes, num_classes, device=device)
    for s in subject_ids:
        available = [c for c in range(num_classes) if (s,c) in norm_sums]
        if len(available) < 2:
            continue
        proto = {}
        for c in available:
            p = norm_sums[(s,c)] / norm_counts[(s,c)]
            proto[c] = F.normalize(p, dim=0)
        for c1 in available:
            for c2 in available:
                sim_sum[c1,c2] += torch.dot(proto[c1], proto[c2])
                sim_count[c1,c2] += 1

    valid = sim_count > 0
    ref = torch.zeros_like(sim_sum)
    ref[valid] = sim_sum[valid] / sim_count[valid]
    return SourceStatistics(class_means, class_scales, ref, valid)


def relational_consistency_loss(z, y, stats: SourceStatistics):
    classes = torch.unique(y)
    protos = {}
    for c_t in classes:
        c = int(c_t.item())
        zn = F.normalize(z[y == c], dim=1)
        protos[c] = F.normalize(zn.mean(dim=0), dim=0)

    losses = []
    cls = sorted(protos)
    # Off-diagonal pairs carry the useful inter-class geometry.
    for i, c1 in enumerate(cls):
        for c2 in cls[i+1:]:
            if bool(stats.valid_pairs[c1, c2]):
                s = torch.dot(protos[c1], protos[c2])
                losses.append((s - stats.reference_similarity[c1,c2]) ** 2)
    if not losses:
        return z.sum() * 0.0
    return torch.stack(losses).mean()


def subject_level_perturb(z, y, stats: SourceStatistics, strength=1.0):
    mu = stats.class_mean[y]
    scale = stats.class_scale[y].unsqueeze(1)
    return mu + float(strength) * scale * (z - mu)
