from __future__ import annotations
import argparse, copy, os, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from data import load_dataset, split_loso, split_fewshot, EEGWindowDataset, DATASET_DEFAULTS
from models import EEGCrossSubjectNet
from module import build_source_statistics, relational_consistency_loss, subject_level_perturb
from utils import setup_logging, seed_everything, unique_path, fmt_mean_std


def parse_args():
    p = argparse.ArgumentParser("LOSO few-shot raw-EEG training with Duala L_rc + subject-level perturbation")
    p.add_argument("--dataset", required=True, choices=list(DATASET_DEFAULTS))
    p.add_argument("--data_path", default=None)
    p.add_argument("--target", default="valence",
                   choices=["valence","arousal","dominance","liking"],
                   help="Used by DREAMER/DEAP; ignored by other datasets.")
    p.add_argument("--n_shot", type=int, default=5, choices=[1,5,10,20])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_of_channels", type=int, default=0,
                   help="0=dataset default; otherwise assert/use this EEG channel count.")
    p.add_argument("--time_segments", type=int, default=400,
                   help="Samples per 2-s window. With target_fs=200 this must be 400.")
    p.add_argument("--transformer_tokens", type=int, default=400,
                   help="Temporal feature tokens kept before Spatial CNN/Transformer. "
                        "Default 100 reduces 400->100; 0 disables reduction.")
    p.add_argument("--target_fs", type=int, default=200)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--finetune_epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--finetune_lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lambda_rc", type=float, default=1.0)
    p.add_argument("--perturb_strength", type=float, default=1.0)
    p.add_argument("--temporal_kernels", type=str, default="7,15,31",
                   help="Comma-separated positive kernel sizes. Odd/even kernels are both supported.")
    p.add_argument("--run_tag", default="",
                   help="Optional run identifier appended to result/log directories; run_parallel.py "
                        "sets this automatically to avoid collisions.")
    p.add_argument("--temporal_branch_dim", type=int, default=16)
    p.add_argument("--temporal_dim", type=int, default=64)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--transformer_layers", type=int, default=2)
    p.add_argument("--transformer_heads", type=int, default=4)
    p.add_argument("--ff_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--result_root", default="results")
    p.add_argument("--log_root", default="logs")
    return p.parse_args()


def parse_temporal_kernels(spec):
    kernels = tuple(int(x.strip()) for x in spec.split(",") if x.strip())
    if not kernels:
        raise ValueError("--temporal_kernels must contain at least one integer")
    if any(k <= 0 for k in kernels):
        raise ValueError(f"All temporal kernels must be positive, got {kernels}")
    return kernels


def make_model(args, channels, classes):
    kernels = parse_temporal_kernels(args)
    return EEGCrossSubjectNet(
        num_channels=channels, num_classes=classes, time_segments=args.time_segments,
        transformer_tokens=args.transformer_tokens,
        temporal_kernels=kernels, temporal_branch_dim=args.temporal_branch_dim,
        temporal_dim=args.temporal_dim, embed_dim=args.embed_dim,
        transformer_layers=args.transformer_layers, transformer_heads=args.transformer_heads,
        ff_dim=args.ff_dim, dropout=args.dropout,
    )


def train_source(model, loader, args, device, fold):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for ep in range(1, args.epochs + 1):
        losses, correct, total = [], 0, 0
        for x, y, _ in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()
        print(f"[Fold {fold}] source epoch {ep:03d}/{args.epochs} "
              f"loss={np.mean(losses):.5f} acc={correct/max(total,1):.4f}")


def finetune_target(model, support_loader, stats, args, device, fold):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.finetune_lr, weight_decay=args.weight_decay)
    for ep in range(1, args.finetune_epochs + 1):
        for x, y, _ in support_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            z = model.encode(x)
            z_for_cls = subject_level_perturb(z, y, stats, args.perturb_strength)
            logits = model.classify_z(z_for_cls)
            loss_cls = F.cross_entropy(logits, y)
            loss_rc = relational_consistency_loss(z, y, stats)
            loss = loss_cls + args.lambda_rc * loss_rc
            loss.backward()
            opt.step()
        print(f"[Fold {fold}] finetune epoch {ep:03d}/{args.finetune_epochs} "
              f"loss={loss.item():.5f} cls={loss_cls.item():.5f} rc={loss_rc.item():.5f}")



@torch.no_grad()
def evaluate(model, loader, device):
    # Test with ORIGINAL z (no perturbation), exactly as requested.
    model.eval()
    ys, ps = [], []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        z = model.encode(x)
        pred = model.classify_z(z).argmax(1).cpu().numpy()
        ys.extend(y.numpy().tolist())
        ps.extend(pred.tolist())
    return accuracy_score(ys, ps), f1_score(ys, ps, average="macro", zero_division=0)


def write_header(f, args, data_path, channels, classes, subjects):
    f.write("="*118 + "\n")
    f.write("LOSO Few-shot EEG / Duala-style L_rc + Subject-level Feature Perturbation\n")
    f.write("="*118 + "\n")
    fields = {
        "dataset":args.dataset, "data_path":data_path, "target":args.target, "subjects":len(subjects),
        "classes":classes, "epochs":args.epochs, "finetune_epochs":args.finetune_epochs,
        "batch_size":args.batch_size, "n_shot":args.n_shot, "seed":args.seed, "gpu_id":args.gpu_id,
        "channels":channels, "time_segments":args.time_segments,
        "transformer_tokens":args.transformer_tokens, "target_fs":args.target_fs,
        "lr":args.lr, "finetune_lr":args.finetune_lr, "weight_decay":args.weight_decay,
        "lambda_rc":args.lambda_rc, "perturb_strength":args.perturb_strength,
        "run_tag":args.run_tag or "-",
        "temporal_kernels":args.temporal_kernels, "embed_dim":args.embed_dim,
        "transformer_layers":args.transformer_layers, "transformer_heads":args.transformer_heads,
    }
    for k,v in fields.items():
        f.write(f"{k:<24}: {v}\n")
    f.write("-"*118 + "\n")
    f.write(f"{'Fold':<10}{'TestSubject':<16}{'SourceTrainN':<18}{'FewShotN':<14}{'TestN':<14}"
            f"{'Accuracy(%)':<18}{'MacroF1(%)':<18}\n")
    f.write("-"*118 + "\n")
    f.flush()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required by this training script.")
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")
    seed_everything(args.seed)

    run_name = f"{args.dataset}_seed{args.seed}_{args.n_shot}shot"
    if args.run_tag:
        safe_tag = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in args.run_tag)
        run_name += f"_{safe_tag}"

    result_dir = Path(args.result_root) / run_name
    log_dir = Path(args.log_root) / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path, log_f = setup_logging(log_dir, f"train_{run_name}")

    start = time.time()
    samples, subjects, classes, channels, used_path = load_dataset(
        args.dataset, args.data_path, args.target_fs, args.time_segments,
        args.num_of_channels, args.target
    )
    effective_tokens = args.time_segments if args.transformer_tokens == 0 else args.transformer_tokens
    print(f"Loaded dataset={args.dataset}, path={used_path}, samples={len(samples)}, "
          f"subjects={len(subjects)}, classes={classes}, channels={channels}")
    kernels = parse_temporal_kernels(args)
    print(f"Temporal kernels: {kernels}")
    print(f"Temporal path: {args.time_segments} raw points -> {effective_tokens} tokens "
          f"before Spatial CNN/Transformer")

    result_stem = f"result_{args.dataset}_{args.n_shot}shot"
    result_path = unique_path(result_dir / f"{result_stem}.txt")
    accs, f1s = [], []
    with open(result_path, "a", encoding="utf-8", buffering=1) as rf:
        write_header(rf, args, used_path, channels, classes, subjects)

        for fold_idx, test_subject in enumerate(subjects, 1):
            fold_seed = args.seed + fold_idx
            seed_everything(fold_seed)
            source_samples, target_samples = split_loso(samples, test_subject)
            support, query = split_fewshot(target_samples, args.n_shot, classes, fold_seed)
            if not query:
                raise RuntimeError(f"Subject {test_subject}: empty test/query set after few-shot split.")

            print("\n" + "="*100)
            print(f"Fold {fold_idx}/{len(subjects)} | held-out subject={test_subject} | "
                  f"source={len(source_samples)} support={len(support)} test={len(query)}")
            print("="*100)

            source_ds = EEGWindowDataset(source_samples)
            source_loader = DataLoader(
                source_ds, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True, drop_last=False
            )
            # Separate deterministic loader for source statistics.
            stat_loader = DataLoader(
                source_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True, drop_last=False
            )
            support_loader = DataLoader(
                EEGWindowDataset(support), batch_size=len(support), shuffle=False,
                num_workers=0, pin_memory=True
            )
            test_loader = DataLoader(
                EEGWindowDataset(query), batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True
            )

            source_model = make_model(args, channels, classes).to(device)
            train_source(source_model, source_loader, args, device, fold_idx)

            stats = build_source_statistics(source_model, stat_loader, classes, device)
            target_model = copy.deepcopy(source_model)
            finetune_target(target_model, support_loader, stats, args, device, fold_idx)
            acc, mf1 = evaluate(target_model, test_loader, device)
            accs.append(acc); f1s.append(mf1)

            line = (f"{fold_idx:<10}{str(test_subject):<16}{len(source_samples):<18}"
                    f"{len(support):<14}{len(query):<14}{acc*100:<18.2f}{mf1*100:<18.2f}\n")
            rf.write(line)
            rf.flush()  # every fold is persisted immediately
            print(f"[Fold {fold_idx}] Accuracy={acc*100:.2f}% MacroF1={mf1*100:.2f}%")
            del source_model, target_model, stats
            torch.cuda.empty_cache()

        elapsed = time.time() - start
        rf.write("-"*118 + "\n")
        rf.write(f"{'ALL FOLDS':<58}{fmt_mean_std(accs):<18}{fmt_mean_std(f1s):<18}\n")
        rf.write(f"\nAccuracy mean/std : {fmt_mean_std(accs)}\n")
        rf.write(f"MacroF1 mean/std  : {fmt_mean_std(f1s)}\n")
        rf.write(f"Total runtime (s) : {elapsed:.2f}\n")
        rf.write(f"Total runtime     : {elapsed/3600:.3f} h\n")
        rf.flush()

    print(f"\nRESULT FILE: {result_path}")
    print(f"LOG FILE:    {log_path}")
    print(f"TOTAL TIME:  {time.time()-start:.2f} s")


if __name__ == "__main__":
    main()
