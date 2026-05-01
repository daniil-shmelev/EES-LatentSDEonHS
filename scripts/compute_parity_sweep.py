"""Compute-parity sweep: train each solver to a fixed compute budget B,
where B is the total per-trajectory NN-eval count (Geo-Euler N=B, RKMK Heun
N=B/2, CFEES N=B/3). Evaluates final test ELBO + Human Activity classification
accuracy for each (solver, B, seed) cell.

Trade-off: integration grid resolution (number of steps N) is decoupled from
the data observation grid (228 timepoints). At training time we use a coarse
``desired_t`` of length N+1 spanning [0, 0.99], remapping observation indices
``inp_tid in [0, 228)`` into the new grid via floor(inp_tid * (N+1) / 228).
This is a discretization choice; smaller N -> coarser observation buckets.

Output: ``results/compute_parity.csv`` with columns
``solver, B, N, seed, test_elbo, test_acc, val_acc, train_loss, train_time_s``.

Usage::

    .venv/bin/python scripts/compute_parity_sweep.py --budgets 30 100 --seeds 0 1 --epochs 30
"""

import argparse
import csv
import gc
import os
import sys
import time
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.activity_provider import HumanActivityProvider
from core.sde_solvers import geometric_euler, NEVALS_PER_STEP
from core.models import (
    ELBO,
    ActivityRecogNetwork,
    GenericMLP,
    PathToGaussianDecoder,
    PerTimePointCrossEntropyLoss,
    default_SOnPathDistributionEncoder,
)
from utils.misc import set_seed


SOLVERS = ["geometric_euler", "cfees25"]
DEFAULT_BUDGETS = [30, 100, 300]


def n_for_budget(solver: str, B: int) -> int:
    return max(1, B // NEVALS_PER_STEP[solver])


def remap_tid(tid: torch.Tensor, num_observation_tps: int, num_grid_tps: int) -> torch.Tensor:
    """Remap observation indices [0, num_observation_tps) into [0, num_grid_tps).

    Args:
        tid: original observation indices in [0, num_observation_tps).
        num_observation_tps: 228 for Human Activity.
        num_grid_tps: N+1 for the coarser grid.
    """
    new_tid = (tid.float() * num_grid_tps / num_observation_tps).long()
    new_tid.clamp_(max=num_grid_tps - 1)
    return new_tid


class CoarseGridDataset(torch.utils.data.Dataset):
    """Wraps a HumanActivityDataset with on-the-fly tid remapping to a coarser grid."""

    def __init__(self, base_ds, num_grid_tps: int):
        self.base_ds = base_ds
        self.num_grid_tps = num_grid_tps
        # Original observation grid size (228 for Human Activity).
        self.num_observation_tps = base_ds.num_timepoints if hasattr(base_ds, "num_timepoints") else 228

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        item = self.base_ds[idx]
        new_tid = remap_tid(item["inp_tid"], self.num_observation_tps, self.num_grid_tps)
        return {
            "inp_obs": item["inp_obs"],
            "inp_msk": item["inp_msk"],
            "inp_tid": new_tid,
            "inp_tps": new_tid.float() / self.num_grid_tps,
            "evd_obs": item["evd_obs"],
            "evd_msk": item["evd_msk"],
            "evd_tid": new_tid,  # remapped; ELBO scatters to this
            "aux_obs": item["aux_obs"],
            "aux_tid": new_tid,
        }


def build_modules(args, device, num_classes, input_dim, solver_kind, adjoint):
    if solver_kind == "geometric_euler":
        solver_arg = geometric_euler
    else:
        solver_arg = solver_kind
    recog_net = ActivityRecogNetwork(
        mtan_input_dim=input_dim, mtan_hidden_dim=args.h_dim, use_atanh=True
    )
    recon_net = GenericMLP(inp_dim=args.z_dim, out_dim=input_dim, n_layers=1)
    qzx_net = default_SOnPathDistributionEncoder(
        h_dim=args.h_dim, z_dim=args.z_dim, n_deg=args.n_deg, learnable_prior=False,
        time_min=0.0, time_max=2.0,
        solver=solver_arg, adjoint=adjoint,
    )
    pxz_net = PathToGaussianDecoder(mu_map=recon_net, sigma_map=None, initial_sigma=1.0)
    aux_net = GenericMLP(inp_dim=args.z_dim, out_dim=num_classes,
                         n_hidden=args.aux_hidden_dim, n_layers=1)
    modules = nn.ModuleDict({
        "recog_net": recog_net, "recon_net": recon_net,
        "pxz_net": pxz_net, "qzx_net": qzx_net, "aux_net": aux_net,
    }).to(device)
    return modules


def _save_checkpoint_atomic(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def train_one(args, dl_trn, dl_val, dl_tst, modules, desired_t, device, n_epochs,
              checkpoint_path: Optional[str] = None):
    elbo = ELBO(reduction="mean")
    aux_loss = PerTimePointCrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.Adam(modules.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, args.restart, eta_min=0.0)

    start_epoch = 1
    elapsed_before = 0.0
    final_train_loss = float("nan")

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        modules.load_state_dict(ckpt["modules"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        torch.set_rng_state(ckpt["torch_rng"].cpu())
        if torch.cuda.is_available() and ckpt.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ckpt["cuda_rng"].cpu())
        start_epoch = int(ckpt["epoch"]) + 1
        elapsed_before = float(ckpt["elapsed"])
        final_train_loss = float(ckpt["final_train_loss"])
        print(f"  [resume cell] from epoch {start_epoch - 1}/{n_epochs} "
              f"({elapsed_before:.0f}s elapsed)", flush=True)

    t0 = time.time()
    for ep in range(start_epoch, n_epochs + 1):
        aux_w_mul = (ep / 60) ** 2 if ep < 60 else 1.0
        modules.train()
        loss_acc = 0.0
        n_acc = 0
        for batch in dl_trn:
            parts = {k: v.to(device) for k, v in batch.items()}
            inp = (parts["inp_obs"], parts["inp_msk"], parts["inp_tps"])
            h = modules["recog_net"](inp)
            qzx, pz = modules["qzx_net"](h, desired_t)
            zis = qzx.rsample((args.mc_train_samples,))
            pxz = modules["pxz_net"](zis)
            aux = modules["aux_net"](zis)
            elbo_val, _ = elbo(
                qzx, pz, pxz, parts["evd_obs"], parts["evd_tid"], parts["evd_msk"],
                {"kl0_weight": args.kl0_weight, "klp_weight": args.klp_weight,
                 "pxz_weight": args.pxz_weight},
            )
            aux_val = aux_loss(aux, parts["aux_obs"], parts["aux_tid"])
            loss = elbo_val + args.aux_weight * aux_w_mul * aux_val
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_acc += loss.item() * parts["evd_obs"].shape[0]
            n_acc += parts["evd_obs"].shape[0]
        scheduler.step()
        final_train_loss = loss_acc / max(n_acc, 1)

        if checkpoint_path is not None:
            _save_checkpoint_atomic(checkpoint_path, {
                "epoch": ep,
                "modules": modules.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                "final_train_loss": final_train_loss,
                "elapsed": elapsed_before + (time.time() - t0),
            })
    train_time_s = elapsed_before + (time.time() - t0)

    # Eval
    modules.eval()
    test_acc, test_elbo, val_acc = evaluate(args, modules, dl_val, dl_tst, desired_t,
                                            elbo, aux_loss, device)
    return {
        "test_elbo": test_elbo,
        "test_acc": test_acc,
        "val_acc": val_acc,
        "train_loss": final_train_loss,
        "train_time_s": train_time_s,
    }


def evaluate(args, modules, dl_val, dl_tst, desired_t, elbo, aux_loss, device):
    def _eval(dl):
        modules.eval()
        elbo_acc = 0.0
        acc_acc = 0.0
        n_acc = 0
        with torch.no_grad():
            for batch in dl:
                parts = {k: v.to(device) for k, v in batch.items()}
                inp = (parts["inp_obs"], parts["inp_msk"], parts["inp_tps"])
                h = modules["recog_net"](inp)
                qzx, pz = modules["qzx_net"](h, desired_t)
                zis = qzx.rsample((args.mc_eval_samples,))
                pxz = modules["pxz_net"](zis)
                aux = modules["aux_net"](zis)
                elbo_val, _ = elbo(
                    qzx, pz, pxz, parts["evd_obs"], parts["evd_tid"], parts["evd_msk"],
                    {"kl0_weight": args.kl0_weight, "klp_weight": args.klp_weight,
                     "pxz_weight": args.pxz_weight},
                )
                # accuracy
                mc, batch_len, num_tps, dim = aux.shape
                idx = (parts["aux_tid"].view([1, batch_len, -1, 1])
                       .repeat(mc, 1, 1, dim).long())
                input_at_tps = aux.gather(2, idx).mean(dim=0)
                acc = (input_at_tps.flatten(0, 1).argmax(1) == parts["aux_obs"].flatten()).float().mean()
                bsz = parts["evd_obs"].shape[0]
                elbo_acc += elbo_val.item() * bsz
                acc_acc += acc.item() * bsz
                n_acc += bsz
        return elbo_acc / max(n_acc, 1), acc_acc / max(n_acc, 1)

    val_elbo, val_acc = _eval(dl_val)
    test_elbo, test_acc = _eval(dl_tst)
    return test_acc, test_elbo, val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default="results/compute_parity.csv")
    parser.add_argument("--budgets", type=int, nargs="*", default=DEFAULT_BUDGETS)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    parser.add_argument("--solvers", type=str, nargs="*", default=SOLVERS)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--restart", type=int, default=30)
    parser.add_argument("--z-dim", type=int, default=16)
    parser.add_argument("--h-dim", type=int, default=128)
    parser.add_argument("--n-deg", type=int, default=4)
    parser.add_argument("--kl0-weight", type=float, default=1e-4)
    parser.add_argument("--klp-weight", type=float, default=1e-4)
    parser.add_argument("--pxz-weight", type=float, default=1.0)
    parser.add_argument("--aux-weight", type=float, default=10.0)
    parser.add_argument("--aux-hidden-dim", type=int, default=32)
    parser.add_argument("--mc-train-samples", type=int, default=1)
    parser.add_argument("--mc-eval-samples", type=int, default=1)
    parser.add_argument("--data-dir", default="data_dir")
    parser.add_argument(
        "--download", action="store_true",
        help="Download the UCI Human Activity dataset if it's not already at "
             "--data-dir. Off by default; turn on the first time you run on a "
             "machine without the data."
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="If set and the output CSV already exists, skip (solver, B, seed) "
             "cells already present. Useful for long runs that may be interrupted."
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="If set, save a per-epoch training checkpoint (model, optimizer, "
             "scheduler, RNG state) for each cell to <dir>/parity_<solver>_B<B>"
             "_seed<seed>.pt. On a crash or reboot, re-running the script "
             "resumes mid-cell from the last completed epoch (in addition to "
             "the cell-level resume from --resume). Default ``<args.out>.ckpt`` "
             "when ``--preset zeng2023`` is used; otherwise off."
    )
    parser.add_argument(
        "--preset", choices=["default", "zeng2023"], default="default",
        help="``zeng2023`` reproduces tab:sphere_parity with the published Zeng "
             "et al. 2023 hyperparameters (990 epochs, 10 seeds, B=30) using the "
             "original ELBO loss with no subsampling. Equivalent to: "
             "--epochs 990 --seeds 0..9 --budgets 30 --resume "
             "--out results/compute_parity_990epochs.csv. Other CLI args still "
             "override (e.g. --device cuda:1)."
    )
    args = parser.parse_args()
    if args.preset == "zeng2023":
        # Apply preset only to fields the user did NOT explicitly set, so
        # --device/--data-dir/--batch-size etc. still override.
        defaults = parser.parse_args([])
        if args.epochs == defaults.epochs:
            args.epochs = 990
        if args.seeds == defaults.seeds:
            args.seeds = list(range(10))
        if args.budgets == defaults.budgets:
            args.budgets = [30]
        if args.out == defaults.out:
            args.out = "results/compute_parity_990epochs.csv"
        args.resume = True
        if args.checkpoint_dir is None:
            args.checkpoint_dir = args.out + ".ckpt"

    device = torch.device(args.device)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    provider = HumanActivityProvider(args.data_dir, download=args.download)

    # Resume support: load any rows that are already present in the output CSV.
    rows = []
    done_keys = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out, "r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                # Coerce numeric fields back to int/float for downstream consistency.
                row = dict(r)
                for k in ("B", "N", "seed"):
                    if k in row:
                        row[k] = int(row[k])
                for k in ("test_elbo", "test_acc", "val_acc", "train_loss", "train_time_s"):
                    if k in row and row[k] != "":
                        row[k] = float(row[k])
                rows.append(row)
                done_keys.add((row["solver"], int(row["B"]), int(row["seed"])))
        print(f"[resume] loaded {len(rows)} completed cells from {args.out}", flush=True)

    for B in args.budgets:
        for solver in args.solvers:
            N = n_for_budget(solver, B)
            print(f"\n=== B={B}, solver={solver}, N={N} ===", flush=True)
            for seed in args.seeds:
                if (solver, B, seed) in done_keys:
                    print(f"  seed={seed}: SKIP (already in CSV)", flush=True)
                    continue
                set_seed(seed + 1)
                # Build coarser-grid datasets/loaders.
                dl_trn = torch.utils.data.DataLoader(
                    CoarseGridDataset(provider._ds_trn, N + 1),
                    batch_size=args.batch_size, shuffle=True,
                )
                dl_val = torch.utils.data.DataLoader(
                    CoarseGridDataset(provider._ds_val, N + 1),
                    batch_size=args.batch_size, shuffle=False,
                )
                dl_tst = torch.utils.data.DataLoader(
                    CoarseGridDataset(provider._ds_tst, N + 1),
                    batch_size=args.batch_size, shuffle=False,
                )
                desired_t = torch.linspace(0.0, 0.99, N + 1, device=device)

                adjoint = "reversible" if solver == "cfees25" else "autograd"
                modules = build_modules(args, device, provider.num_classes,
                                        provider.input_dim, solver, adjoint)

                ckpt_path = (os.path.join(args.checkpoint_dir,
                                          f"parity_{solver}_B{B}_seed{seed}.pt")
                             if args.checkpoint_dir else None)
                result = train_one(args, dl_trn, dl_val, dl_tst, modules,
                                   desired_t, device, args.epochs,
                                   checkpoint_path=ckpt_path)
                row = {
                    "solver": solver, "B": B, "N": N, "seed": seed,
                    **result,
                }
                rows.append(row)
                print(
                    f"  seed={seed}: test_elbo={result['test_elbo']:.4f}  "
                    f"test_acc={result['test_acc']:.4f}  "
                    f"val_acc={result['val_acc']:.4f}  "
                    f"time={result['train_time_s']:.0f}s", flush=True,
                )
                # Atomic write so an interruption mid-write can't corrupt the
                # resume file: write to .tmp, then rename.
                tmp_path = args.out + ".tmp"
                with open(tmp_path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    for r in rows:
                        w.writerow(r)
                os.replace(tmp_path, args.out)
                # Cell durably persisted -- safe to drop the per-epoch checkpoint.
                if ckpt_path is not None and os.path.exists(ckpt_path):
                    os.remove(ckpt_path)
                del modules, dl_trn, dl_val, dl_tst, desired_t
                gc.collect()
                torch.cuda.empty_cache()

    print(f"\nWrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
