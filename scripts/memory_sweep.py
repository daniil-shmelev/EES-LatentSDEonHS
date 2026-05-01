"""Memory sweep: peak VRAM as a function of integrator-step count N.

Sweeps three (solver, adjoint) cells across a range of N values, doing a single
forward + backward pass through a synthetic mini-batch matched to the Human
Activity model size (batch=64, z_dim=16, h_dim=128). Records
``torch.cuda.max_memory_allocated()`` per cell. CUDA OOMs are caught and
recorded as ``oom``.

Cells:
1. ``geometric_euler`` + autograd (full-tape baseline)
2. ``geometric_euler`` + ``torch.utils.checkpoint`` (whole-integrator chunk)
3. ``cfees25`` + reversible adjoint (the headline)

Output: ``results/memory_sweep.csv``.

Run::

    .venv/bin/python scripts/memory_sweep.py
"""

import argparse
import csv
import gc
import os
import sys
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import (
    ActivityRecogNetwork,
    ELBO,
    GenericMLP,
    PathToGaussianDecoder,
    PerTimePointCrossEntropyLoss,
    default_SOnPathDistributionEncoder,
)
from core.sde_solvers import geometric_euler


# Sweep configuration.
N_VALUES = [50, 200, 800, 2000, 5000]
CELLS = [
    ("geometric_euler", "autograd"),
    ("geometric_euler", "checkpoint"),
    ("cfees25", "reversible"),
]
# Number of observation timepoints per UCI HAR sequence -- fixed by the sensor
# sampling rate. The integrator can refine N independently; the encoder and
# decoder operate on this fixed observation grid.
UCI_HAR_K_OBS = 228


def _subsample_recon_targets(zis_obs, evd_obs, evd_msk, evd_tid, aux_obs, aux_tid, M):
    """Pick M random observation timepoints, slice the trajectory and the
    target tensors to those indices, and rebuild ``evd_tid`` / ``aux_tid``
    as identity in the new M-point grid (the original indices were positions
    in the full observation grid).
    """
    T_full = zis_obs.shape[-2]
    idx = torch.randperm(T_full, device=zis_obs.device)[:M]
    idx_sorted, _ = torch.sort(idx)
    zis_obs = zis_obs[..., idx_sorted, :]
    evd_obs = evd_obs[..., idx_sorted, :]
    evd_msk = evd_msk[..., idx_sorted, :]
    aux_obs = aux_obs[..., idx_sorted]
    new_tid = torch.arange(M, device=evd_tid.device).unsqueeze(0).expand(evd_tid.shape[0], -1).long()
    return zis_obs, evd_obs, evd_msk, new_tid, aux_obs, new_tid


def make_modules(device, h_dim=128, z_dim=16, n_deg=4, num_classes=7, input_dim=12,
                 solver=geometric_euler, adjoint="autograd", kl_subsample_M=None):
    recog_net = ActivityRecogNetwork(
        mtan_input_dim=input_dim, mtan_hidden_dim=h_dim, use_atanh=True,
    )
    recon_net = GenericMLP(inp_dim=z_dim, out_dim=input_dim, n_layers=1)
    qzx_net = default_SOnPathDistributionEncoder(
        h_dim=h_dim, z_dim=z_dim, n_deg=n_deg, learnable_prior=False,
        time_min=0.0, time_max=2.0,
        solver=solver, adjoint=adjoint,
        kl_subsample_M=kl_subsample_M,
    )
    pxz_net = PathToGaussianDecoder(mu_map=recon_net, sigma_map=None, initial_sigma=1.0)
    aux_net = GenericMLP(inp_dim=z_dim, out_dim=num_classes, n_hidden=32, n_layers=1)
    modules = nn.ModuleDict({
        "recog_net": recog_net,
        "recon_net": recon_net,
        "pxz_net": pxz_net,
        "qzx_net": qzx_net,
        "aux_net": aux_net,
    }).to(device)
    return modules


def synthesize_batch(device, batch_size=64, num_timepoints=228, input_dim=12, num_classes=7):
    """Build a batch shaped like Human Activity but with random data."""
    obs = torch.randn(batch_size, num_timepoints, input_dim, device=device)
    msk = torch.ones(batch_size, num_timepoints, input_dim, device=device, dtype=torch.long)
    tid = torch.arange(num_timepoints, device=device).unsqueeze(0).expand(batch_size, -1).long()
    tps = tid.float() / num_timepoints
    aux_obs = torch.randint(0, num_classes, (batch_size, num_timepoints), device=device).long()
    return {
        "inp_obs": obs, "inp_msk": msk, "inp_tid": tid, "inp_tps": tps,
        "evd_obs": obs, "evd_msk": msk, "evd_tid": tid,
        "aux_obs": aux_obs, "aux_tid": tid,
    }


def run_cell(N: int, kind: str, adjoint: str, device: str, dtype: torch.dtype,
             kl_subsample_M: Optional[int] = None,
             recon_subsample_M: Optional[int] = None) -> dict:
    """Execute one (N, kind, adjoint) cell. Return dict with peak alloc + status."""
    torch.manual_seed(42)
    if kind == "geometric_euler":
        solver = geometric_euler
    else:
        solver = kind  # string for dispatch
    use_checkpoint = (adjoint == "checkpoint")
    if use_checkpoint:
        # checkpointed runs use plain solver (the integrator itself is unchanged;
        # we wrap the *full forward pass* in torch.utils.checkpoint).
        adjoint_for_encoder = "autograd"
    else:
        adjoint_for_encoder = adjoint

    modules = make_modules(
        device=device, solver=solver, adjoint=adjoint_for_encoder,
        kl_subsample_M=kl_subsample_M,
    )

    # Synthetic batch with K_obs fixed (matching real Zeng HAR), independent
    # of the integrator step count N. The previous default ``num_timepoints=N+1``
    # tied the encoder's mTAN attention size to N, making it O(N^2) and
    # masking the integrator's memory contribution.
    batch = synthesize_batch(device=device, num_timepoints=UCI_HAR_K_OBS)
    parts = batch
    desired_t = torch.linspace(0.0, 0.99, N + 1, device=device)

    elbo = ELBO(reduction="mean")
    aux_loss_fn = PerTimePointCrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.Adam(modules.parameters(), lr=1e-3)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()

    # Slice the trajectory to UCI_HAR_K_OBS observation indices before the
    # decoder runs -- decoder + aux memory is O(K_obs) instead of O(N+1).
    obs_idx = torch.linspace(0, N, UCI_HAR_K_OBS, device=device).long().clamp_(0, N)

    def fwd():
        h = modules["recog_net"]((parts["inp_obs"], parts["inp_msk"], parts["inp_tps"]))
        qzx, pz = modules["qzx_net"](h, desired_t)
        zis = qzx.rsample((1,))
        zis_obs = zis[..., obs_idx, :]
        evd_obs, evd_msk, evd_tid = parts["evd_obs"], parts["evd_msk"], parts["evd_tid"]
        aux_obs, aux_tid = parts["aux_obs"], parts["aux_tid"]
        if recon_subsample_M is not None and recon_subsample_M < zis_obs.shape[-2]:
            zis_obs, evd_obs, evd_msk, evd_tid, aux_obs, aux_tid = _subsample_recon_targets(
                zis_obs, evd_obs, evd_msk, evd_tid, aux_obs, aux_tid, recon_subsample_M
            )
        pxz = modules["pxz_net"](zis_obs)
        aux = modules["aux_net"](zis_obs)
        elbo_val, _ = elbo(
            qzx, pz, pxz, evd_obs, evd_tid, evd_msk,
            {"kl0_weight": 1e-4, "klp_weight": 1e-4, "pxz_weight": 1.0},
        )
        a_val = aux_loss_fn(aux, aux_obs, aux_tid)
        return elbo_val + 10.0 * a_val

    try:
        modules.train()
        if use_checkpoint:
            # torch.utils.checkpoint requires a callable returning a Tensor.
            # We wrap fwd, but fwd has no Tensor inputs -- use a dummy input
            # marked use_reentrant=False so it works without a leaf input.
            loss = checkpoint(lambda _x: fwd(), torch.zeros(1, device=device, requires_grad=True), use_reentrant=False)
        else:
            loss = fwd()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        peak_alloc = torch.cuda.max_memory_allocated()
        result = {
            "N": N, "kind": kind, "adjoint": adjoint,
            "peak_alloc_bytes": int(peak_alloc),
            "peak_alloc_mib": peak_alloc / 1024 / 1024,
            "loss": float(loss.item()),
            "status": "ok",
        }
    except torch.cuda.OutOfMemoryError as e:
        peak_alloc = torch.cuda.max_memory_allocated()
        result = {
            "N": N, "kind": kind, "adjoint": adjoint,
            "peak_alloc_bytes": int(peak_alloc),
            "peak_alloc_mib": peak_alloc / 1024 / 1024,
            "loss": None,
            "status": "oom",
        }
    except Exception as e:
        result = {
            "N": N, "kind": kind, "adjoint": adjoint,
            "peak_alloc_bytes": None, "peak_alloc_mib": None, "loss": None,
            "status": f"error: {type(e).__name__}: {e}",
        }

    # cleanup
    del modules, batch, parts, desired_t, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default="results/memory_sweep.csv")
    parser.add_argument("--n-values", type=int, nargs="*", default=N_VALUES)
    parser.add_argument(
        "--kl-subsample-m", type=int, default=None,
        help="If set, estimate the path-KL by Monte-Carlo sampling M random "
             "timepoints from the integrator's grid instead of summing over all N. "
             "Drops the path-KL's autograd-tape memory from O(N) to O(M).")
    parser.add_argument(
        "--recon-subsample-m", type=int, default=None,
        help="If set, subsample M random timepoints from the trajectory before the "
             "decoder + aux_net are applied, MC-estimating the per-timepoint "
             "reconstruction and classification losses. Drops decoder+aux activation "
             "memory from O(N) to O(M).")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    for N in args.n_values:
        for kind, adj in CELLS:
            print(f"\n=== N={N}, solver={kind}, adjoint={adj} ===", flush=True)
            r = run_cell(N, kind, adj, device, dtype,
                         kl_subsample_M=args.kl_subsample_m,
                         recon_subsample_M=args.recon_subsample_m)
            print(
                f"  status={r['status']:<10} peak={r['peak_alloc_mib'] if r['peak_alloc_mib'] is not None else 'n/a':>10} MiB"
                f"  loss={r['loss']:.4f}" if r["loss"] is not None else
                f"  status={r['status']:<10} peak={r['peak_alloc_mib'] if r['peak_alloc_mib'] is not None else 'n/a'} MiB",
                flush=True,
            )
            rows.append(r)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nWrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
