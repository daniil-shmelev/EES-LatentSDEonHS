"""*Integrator-only* memory sweep: measures the peak VRAM of one
forward + backward through the SDE integrator alone, with a trivial linear
loss on the trajectory. This isolates the integrator's autograd-tape cost
from the rest of the Latent SDE training pipeline (which has its own
O(N * dim^2) tape from the Girsanov closed-form path-KL formula).

Mirrors the RNA torus memory-scaling setup used elsewhere in the paper, where
CFEES + reversible adjoint stays flat across n_steps while full-tape solvers
grow.

Output: ``results/memory_sweep_integrator.csv``.

Usage::

    .venv/bin/python scripts/memory_sweep_integrator.py
"""

import argparse
import csv
import gc
import os
import sys
from typing import List

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cfees_solver import cfees25_forward_only
from core.reversible_adjoint import reversible_cfees_integrate
from core.sde_solvers import geometric_euler
from core.models import Chebyshev


N_VALUES = [50, 200, 800, 2000, 5000]
CELLS = [
    ("geometric_euler", "autograd"),
    ("cfees25", "reversible"),
]


def make_so_n_basis(dim, device, dtype):
    group_dim = int(dim * (dim - 1) / 2)
    idx = torch.tril_indices(row=dim, col=dim, offset=-1)
    basis = torch.zeros(group_dim, dim, dim, device=device, dtype=dtype)
    basis[:, idx[0], idx[1]] = torch.eye(group_dim, dtype=dtype, device=device)
    basis = basis - basis.permute(0, 2, 1)
    return basis, group_dim


def run_cell(N: int, kind: str, adjoint: str, device: str,
             dtype: torch.dtype, batch: int = 64, dim: int = 16, n_deg: int = 4) -> dict:
    torch.manual_seed(42)
    basis, group_dim = make_so_n_basis(dim, device, dtype)

    h_dim = 128
    # Build a Chebyshev module so K_fn has parameters; matches the Latent SDE setup.
    time_fn = Chebyshev(h_dim, n_deg, group_dim, time_min=0.0, time_max=2.0).to(device)
    h = torch.randn(batch, h_dim, device=device, dtype=dtype, requires_grad=True)
    sigma = torch.tensor(0.1, device=device, dtype=dtype, requires_grad=True)

    z0 = torch.randn(batch, dim, device=device, dtype=dtype)
    z0 = z0 / z0.norm(dim=-1, keepdim=True)
    z0 = z0.requires_grad_(True)

    t = torch.linspace(0.0, 1.0, N + 1, device=device, dtype=dtype)
    base_seed = 12345

    def K_fn(arg_t):
        return time_fn(h, arg_t)

    K_params: List[torch.Tensor] = [h, time_fn.map.weight, time_fn.map.bias]

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        if kind == "geometric_euler":
            # Use Zeng's geometric_euler with precomputed Kt (matches legacy path).
            Kt = K_fn(t[:-1])
            dt = torch.diff(t)
            zi = geometric_euler(z0, Kt, sigma, dt, basis)
        elif kind == "cfees25" and adjoint == "reversible":
            zi = reversible_cfees_integrate(z0, K_fn, K_params, sigma, t, basis, base_seed)
        elif kind == "cfees25" and adjoint == "autograd":
            zi = cfees25_forward_only(z0, K_fn, sigma, t, basis, base_seed)
        else:
            raise ValueError(f"unknown {kind}/{adjoint}")

        # Trivial linear loss -- non-trivial gradient w.r.t. all params.
        loss = (zi[..., 0] + 0.5 * zi[..., 1]).sum()
        loss.backward()
        peak = torch.cuda.max_memory_allocated()
        result = {
            "N": N, "kind": kind, "adjoint": adjoint,
            "peak_alloc_bytes": int(peak), "peak_alloc_mib": peak / 1024 / 1024,
            "loss": float(loss.item()), "status": "ok",
        }
    except torch.cuda.OutOfMemoryError:
        peak = torch.cuda.max_memory_allocated()
        result = {
            "N": N, "kind": kind, "adjoint": adjoint,
            "peak_alloc_bytes": int(peak), "peak_alloc_mib": peak / 1024 / 1024,
            "loss": None, "status": "oom",
        }
    except Exception as e:
        result = {
            "N": N, "kind": kind, "adjoint": adjoint,
            "peak_alloc_bytes": None, "peak_alloc_mib": None,
            "loss": None, "status": f"error: {type(e).__name__}: {e}",
        }

    del basis, time_fn, h, sigma, z0, t
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default="results/memory_sweep_integrator.csv")
    parser.add_argument("--n-values", type=int, nargs="*", default=N_VALUES)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--dim", type=int, default=16)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = []
    for N in args.n_values:
        for kind, adj in CELLS:
            print(f"\n=== N={N}, solver={kind}, adjoint={adj} ===", flush=True)
            r = run_cell(N, kind, adj, device, dtype, batch=args.batch, dim=args.dim)
            mib_str = f"{r['peak_alloc_mib']:.2f}" if r['peak_alloc_mib'] is not None else "n/a"
            print(f"  status={r['status']:<10} peak={mib_str} MiB", flush=True)
            rows.append(r)
            with open(args.out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                for row in rows:
                    w.writerow(row)
    print(f"\nWrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
