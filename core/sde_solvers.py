"""Implementation of the geometric Euler-Maryuama SDE solver.

Plus an additive dispatch (``dispatch_solver``) that routes to the EES /
RKMK(Heun) integrators added for the EES paper. The default path through
``geometric_euler`` is unchanged.
"""

from typing import Callable, List, Optional

import torch
from torch import Tensor
from utils.misc import vec_to_matrix

def geometric_euler(z0, drift, cov, dt, basis):
    noise = torch.randn(z0.shape[:-2]+drift.shape, device=z0.device)
    if cov.numel() == 1:
        noise = noise * cov
    else:
        noise = torch.einsum('btij, ...btj -> ...bti', cov, noise)

    noise = torch.einsum('...td, t -> ...td', noise, dt.sqrt())
    drift = torch.einsum('...td, t -> ...td', drift, dt)
    omegas = drift + noise
    omegas = vec_to_matrix(omegas, basis)

    Qs = torch.matrix_exp(omegas.contiguous())
    zi = [z0]
    for t_idx in range(len(dt)):
        Qt = Qs[...,t_idx,:,:]
        Qtz = torch.einsum('...ij, ...j -> ...i', Qt, zi[-1])
        zi.append(Qtz)
    zi = torch.stack(zi, dim=-2)
    return zi

# Per-step NN-eval counts -- single source of truth for the compute-parity sweep.
NEVALS_PER_STEP = {
    "geometric_euler": 1,
    "cfees25": 3,
}


def dispatch_solver(
    z0: Tensor,
    K_fn: Callable[[Tensor], Tensor],
    K_params: List[Tensor],
    sigma: Tensor,
    t: Tensor,
    basis: Tensor,
    *,
    kind: str = "geometric_euler",
    adjoint: str = "autograd",
    base_seed: int = 0,
) -> Tensor:
    """Route the integration to the requested (kind, adjoint) combination.

    Args:
        z0: ``[..., dim]`` initial state on S^{dim - 1}.
        K_fn: callable ``t -> [..., L, group_dim]`` drift function (state-
            independent, captures the encoder output).
        K_params: list of Tensors that ``K_fn`` closes over -- needed by the
            reversible adjoint to route grads back through ``K_fn``. Pass an
            empty list for constant ``K`` (e.g. Brownian-motion prior).
        sigma: scalar Tensor (constant diffusion).
        t: ``[N+1]`` save grid times.
        basis: ``[group_dim, dim, dim]`` antisymmetric so(n) basis.
        kind: one of ``"geometric_euler"``, ``"rkmk_heun"``, ``"cfees25"``.
        adjoint: one of ``"autograd"``, ``"reversible"``. Reversible is only
            valid for ``"cfees25"`` -- it is the headline contribution.
        base_seed: int seed for deterministic per-step noise. Used by
            ``"cfees25"`` so the reversible adjoint can re-derive identical
            noise in the backward pass.

    Returns:
        ``[..., len(t), dim]`` trajectory including ``z0``.
    """
    if kind == "geometric_euler":
        # Match the legacy signature: precompute Kt at save times, call directly.
        Kt = K_fn(t[:-1])
        dt = torch.diff(t)
        return geometric_euler(z0, Kt, sigma, dt, basis)

    if kind == "cfees25":
        if adjoint == "reversible":
            from core.cfees_solver import cfees25_step
            from core.reversible_adjoint import reversible_cfees_integrate
            return reversible_cfees_integrate(
                z0, K_fn, K_params, sigma, t, basis, base_seed,
                step_fn=cfees25_step,
            )
        if adjoint == "autograd":
            # Pure-forward CFEES with full autograd tape -- intended for unit
            # tests and linear-stability checks only. Production training uses
            # the reversible adjoint.
            from core.cfees_solver import cfees25_forward_only
            return cfees25_forward_only(z0, K_fn, sigma, t, basis, base_seed)
        raise ValueError(f"Unknown adjoint mode: {adjoint!r}")

    raise ValueError(f"Unknown solver kind: {kind!r}")