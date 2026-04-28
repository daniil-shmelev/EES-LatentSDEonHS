"""CFEES(2, 5; 1/10) integrator on the sphere S^{n-1} = SO(n) / SO(n-1).

Williamson 2N low-storage chained-exponential recurrence parameterised by a
tableau (A, B, C). Coefficients ported verbatim from
``georax/_solver/cf_ees25.py`` (lines 14--18).

Same Lie-algebra-coordinate formulation as Zeng's ``geometric_euler``: each
sub-step's increment is a vector in so(n), applied to z in S^{n-1} via
``matrix_exp(vec_to_matrix(omega, basis)) @ z``. No separate sphere chart is
needed -- the basis + matrix_exp already gives geodesics on S^{n-1} via the
SO(n) action.

The step is *algorithmically* time-symmetric (forward and backward steps
share the same recurrence with swapped (t0, t1)) -- this is what enables the
O(1)-memory reversible adjoint in ``core/reversible_adjoint.py``.
"""

from typing import Callable

import torch
from torch import Tensor

from utils.misc import vec_to_matrix


# Williamson 2N coefficients for CFEES(2, 5; 1/10).
# Ported verbatim from georax/_solver/cf_ees25.py:14-18.
EES25_A = (-7.0 / 15.0, -35.0 / 32.0)
EES25_B = (1.0 / 3.0, 15.0 / 16.0, 2.0 / 5.0)
EES25_C = (0.0, 1.0 / 3.0, 5.0 / 6.0)

# Backwards-compatible aliases.
_A = EES25_A
_B = EES25_B
_C = EES25_C
NUM_STAGES = 3
NEVALS_PER_STEP = 3


def _apply_increment(z: Tensor, omega: Tensor, basis: Tensor) -> Tensor:
    """Apply ``exp(vec_to_matrix(omega, basis)) @ z``."""
    M = vec_to_matrix(omega, basis)
    Q = torch.matrix_exp(M.contiguous())
    return torch.einsum("...ij, ...j -> ...i", Q, z)


def _cfees_step_generic(
    z: Tensor,
    K_fn: Callable[[Tensor], Tensor],
    sigma: Tensor,
    t0: Tensor,
    t1: Tensor,
    basis: Tensor,
    noise: Tensor,
    A,
    B,
    C,
) -> Tensor:
    """Generic Williamson 2N CFEES step parameterised by a tableau (A, B, C).

    With ``len(B) == k`` stages, performs ``k`` drift evaluations, ``k``
    matrix exponentials, and ``k`` left actions on z. The Brownian increment
    over the full step is shared across stages (Stratonovich convention);
    ``sign(dt)`` flips the increment when this function is called with
    ``t0, t1`` swapped (the reverse pass of the reversible adjoint).
    """
    num_stages = len(B)
    dt = t1 - t0
    abs_sqrt_dt = dt.abs().sqrt()
    sign_dt = dt.sign()
    diffusion = sigma * sign_dt * abs_sqrt_dt * noise

    stage_times = torch.stack([t0 + c * dt for c in C])
    K_at_stages = K_fn(stage_times)
    drift_terms = K_at_stages * dt
    raws = drift_terms + diffusion.unsqueeze(-2)

    tmp = raws[..., 0, :]
    z = _apply_increment(z, B[0] * tmp, basis)
    for s in range(1, num_stages):
        tmp = A[s - 1] * tmp + raws[..., s, :]
        z = _apply_increment(z, B[s] * tmp, basis)
    return z


def cfees25_step(
    z: Tensor,
    K_fn: Callable[[Tensor], Tensor],
    sigma: Tensor,
    t0: Tensor,
    t1: Tensor,
    basis: Tensor,
    noise: Tensor,
) -> Tensor:
    """One CFEES(2, 5; 1/10) step from ``t0`` to ``t1`` (3 stages)."""
    return _cfees_step_generic(
        z, K_fn, sigma, t0, t1, basis, noise, EES25_A, EES25_B, EES25_C
    )


def step_noise(
    base_seed: int,
    step_idx: int,
    noise_shape,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Deterministic per-step Gaussian noise drawn from a step-derived seed.

    The reversible adjoint and the unit-test driver both use this so the
    forward and backward passes draw identical noise.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(int(base_seed) * (1 << 20) + int(step_idx))
    return torch.randn(*noise_shape, generator=gen, device=device, dtype=dtype)


def _forward_only_generic(z0, K_fn, sigma, t, basis, base_seed, step_fn):
    *batch_shape, dim = z0.shape
    group_dim = basis.shape[0]
    noise_shape = (*batch_shape, group_dim)

    z = z0
    trajectory = [z0]
    for n in range(len(t) - 1):
        noise = step_noise(base_seed, n, noise_shape, z0.device, z0.dtype)
        z = step_fn(z, K_fn, sigma, t[n], t[n + 1], basis, noise)
        trajectory.append(z)
    return torch.stack(trajectory, dim=-2)


def cfees25_forward_only(
    z0: Tensor,
    K_fn: Callable[[Tensor], Tensor],
    sigma: Tensor,
    t: Tensor,
    basis: Tensor,
    base_seed: int,
) -> Tensor:
    """Forward integration with full autograd tape -- O(N) memory.

    For unit-testing the integrator math and for diagnostic comparisons. The
    production training path uses ``ReversibleCFEESFunction`` from
    ``core.reversible_adjoint`` which gives O(1) memory.

    Returns:
        Trajectory ``[..., len(t), dim]`` including ``z0`` at index 0.
    """
    return _forward_only_generic(z0, K_fn, sigma, t, basis, base_seed, cfees25_step)
