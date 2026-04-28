"""O(1)-memory reversible adjoint for the CFEES(2, 5; 1/10) integrator.

This is the *only* differentiable path for CFEES in this codebase: pure-forward
CFEES (``core.cfees_solver.cfees25_forward_only``) is intended only for unit
tests and linear-stability checks; using it for training would defeat the
central memory-saving contribution.

Mechanism (mirrors the diffrax ``ReversibleAdjoint`` pattern):

* ``forward(ctx, ...)`` runs CFEES under ``torch.no_grad()`` and saves only
  the terminal state plus the per-step noise base-seed (the per-step noise is
  re-derived from the seed in the backward pass, so it is not stored).
* ``backward(ctx, grad_zi)`` reconstructs the trajectory in reverse via the
  algorithmic time-symmetry of CFEES (call the same step with swapped
  ``(t0, t1)``); then for each step, runs *one* forward step under autograd
  to compute the local gradient contribution, accumulates ``grad_sigma`` and
  ``grad_K_params`` across steps, and propagates ``grad_z`` to the previous
  state.

To get gradients back to the parameters that ``K_fn`` closes over (encoder
output ``h``, Chebyshev linear weight/bias, etc.), the Function takes those
parameters as explicit Tensor inputs (``*K_params``). PyTorch then routes
the returned per-parameter grads back into the outer autograd graph through
its standard ``Function.apply`` machinery.

The reverse-step error is ``O(dt^6)`` per step (CFEES has antisymmetric
order 5), so trajectory reconstruction is faithful enough for stable
training over the trajectory lengths exercised in the memory sweep.
"""

from typing import Callable, List

import torch
from torch import Tensor

from core.cfees_solver import cfees25_step, step_noise


class ReversibleCFEESFunction(torch.autograd.Function):
    """``torch.autograd.Function`` for O(1)-memory CFEES backprop.

    Inputs (positional, after ``ctx``):

    * ``z0`` -- ``[..., dim]`` initial state (differentiable).
    * ``sigma`` -- scalar Tensor (differentiable iff its ``requires_grad``).
    * ``t`` -- ``[N+1]`` save grid (NOT differentiable).
    * ``basis`` -- ``[group_dim, dim, dim]`` (NOT differentiable).
    * ``base_seed`` -- Python int (NOT a tensor).
    * ``K_fn`` -- callable ``t -> drift`` (NOT a tensor; closes over the
      following ``*K_params``).
    * ``*K_params`` -- list of Tensors that ``K_fn`` depends on; gradients
      flow back into these via the returned grads.
    """

    @staticmethod
    def forward(
        ctx,
        z0: Tensor,
        sigma: Tensor,
        t: Tensor,
        basis: Tensor,
        base_seed: int,
        K_fn: Callable[[Tensor], Tensor],
        step_fn: Callable,
        *K_params: Tensor,
    ) -> Tensor:
        with torch.no_grad():
            *batch_shape, dim = z0.shape
            group_dim = basis.shape[0]
            noise_shape = (*batch_shape, group_dim)
            traj = [z0]
            z = z0
            for n in range(len(t) - 1):
                noise = step_noise(base_seed, n, noise_shape, z0.device, z0.dtype)
                z = step_fn(z, K_fn, sigma, t[n], t[n + 1], basis, noise)
                traj.append(z)
            zi = torch.stack(traj, dim=-2)

        ctx.save_for_backward(z0, sigma, z, *K_params)
        ctx.t = t
        ctx.basis = basis
        ctx.base_seed = base_seed
        ctx.K_fn = K_fn
        ctx.step_fn = step_fn
        ctx.n_K_params = len(K_params)
        return zi

    @staticmethod
    def backward(ctx, grad_zi: Tensor):
        saved = ctx.saved_tensors
        z0_unused = saved[0]
        sigma = saved[1]
        z_T = saved[2]
        K_params = list(saved[3:])
        t = ctx.t
        basis = ctx.basis
        base_seed = ctx.base_seed
        K_fn = ctx.K_fn
        step_fn = ctx.step_fn
        N = len(t) - 1

        *batch_shape, n_t_plus_1, dim = grad_zi.shape
        group_dim = basis.shape[0]
        noise_shape = (*batch_shape, group_dim)

        z = z_T
        grad_z = grad_zi[..., N, :].clone()
        grad_sigma = torch.zeros_like(sigma) if sigma.requires_grad else None
        grad_K_params = [
            torch.zeros_like(p) if p.requires_grad else None for p in K_params
        ]

        for n in range(N - 1, -1, -1):
            t_n = t[n]
            t_n_plus_1 = t[n + 1]

            # Draw the per-step noise once and reuse for both the reverse step
            # (state reconstruction) and the forward step (gradient
            # recomputation).
            noise = step_noise(base_seed, n, noise_shape, z.device, z.dtype)

            # Reconstruct y_n by stepping CFEES with (t1, t0) swapped (algorithmic
            # time-symmetry; AS-order 5 makes the per-step error O(|dt|^6)).
            with torch.no_grad():
                y_n = step_fn(z, K_fn, sigma, t_n_plus_1, t_n, basis, noise)

            # Local one-step autograd to compute grad contributions.
            with torch.enable_grad():
                y_n_leaf = y_n.detach().requires_grad_(True)
                if sigma.requires_grad:
                    sigma_leaf = sigma.detach().requires_grad_(True)
                else:
                    sigma_leaf = sigma
                z_recon = step_fn(
                    y_n_leaf, K_fn, sigma_leaf, t_n, t_n_plus_1, basis, noise
                )
                grad_input_tensors: List[Tensor] = [y_n_leaf]
                if sigma.requires_grad:
                    grad_input_tensors.append(sigma_leaf)
                # K_params (the *original* tensors that K_fn closes over) need to
                # appear here so torch.autograd.grad can compute and return their
                # grads. K_fn references them through its closure; they sit in the
                # outer autograd graph as nn.Parameter leaves.
                params_with_grad = [p for p in K_params if p.requires_grad]
                grad_input_tensors.extend(params_with_grad)

                grads = torch.autograd.grad(
                    z_recon,
                    grad_input_tensors,
                    grad_outputs=grad_z,
                    retain_graph=False,
                    allow_unused=True,
                )

            idx = 0
            grad_y_n_local = grads[idx]
            idx += 1
            if grad_y_n_local is None:
                grad_y_n_local = torch.zeros_like(y_n)
            if sigma.requires_grad:
                if grads[idx] is not None:
                    grad_sigma = grad_sigma + grads[idx]
                idx += 1
            for k_idx, p in enumerate(K_params):
                if p.requires_grad:
                    g = grads[idx]
                    idx += 1
                    if g is not None:
                        grad_K_params[k_idx] = grad_K_params[k_idx] + g

            # Add the user-loss gradient at intermediate save point y_n.
            grad_z = grad_y_n_local + grad_zi[..., n, :]
            z = y_n

        # Return order must mirror ``forward`` arg order:
        #   z0, sigma, t, basis, base_seed, K_fn, step_fn, *K_params
        return (
            grad_z,
            grad_sigma,
            None,  # t
            None,  # basis
            None,  # base_seed
            None,  # K_fn
            None,  # step_fn
            *grad_K_params,
        )


def reversible_cfees_integrate(
    z0: Tensor,
    K_fn: Callable[[Tensor], Tensor],
    K_params: List[Tensor],
    sigma: Tensor,
    t: Tensor,
    basis: Tensor,
    base_seed: int,
    step_fn: Callable = cfees25_step,
) -> Tensor:
    """Run a CFEES scheme forward with the O(1)-memory reversible adjoint.

    Returns the trajectory ``[..., len(t), dim]`` including ``z0`` at index 0.
    Gradients backprop through the returned tensor with O(1) memory cost in the
    number of integrator steps.

    Args:
        z0: ``[..., dim]`` initial state on S^{dim - 1}.
        K_fn: callable ``t -> [..., L, group_dim]`` drift function.
        K_params: list of Tensors that ``K_fn`` closes over (the autograd path
            for these is the explicit input list of the Function; pass an empty
            list if ``K_fn`` is a constant function).
        sigma: scalar Tensor (constant diffusion magnitude).
        t: ``[N+1]`` save grid times.
        basis: ``[group_dim, dim, dim]`` antisymmetric so(n) basis.
        base_seed: int -- per-step noise seeds derive from this.
        step_fn: callable implementing one step of the chosen CFEES scheme;
            defaults to ``cfees25_step``. Pass ``cfees27_step`` for EES(2, 7).
    """
    return ReversibleCFEESFunction.apply(
        z0, sigma, t, basis, base_seed, K_fn, step_fn, *K_params
    )
