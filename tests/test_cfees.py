"""Unit tests for the CFEES(2, 5; 1/10) integrator and its O(1)-memory adjoint.

Run with::

    .venv/bin/python -m pytest tests/test_cfees.py -v

or as a script::

    .venv/bin/python tests/test_cfees.py
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cfees_solver import (
    cfees25_forward_only,
    cfees25_step,
    step_noise,
    _A,
    _B,
    _C,
)
from core.reversible_adjoint import reversible_cfees_integrate
from utils.misc import vec_to_matrix


def _so2_basis(device, dtype=torch.float64):
    """Single antisymmetric basis element for so(2). group_dim = 1, dim = 2."""
    basis = torch.zeros(1, 2, 2, device=device, dtype=dtype)
    basis[0, 1, 0] = 1.0
    basis[0, 0, 1] = -1.0
    return basis


def _so3_basis(device, dtype=torch.float64):
    """Antisymmetric basis of so(3) -- group_dim = 3, dim = 3 (Zeng convention)."""
    group_dim = 3
    idx = torch.tril_indices(row=3, col=3, offset=-1)
    basis = torch.zeros(group_dim, 3, 3, device=device, dtype=dtype)
    basis[:, idx[0], idx[1]] = torch.eye(group_dim, dtype=dtype, device=device)
    basis = basis - basis.permute(0, 2, 1)
    return basis


class TestCFEESCoefficients(unittest.TestCase):
    """The Williamson 2N coefficients should sum to 1 for a state-INdependent K.

    The aggregate Lie-algebra rotation accumulated across the three stages must
    equal ``rho`` (where ``rho = lambda * dt``). With the Zeng formulation
    (multiplicative apply_increment via matrix_exp), this translates to
    ``y_3 = exp(rho) * y_0`` exactly for constant K. The check below is
    algebraic, on the (A, B, C) tuple alone.
    """

    def test_aggregate_weight_equals_one(self):
        # B_0 + B_1 (A_0 + 1) + B_2 (A_1 (A_0 + 1) + 1)  should equal 1
        agg = (
            _B[0]
            + _B[1] * (_A[0] + 1)
            + _B[2] * (_A[1] * (_A[0] + 1) + 1)
        )
        self.assertAlmostEqual(agg, 1.0, places=12)

    def test_C_matches_porting_reference(self):
        self.assertEqual(_C, (0.0, 1.0 / 3.0, 5.0 / 6.0))


class TestCFEESLinearODE(unittest.TestCase):
    """For state-INdependent constant K = lambda, CFEES on the matrix Lie group
    must give the *exact* solution ``y = exp(lambda * t) y_0`` (the recurrence
    sums to a single matrix_exp of the cumulative Lie-algebra increment).
    """

    def test_so2_constant_K_matches_exp(self):
        device = torch.device("cpu")
        dtype = torch.float64
        basis = _so2_basis(device, dtype)
        # Constant drift: K(t) = (lambda,) for all t.
        lam = 0.7

        def K_fn(t):
            return torch.full((t.shape[0], 1), lam, device=device, dtype=dtype)

        sigma = torch.tensor(0.0, device=device, dtype=dtype)  # deterministic
        y0 = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
        # One step from t=0 to t=dt with sigma=0 -> deterministic.
        for dt_val in (0.05, 0.1, 0.5, 1.0, 2.0):
            dt = torch.tensor(dt_val, device=device, dtype=dtype)
            t = torch.tensor([0.0, dt_val], device=device, dtype=dtype)
            y_traj = cfees25_forward_only(y0, K_fn, sigma, t, basis, base_seed=12345)
            y_final = y_traj[-1]
            # Analytic: y_final = exp(lambda * dt * basis[0]) y_0
            #         = (cos(lambda*dt), sin(lambda*dt)) since basis[0] is the
            #         standard so(2) generator [[0,-1],[1,0]].
            expected = torch.tensor(
                [
                    torch.cos(torch.tensor(lam * dt_val, dtype=dtype)),
                    torch.sin(torch.tensor(lam * dt_val, dtype=dtype)),
                ],
                dtype=dtype,
            )
            err = (y_final - expected).abs().max().item()
            # The recurrence is exact for state-INdependent constant K (the three
            # accumulated stage exponents sum to 1*rho); residual is float64 noise
            # from the matrix_exp + matmul chain, which scales with rho.
            self.assertLess(err, 1e-10, f"dt={dt_val}: error {err}")


class TestCFEESReversibility(unittest.TestCase):
    """AS-order 5: forward step from y_n to y_{n+1} then "reverse step" (same
    integrator with swapped t0/t1, same noise) should return to y_n with error
    O(|dt|^6) per step.
    """

    def test_round_trip_deterministic_order_six(self):
        """Deterministic case (sigma=0): AS-order 5 implies round-trip error
        scales as O(dt^6). Test by measuring the slope between dt=0.2 and dt=0.1
        (in the asymptotic regime, before float64 noise floor).
        """
        device = torch.device("cpu")
        dtype = torch.float64
        basis = _so3_basis(device, dtype)
        group_dim = basis.shape[0]

        def K_fn(t):
            base = torch.tensor([0.3, -0.5, 0.7], dtype=dtype, device=device)
            return torch.outer(t, torch.ones(group_dim, dtype=dtype, device=device)) * base

        sigma = torch.tensor(0.0, device=device, dtype=dtype)  # deterministic
        y0 = torch.tensor([0.1, 0.4, 0.91], dtype=dtype, device=device)
        y0 = y0 / y0.norm()

        errors = []
        for dt_val in (0.2, 0.1):
            t0 = torch.tensor(0.0, device=device, dtype=dtype)
            t1 = torch.tensor(dt_val, device=device, dtype=dtype)
            noise = step_noise(0, 0, (group_dim,), device, dtype)
            y_fwd = cfees25_step(y0, K_fn, sigma, t0, t1, basis, noise)
            y_back = cfees25_step(y_fwd, K_fn, sigma, t1, t0, basis, noise)
            err = (y_back - y0).abs().max().item()
            errors.append((dt_val, err))

        import math
        slope = math.log(errors[0][1] / max(errors[1][1], 1e-30)) / math.log(
            errors[0][0] / errors[1][0]
        )
        # Want slope ~ 6 (AS-order 5 -> O(dt^{AS+1}) round trip). Allow margin.
        self.assertGreater(slope, 5.5, f"observed slope {slope:.3f}, errors {errors}")

    def test_round_trip_stochastic_decreases(self):
        """Stochastic case (sigma>0): the recurrence is not symmetric for the
        Brownian increment so AS-order does not directly transfer; we just
        verify the round-trip error monotonically decreases with dt.
        """
        device = torch.device("cpu")
        dtype = torch.float64
        basis = _so3_basis(device, dtype)
        group_dim = basis.shape[0]

        def K_fn(t):
            base = torch.tensor([0.3, -0.5, 0.7], dtype=dtype, device=device)
            return torch.outer(t, torch.ones(group_dim, dtype=dtype, device=device)) * base

        sigma = torch.tensor(0.2, device=device, dtype=dtype)
        y0 = torch.tensor([0.1, 0.4, 0.91], dtype=dtype, device=device)
        y0 = y0 / y0.norm()

        prev_err = None
        for dt_val in (0.4, 0.2, 0.1, 0.05):
            t0 = torch.tensor(0.0, device=device, dtype=dtype)
            t1 = torch.tensor(dt_val, device=device, dtype=dtype)
            noise = step_noise(0, 0, (group_dim,), device, dtype)
            y_fwd = cfees25_step(y0, K_fn, sigma, t0, t1, basis, noise)
            y_back = cfees25_step(y_fwd, K_fn, sigma, t1, t0, basis, noise)
            err = (y_back - y0).abs().max().item()
            if prev_err is not None:
                self.assertLess(err, prev_err)
            prev_err = err


class TestCFEESPreservesSphereNorm(unittest.TestCase):
    """SDE on S^{n-1}: trajectory points should remain on the unit sphere."""

    def test_norm_preserved_so16(self):
        device = torch.device("cpu")
        dtype = torch.float64
        # Build so(16) basis as Zeng does.
        dim = 16
        group_dim = int(dim * (dim - 1) / 2)
        idx = torch.tril_indices(row=dim, col=dim, offset=-1)
        basis = torch.zeros(group_dim, dim, dim, device=device, dtype=dtype)
        basis[:, idx[0], idx[1]] = torch.eye(group_dim, dtype=dtype, device=device)
        basis = basis - basis.permute(0, 2, 1)

        torch.manual_seed(0)
        y0 = torch.randn(4, dim, dtype=dtype, device=device)
        y0 = y0 / y0.norm(dim=-1, keepdim=True)

        def K_fn(t):
            # Random but smooth drift coefficients in so(16). Time-varying.
            base = torch.randn(4, group_dim, dtype=dtype, device=device, generator=torch.Generator(device=device).manual_seed(7))
            # Make drift depend on t by a sinusoid
            return base.unsqueeze(1) * torch.sin(t).unsqueeze(0).unsqueeze(-1)

        sigma = torch.tensor(0.5, device=device, dtype=dtype)
        N = 50
        t = torch.linspace(0.0, 1.0, N + 1, dtype=dtype, device=device)
        y_traj = cfees25_forward_only(y0, K_fn, sigma, t, basis, base_seed=7)
        norms = y_traj.norm(dim=-1)
        max_dev = (norms - 1.0).abs().max().item()
        self.assertLess(max_dev, 1e-10, f"max norm deviation {max_dev}")


class TestReversibleAdjointCorrectness(unittest.TestCase):
    """Backward through the reversible adjoint should match the gradient
    obtained via full-autograd CFEES forward (within tolerance set by the
    reverse-step reconstruction error).
    """

    def test_gradient_matches_full_autograd(self):
        device = torch.device("cpu")
        dtype = torch.float64
        # Small system for exact reproducibility.
        dim = 4
        group_dim = int(dim * (dim - 1) / 2)
        idx = torch.tril_indices(row=dim, col=dim, offset=-1)
        basis = torch.zeros(group_dim, dim, dim, device=device, dtype=dtype)
        basis[:, idx[0], idx[1]] = torch.eye(group_dim, dtype=dtype, device=device)
        basis = basis - basis.permute(0, 2, 1)

        torch.manual_seed(13)
        h = torch.randn(2, group_dim, dtype=dtype, requires_grad=True)
        # K_fn closes over h. To compare against full autograd, we use the same h.
        # Time-dependence: K_fn(t) returns h * f(t) (broadcast).
        def K_fn(t):
            f = (1.0 + 0.3 * torch.sin(t))  # time profile
            return h.unsqueeze(1) * f.unsqueeze(0).unsqueeze(-1)

        sigma = torch.tensor(0.1, dtype=dtype, requires_grad=True)
        y0 = torch.randn(2, dim, dtype=dtype, requires_grad=True)
        y0 = y0 / y0.norm(dim=-1, keepdim=True)

        N = 8
        t = torch.linspace(0.0, 1.0, N + 1, dtype=dtype)
        base_seed = 999

        # Path A: full-autograd CFEES forward.
        h_a = h.detach().clone().requires_grad_(True)
        sigma_a = sigma.detach().clone().requires_grad_(True)
        y0_a = y0.detach().clone().requires_grad_(True)

        def K_fn_a(t):
            f = (1.0 + 0.3 * torch.sin(t))
            return h_a.unsqueeze(1) * f.unsqueeze(0).unsqueeze(-1)

        y_a = cfees25_forward_only(y0_a, K_fn_a, sigma_a, t, basis, base_seed)
        # NB: ||y||^2 is identically 1 on the sphere -> use a non-trivial linear
        # functional of the trajectory components.
        loss_a = (y_a[..., 0] + 0.5 * y_a[..., 1]).sum()
        loss_a.backward()

        # Path B: reversible adjoint.
        h_b = h.detach().clone().requires_grad_(True)
        sigma_b = sigma.detach().clone().requires_grad_(True)
        y0_b = y0.detach().clone().requires_grad_(True)

        def K_fn_b(t):
            f = (1.0 + 0.3 * torch.sin(t))
            return h_b.unsqueeze(1) * f.unsqueeze(0).unsqueeze(-1)

        y_b = reversible_cfees_integrate(
            y0_b, K_fn_b, [h_b], sigma_b, t, basis, base_seed
        )
        loss_b = (y_b[..., 0] + 0.5 * y_b[..., 1]).sum()
        loss_b.backward()

        # Compare gradients. The reversible adjoint reconstructs intermediate
        # states with O(dt^6) per-step error (deterministic) plus a stochastic
        # contribution; tolerate absolute errors ~1e-6 and relative errors
        # against the larger of (this grad, the *full grad's* magnitude).
        for name, ga, gb in [("y0", y0_a.grad, y0_b.grad), ("h", h_a.grad, h_b.grad), ("sigma", sigma_a.grad, sigma_b.grad)]:
            err = (ga - gb).abs().max().item()
            # Use the *combined* magnitude as the denominator so gradients near
            # zero in one path don't blow up the relative error.
            scale = max(ga.abs().max().item(), gb.abs().max().item(), 1e-6)
            rel = err / scale
            self.assertLess(
                rel, 1e-2,
                f"{name}: rel-err {rel:.3e}, abs-err {err:.3e}, "
                f"|ga|max {ga.abs().max().item():.3e}, |gb|max {gb.abs().max().item():.3e}"
            )


if __name__ == "__main__":
    unittest.main()
