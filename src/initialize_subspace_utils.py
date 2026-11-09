#!/usr/bin/env python3
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np
import torch
from scipy.optimize import linprog
from dataset_utils import xshape_dict, ds_type_dict, ds_mean, ds_std

def _flatten_bounds(lo, hi, B_or_1, img_shape, device, dtype):
    lo_flat = lo.to(device=device, dtype=dtype).expand(B_or_1, *img_shape).reshape(B_or_1, -1)
    hi_flat = hi.to(device=device, dtype=dtype).expand(B_or_1, *img_shape).reshape(B_or_1, -1)
    return lo_flat, hi_flat

def find_one_feasible_a(mu, D, lo, hi, img_shape, tol=1e-6):
    """
    Finds one feasible a such that:

        lo - tol <= reshape(mu + D @ a) <= hi + tol

    Keeps torch outputs in the original dtype.
    SciPy internally uses float64, but returned tensor uses mu.dtype.
    """
    device, dtype = mu.device, mu.dtype
    d, k = D.shape
    assert mu.shape == (d,)
    assert d == math.prod(img_shape), f"d={d}, img_shape={img_shape} mismatch"

    # B=1 => k=0 (affine case): no LP variables; only reachable point is mu
    if k == 0:
        return torch.empty((0,), device=device, dtype=dtype)

    lo_flat, hi_flat = _flatten_bounds(lo, hi, 1, img_shape, device, dtype)
    lo_flat = lo_flat[0]
    hi_flat = hi_flat[0]

    A_ub = torch.cat([D, -D], dim=0).detach().cpu().numpy()
    b_ub = torch.cat([hi_flat - mu + tol, mu - lo_flat + tol], dim=0).detach().cpu().numpy()

    res = linprog(
        c=np.zeros(k),
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=[(None, None)] * k,
        method="highs",
    )

    if not res.success:
        raise ValueError(
            "No feasible point exists: cannot find a such that "
            "lo <= mu + D @ a <= hi."
        )

    return torch.as_tensor(res.x, device=device, dtype=dtype)


def farthest_point_subset(A_pool, B):
    """
    Greedy farthest-point sampling in coefficient space.

    A_pool: [M, k]
    returns: [B, k]
    """
    M, k = A_pool.shape
    B = min(B, M)

    selected = []

    first = torch.randint(0, M, (), device=A_pool.device).item()
    selected.append(first)

    min_dist2 = torch.full((M,), float("inf"), device=A_pool.device, dtype=A_pool.dtype)

    for _ in range(1, B):
        last = A_pool[selected[-1]][None, :]
        dist2 = ((A_pool - last) ** 2).sum(dim=1)

        min_dist2 = torch.minimum(min_dist2, dist2)
        min_dist2[selected] = -1.0

        nxt = torch.argmax(min_dist2).item()
        selected.append(nxt)

    idx = torch.tensor(selected, device=A_pool.device, dtype=torch.long)
    return A_pool[idx]


def batched_hit_and_run_from_anchor(
    mu,
    D,
    a0,
    lo,
    hi,
    B,
    img_shape,
    chains_mult=4,
    steps=128,
    burn_in=32,
    eps=1e-10,
    tol=1e-6,
):
    """
    Batched hit-and-run sampler starting from one feasible anchor.

    Returns:
        A: [B, k]
        x: [B,*img_shape]
    """
    device, dtype = mu.device, mu.dtype
    d, k = D.shape
    assert d == math.prod(img_shape), f"d={d}, img_shape={img_shape} mismatch"

    if k == 0:
        A = torch.empty((B, 0), device=device, dtype=dtype)
        x = mu.reshape(1, *img_shape).expand(B, *img_shape).clone()

        if not ((x >= lo - tol).all() and (x <= hi + tol).all()):
            raise RuntimeError("k=0, so only mu is reachable, but mu violates bounds.")

        return A, x

    M = chains_mult * B

    lo_flat, hi_flat = _flatten_bounds(lo, hi, 1, img_shape, device, dtype)
    lo_flat = lo_flat.expand(M, d)
    hi_flat = hi_flat.expand(M, d)

    A = a0[None, :].expand(M, k).clone()

    total_steps = burn_in + steps

    for _ in range(total_steps):
        V = torch.randn(M, k, device=device, dtype=dtype)
        V = V / (V.norm(dim=1, keepdim=True) + eps)

        X = mu[None, :] + A @ D.T
        Dir = V @ D.T

        t_low = torch.full((M, d), -float("inf"), device=device, dtype=dtype)
        t_high = torch.full((M, d), float("inf"), device=device, dtype=dtype)

        pos = Dir > eps
        neg = Dir < -eps

        t_low[pos] = (lo_flat[pos] - X[pos]) / Dir[pos]
        t_high[pos] = (hi_flat[pos] - X[pos]) / Dir[pos]

        t_low[neg] = (hi_flat[neg] - X[neg]) / Dir[neg]
        t_high[neg] = (lo_flat[neg] - X[neg]) / Dir[neg]

        t_min = t_low.max(dim=1).values
        t_max = t_high.min(dim=1).values

        valid = t_max > t_min

        t = torch.zeros(M, device=device, dtype=dtype)
        r = torch.rand(M, device=device, dtype=dtype)
        t[valid] = t_min[valid] + r[valid] * (t_max[valid] - t_min[valid])

        A = A + t[:, None] * V

    A = farthest_point_subset(A, B)

    X = mu[None, :] + A @ D.T
    x = X.reshape(B, *img_shape)

    if not ((x >= lo - tol).all() and (x <= hi + tol).all()):
        below = (x < lo - tol).float().mean()
        above = (x > hi + tol).float().mean()
        raise RuntimeError(
            f"Sampled points violated bounds. "
            f"frac below={below.item():.6f}, frac above={above.item():.6f}"
        )

    return A, x


class HARCoefficientSampler:
    """
    Hit-and-Run sampler for generating one feasible starting batch at a time.
    A linear program is solved once during initialization to find a feasible anchor.

    Args:
        mu: Flattened base vector with shape [d] or [1,d].
        D: Basis/direction matrix with shape [d, k].
        B: Number of feasible samples to return per run.
        ds: Dataset key used to look up image shape and normalization stats.
        ds_normalize_bool: Whether to use dataset normalization bounds.
        chains_mult: Number of parallel chains per requested sample.
        steps: Number of hit-and-run sampling steps after burn-in.
        burn_in: Number of initial hit-and-run steps before collecting samples.
        eps: Small value used to avoid division by zero.
        tol: Numerical tolerance for feasibility checks.
    """
    def __init__(
        self,
        mu,
        D,
        B,
        ds,
        ds_normalize_bool=True,
        chains_mult=4,
        steps=128,
        burn_in=32,
        eps=1e-10,
        tol=1e-6,
    ):
        if D.ndim != 2:
            raise ValueError(
                f"D must have shape [d, k], but received {tuple(D.shape)}."
            )

        d, k = D.shape

        if mu.shape == (1, d):
            mu = mu.squeeze(0) # ensure mu is [d]

        if mu.shape != (d,):
            raise ValueError(
                f"mu must have shape ({d},) or (1, {d}), "
                f"but received {tuple(mu.shape)}."
            )

        if B <= 0:
            raise ValueError(f"B must be positive, but received {B}.")

        if chains_mult <= 0:
            raise ValueError(
                f"chains_mult must be positive, but received {chains_mult}."
            )

        if steps < 0:
            raise ValueError(f"steps must be nonnegative, but received {steps}.")

        if burn_in < 0:
            raise ValueError(
                f"burn_in must be nonnegative, but received {burn_in}."
            )

        self.mu = mu
        self.D = D
        self.B = B
        self.ds = ds

        self.chains_mult = chains_mult
        self.steps = steps
        self.burn_in = burn_in
        self.eps = eps
        self.tol = tol

        self.device = mu.device
        self.dtype = mu.dtype
        self.d = d
        self.k = k

        self.img_shape = xshape_dict[ds]

        expected_d = math.prod(self.img_shape)
        if d != expected_d:
            raise ValueError(
                f"D has d={d}, but dataset {ds!r} has image shape "
                f"{self.img_shape}, containing {expected_d} elements."
            )

        self.lo, self.hi = self._build_bounds(ds_normalize_bool)

        # This LP is solved only once.
        self.anchor = find_one_feasible_a(
            mu=self.mu,
            D=self.D,
            lo=self.lo,
            hi=self.hi,
            img_shape=self.img_shape,
            tol=self.tol,
        )

    def _build_bounds(self, ds_normalize_bool):
        if ds_normalize_bool:
            mean = torch.tensor(ds_mean[self.ds], device=self.device, dtype=self.dtype)[None, :, None, None]
            std = torch.tensor(ds_std[self.ds], device=self.device, dtype=self.dtype)[None, :, None, None]
        else:
            mean = torch.zeros(1, device=self.device, dtype=self.dtype)
            std = torch.ones(1, device=self.device, dtype=self.dtype)

        lo = -mean / std
        hi = (1 - mean) / std

        return lo, hi

    def sample(self, return_X=False):
        """
        Generate one batch of diverse feasible starting points.

        Args:
            return_X:
                If True, return both coefficient vectors and reconstructed
                images.

        Returns:
            If return_X=False:
                A with shape [B, k].

            If return_X=True:
                A with shape [B, k] and x with shape
                [B, *img_shape].
        """
        A, x = batched_hit_and_run_from_anchor(
            mu=self.mu,
            D=self.D,
            a0=self.anchor,
            lo=self.lo,
            hi=self.hi,
            B=self.B,
            img_shape=self.img_shape,
            chains_mult=self.chains_mult,
            steps=self.steps,
            burn_in=self.burn_in,
            eps=self.eps,
            tol=self.tol,
        )

        if return_X:
            return A, x

        return A

    def __call__(self, return_X=False):
        return self.sample(return_X=return_X)
