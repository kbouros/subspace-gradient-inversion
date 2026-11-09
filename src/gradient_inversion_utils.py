#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Adapted from facebookresearch/cocktail_party_attack.
# Modified for the subspace-gradient-inversion project; see
# THIRD_PARTY_NOTICES.md for provenance and license exceptions.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

attack_name_dict = {"cp": "Cocktail Party Attack", "gm": "Gradient Matching Attack"}


class GradientMetric:
    def __init__(self, indices="all") -> None:
        self.indices = indices

    def determine_indices(self, input_gradient):
        indices = self.indices
        if isinstance(indices, list):
            pass
        elif indices == "all":
            indices = torch.arange(len(input_gradient))
        else:
            raise ValueError()
        return indices


class CosineDistance(GradientMetric):
    def __init__(self, indices="all") -> None:
        super().__init__(indices=indices)

    def __call__(self, trial_gradients, input_gradients):
        with torch.no_grad():
            indices = self.determine_indices(input_gradients)
            filtered_trial_gradients = [trial_gradients[i] for i in indices]
            filtered_input_gradients = [input_gradients[i] for i in indices]

        costs = sum(
            (x * y).sum() for x, y in zip(filtered_input_gradients, filtered_trial_gradients)
        )

        trial_norm = sum(x.pow(2).sum() for x in filtered_trial_gradients).sqrt()
        input_norm = sum(y.pow(2).sum() for y in filtered_input_gradients).sqrt()
        costs = 1 - (costs / trial_norm / input_norm)
        return costs


def estimate_hull(
    GW: torch.Tensor,
    gb: torch.Tensor | None,
    batch_size: int,
    subspace_type: str = "affine",
    s_thresh: float = 1e-8,
    alpha_w: float = 2.0,
):
    """
    Estimate an affine or linear hull from GW and gb.

    Affine case:
      - Uses ratio points U = GW / gb.
      - Masks small denominators.
      - Fits a centered weighted subspace.
      - Uses k = batch_size - 1.

    Linear case:
      - Uses U = GW directly.
      - Fits an uncentered subspace through the origin.
      - Returns mu = 0.
      - Uses k = batch_size.

    Args:
        GW:
            Tensor of shape (m, d).
        gb:
            Tensor with m elements in affine mode. Unused in linear mode
            and may be None.
        batch_size:
            Determines the requested hull dimension.
        subspace_type:
            Either "affine" or "linear".
        s_thresh:
            Minimum absolute denominator retained in affine mode.
        alpha_w:
            Exponent used for denominator-based reliability weights.

    Returns:
        mu:
            Tensor of shape (d,).
        D:
            Tensor of shape (d, k), where k is batch_size - 1 for an
            affine hull and batch_size for a linear hull.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    if subspace_type not in {"affine", "linear"}:
        raise ValueError(
            f"subspace_type must be 'affine' or 'linear', got {subspace_type}."
        )

    if s_thresh < 0:
        raise ValueError("s_thresh must be non-negative.")

    affine = (subspace_type == "affine") # Boolean that checks if we are in affine or linear mode

    if gb is None and affine:
            raise ValueError("gb is required when subspace_type='affine'.")

    return_dtype = GW.dtype # original dtype to convert back to
    k = batch_size - 1 if affine else batch_size

    # Convert to double precision for numerical stability and accuracy
    GW = GW.to(torch.float64)
    if affine:
        gb = gb.to(torch.float64).reshape(-1)

    if affine:            
        # Mask unstable ratios.
        mask = gb.abs() > s_thresh
        num_active = int(mask.sum().item())

        if num_active < batch_size:
            raise ValueError(
                f"Not enough active rows: need at least {batch_size}, "
                f"got {num_active}."
            )

        GW = GW[mask]      # (m,d)
        gb  = gb[mask]      # (m,)
    else:
        if GW.shape[0] < batch_size:
            raise ValueError(
                f"Not enough rows: need at least {batch_size}, "
                f"got {GW.shape[0]}."
            )

    m, d = GW.shape
    if d < k:
        raise ValueError(f"Hull dimension k={k} exceeds ambient dimension d={d}.")

    # Ratio points (equal to GW for linear)
    U = GW / gb.unsqueeze(1) if affine else GW

    # Denominator based reliability weight (applied only to the affine ratio points).
    if affine and alpha_w != 0.0:
        w = gb.abs().pow(alpha_w)
    else:
        w = torch.ones(m, device=U.device, dtype=U.dtype)

    # Weighted PCA
    w_sum = w.sum()
    if not torch.isfinite(w_sum):
        raise FloatingPointError("Weight sum is non-finite.")
        
    w = w / w_sum
    if not torch.isfinite(w).all():
        raise FloatingPointError("Weights contain non-finite values.")

    if affine:
        mu = (w.unsqueeze(1) * U).sum(dim=0)
        X = U - mu
    else:
        mu = torch.zeros(d, device=U.device, dtype=U.dtype)
        X = U

    # A single-point affine hull has no direction vectors.
    if k == 0:
        D = torch.empty(d, 0, device=U.device, dtype=U.dtype)
        return mu.to(return_dtype), D.to(return_dtype)
    
    # weighted centered data
    sqrt_w = torch.sqrt(w).unsqueeze(1)
    Xw = X * sqrt_w

    # SVD of weighted centered data
    _, _, Vh = torch.linalg.svd(Xw, full_matrices=False)

    D = Vh[:k].T.contiguous()

    return mu.to(return_dtype), D.to(return_dtype)
