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

import numpy as np
import torch
import torch.nn as nn
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from utils import normalize
from scipy.optimize import linear_sum_assignment


# lpips requires inputs in the range [-1,1]
def normalize_lpips(x):
    return (x - 0.5) * 2


# Function that creates the cost matrix for permutations (less is better).
@torch.no_grad()
def score_matrix(x_hat, x_true, compute_lpips, method, data_range=1.0):
    device = x_hat.device

    assert (x_hat.ndim == 4 or x_hat.ndim == 2), f"x_hat must be [B, C, H, W] or [B, d], got {x_hat.shape}"
    assert (x_true.ndim == 4 or x_true.ndim == 2), f"x_true must be [B, C, H, W] or [B, d], got {x_true.shape}"
    assert x_hat.shape[1:] == x_true.shape[1:], f"Input shape mismatch: {x_hat.shape[1:]} vs {x_true.shape[1:]}"

    B_pred = x_hat.shape[0]
    B_true = x_true.shape[0]

    if method == "l2":
        # Flatten for distance computations
        x_hat_flat = x_hat.reshape(B_pred, -1)
        x_true_flat = x_true.reshape(B_true, -1)

        # Create inverse image
        x_hat_inv_flat = (1.0 - x_hat).reshape(B_pred, -1)

        cost_pos = torch.cdist(x_hat_flat, x_true_flat)
        cost_inv = torch.cdist(x_hat_inv_flat, x_true_flat)

        # Rectangular cost matrix (inverse aware)
        return torch.minimum(cost_pos, cost_inv) # (B_pred, B_true)

    elif method == "lpips":
        if x_hat.shape[1] != 3:
            raise ValueError("LPIPS requires 3-channel RGB images")

        cost = torch.empty(B_pred, B_true, device=device)

        x_hat_norm_pos = normalize_lpips(x_hat)
        x_hat_norm_inv = normalize_lpips(1.0 - x_hat)
        x_true_norm = normalize_lpips(x_true)

        for i in range(B_pred):
            # Compute row scores (each predicted point with all true points)
            xh_pos = x_hat_norm_pos[i:i+1].expand(B_true, -1, -1, -1) # (B_true, C, H, W)
            xh_inv = x_hat_norm_inv[i:i+1].expand(B_true, -1, -1, -1) # (B_true, C, H, W)

            scores_pos = compute_lpips(xh_pos, x_true_norm).view(-1)
            scores_inv = compute_lpips(xh_inv, x_true_norm).view(-1)

            cost[i] = torch.minimum(scores_pos, scores_inv)

        return cost

    elif method == "ssim":
        # SSIM is similarity: higher is better.
        # Return 1 - SSIM so this is a cost matrix: lower is better.
        ssim_metric = StructuralSimilarityIndexMeasure(
            data_range=data_range,
            reduction="none",
        ).to(device)

        cost = torch.empty(B_pred, B_true, device=device)

        for i in range(B_pred):
            xh_pos = x_hat[i:i+1].expand(B_true, -1, -1, -1)
            xh_inv = (1.0 - x_hat[i:i+1]).expand(B_true, -1, -1, -1)

            scores_pos = ssim_metric(xh_pos, x_true).view(-1)
            scores_inv = ssim_metric(xh_inv, x_true).view(-1)

            cost_pos = 1.0 - scores_pos
            cost_inv = 1.0 - scores_inv

            cost[i] = torch.minimum(cost_pos, cost_inv)

        return cost

    elif method == "psnr":
        # PSNR is similarity: higher is better.
        # Return -PSNR so this is a cost matrix: lower is better.
        psnr_metric = PeakSignalNoiseRatio(
            data_range=data_range,
            reduction="none",
            dim=(1, 2, 3),
        ).to(device)

        cost = torch.empty(B_pred, B_true, device=device)

        for i in range(B_pred):
            xh_pos = x_hat[i:i+1].expand(B_true, -1, -1, -1)
            xh_inv = (1.0 - x_hat[i:i+1]).expand(B_true, -1, -1, -1)

            scores_pos = psnr_metric(xh_pos, x_true).view(-1)
            scores_inv = psnr_metric(xh_inv, x_true).view(-1)

            cost_pos = -scores_pos
            cost_inv = -scores_inv

            cost[i] = torch.minimum(cost_pos, cost_inv)

        return cost

    elif method in ["cs", "cs_abs"]:
        # Cosine similarity (with or without sign-invariance via abs)
        # Higher similarity = better -> convert to cost via (1 - sim)
        # Ideally, "cs_abs" should be used only for CPA, where the sign is not preserved, and not for GMA
        x_hat_flat = x_hat.reshape(B_pred, -1)
        x_true_flat = x_true.reshape(B_true, -1)

        x_hat_unit = x_hat_flat / x_hat_flat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        x_true_unit = x_true_flat / x_true_flat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
 
        sim = torch.matmul(x_hat_unit, x_true_unit.T)  # (B_pred, B_true)
        
        if method == "cs_abs":
            sim = sim.abs()

        cost = 1.0 - sim
        return cost

    else:
        raise ValueError("Unknown similarity scoring method")


@torch.no_grad()
def match_perm_hungarian(x_hat, y_hat, x_true, compute_lpips, scoring_method=None):
    """
    Hungarian matching with support for B_pred != B_true.

    Conventions
    -----------
    If B_pred < B_true:
        - output length is B_true
        - outputs are in true-point order
        - unmatched true slots are padded with NaN

    If B_pred > B_true:
        - output length is B_pred
        - first B_true entries are the matched predictions aligned to x_true
        - remaining entries are the unmatched predictions

    If B_pred == B_true:
        - output length is B_true
        - outputs are fully reordered to align with x_true

    Parameters
    ----------
    x_hat  : (B_pred, ...) predicted points
    y_hat  : (B_pred, ...) predicted labels aligned with x_hat or None
    x_true : (B_true, ...) target points
    scoring_method : "lpips" / "ssim" / "psnr" / "cs" / "cs_abs" / "l2"

    Returns
    -------
    x_hat_out : reordered/padded predictions, shape depends on convention above
    y_hat_out : reordered/padded labels, same leading length as x_hat_out or None
    """
    device = x_hat.device
    B_pred = x_hat.shape[0]
    B_true = x_true.shape[0]

    if scoring_method is None:
        scoring_method = "lpips" if x_hat.shape[1] == 3 else "ssim"

    # Rectangular cost matrix
    cost = score_matrix(x_hat, x_true, compute_lpips, scoring_method)  # (B_pred, B_true)

    row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
    row_ind = torch.as_tensor(row_ind, dtype=torch.long, device=device)
    col_ind = torch.as_tensor(col_ind, dtype=torch.long, device=device)

    matched_pred = set(row_ind.tolist())
    unmatched_pred = [i for i in range(B_pred) if i not in matched_pred]

    y_hat_out = None # will stay None if y_hat was None

    # Case 1: B_pred <= B_true
    # Return length B_true in true-point order, padded with NaNs for unmatched true slots
    if B_pred <= B_true:
        out_len = B_true

        # Place matched predictions into true order
        x_hat_out = torch.full((out_len, *x_hat.shape[1:]), float("nan"), dtype=x_hat.dtype, device=x_hat.device)
        x_hat_out[col_ind] = x_hat[row_ind]

        if y_hat is not None:
            y_hat_out = torch.full((out_len, *y_hat.shape[1:]), -1, dtype=y_hat.dtype, device=y_hat.device)
            y_hat_out[col_ind] = y_hat[row_ind]

    # Case 2: B_pred > B_true
    # Return length B_pred: first matched block aligned to x_true order, then unmatched predictions
    else:
        sort_idx = torch.argsort(col_ind)
        matched_pred_order = row_ind[sort_idx]
        unmatched_pred_tensor = torch.as_tensor(unmatched_pred, dtype=torch.long, device=device)

        full_order = torch.cat([matched_pred_order, unmatched_pred_tensor], dim=0)

        x_hat_out = x_hat[full_order]
        if y_hat is not None:
            y_hat_out = y_hat[full_order]

    return x_hat_out, y_hat_out


class Eval:
    def __init__(self, S, fix_order_method=None, fix_sign_method=None, ds="cifar10"):
        self.S = S
        self.device = S.device
        self.compute_lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", reduction="none").to(
            self.device
        )
        self.eps = 1e-20
        self.fix_order_method = fix_order_method
        self.fix_sign_method = fix_sign_method
        self.ds = ds

    def fix_order(self, S_hat):
        if self.fix_order_method is None:
            return S_hat

        S_hat_reordered, _ = match_perm_hungarian(S_hat, None, self.S, self.compute_lpips, self.fix_order_method)

        return S_hat_reordered

    def fix_sign(self, S_hat):
        B = self.S.shape[0]
        B_hat = S_hat.shape[0]

        if self.fix_sign_method is None:
            return S_hat

        elif self.fix_sign_method == "best":
            N = min(B, B_hat)

            S_flat = self.S[:N].reshape(N, -1)
            S_hat_flat = S_hat[:N].reshape(N, -1)

            mse = torch.nanmean((S_flat - S_hat_flat) ** 2, dim=-1)
            mse_inv = torch.nanmean((S_flat - (1 - S_hat_flat)) ** 2, dim=-1)
            
            flip_mask = (mse_inv < mse).float().view(N, 1, 1, 1)

            S_hat_fixed = S_hat.clone()
            S_hat_fixed[:N] = (
                S_hat[:N] * (1 - flip_mask)
                + (1 - S_hat[:N]) * flip_mask
            )

            return S_hat_fixed

        else:
            raise ValueError("Unknown method: " + str(self.fix_sign_method))

class ImageEval(Eval):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.S = normalize(self.S, method="ds", ds=self.ds)
        self.compute_lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", reduction="none").to(
            self.device
        )
        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0, reduction="none", dim=(1, 2, 3)).to(
            self.device
        )
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0, reduction="none").to(
            self.device
        )

    def __call__(self, S_hat):
        S_hat = self.fix_order(S_hat)
        S_hat = self.fix_sign(S_hat)

        # Compare only up to the available number of ground-truth images
        batch_size = min(S_hat.shape[0], self.S.shape[0])
        x_pred = S_hat[:batch_size]
        x_true = self.S[:batch_size]

        # Mask to keep only valid rows
        valid_mask = (
            torch.isfinite(x_pred).flatten(1).all(dim=1)
            & torch.isfinite(x_true).flatten(1).all(dim=1)
        )

        # Initialize batch metrics
        psnr_batch = torch.full((batch_size,), float("nan"), device=self.device)
        ssim_batch = torch.full((batch_size,), float("nan"), device=self.device)
        lpips_batch = torch.full((batch_size,), float("nan"), device=self.device)

        if valid_mask.any():
            # Compute metrics on valid rows
            x_pred_valid = x_pred[valid_mask]
            x_true_valid = x_true[valid_mask]

            psnr_batch[valid_mask] = self.psnr_metric(x_pred_valid, x_true_valid)
            ssim_batch[valid_mask] = self.ssim_metric(x_pred_valid, x_true_valid)

            # Compute lpips only on RGB images (C==3)
            if x_pred_valid.shape[1] == 3:
                lpips_batch[valid_mask] = self.compute_lpips(
                    normalize_lpips(x_pred_valid),
                    normalize_lpips(x_true_valid),
                ).view(-1)

        metrics_batch = {
            "psnr": psnr_batch.cpu().tolist(),
            "ssim": ssim_batch.cpu().tolist(),
            "lpips": lpips_batch.cpu().tolist(),
        }

        # Ignore NaN entries
        metrics_avg = {
            "psnr": torch.nanmean(psnr_batch).item(), 
            "ssim": torch.nanmean(ssim_batch).item(),
            "lpips": torch.nanmean(lpips_batch).item(),
        }

        return metrics_avg, metrics_batch, S_hat


class EmbEval(Eval):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cs = nn.CosineSimilarity(dim=1, eps=self.eps)

    def __call__(self, S_hat, post_process=None):
        # Add a valid post_process action if you want the metric w.r.t. the final reconstruction as well.
        # Currently, this is only meaningful for CPA with post_process="abs" to evaluate the final
        # reconstructed embedding as it will be before being passed to fi, i.e. S_hat.abs().
        # Note that the reorder and the retuned S_hat are still not based on the post_process action.
        if post_process is not None and post_process not in ["abs"]:
            raise ValueError("Invalid post process method")

        S_hat = self.fix_order(S_hat)
        S_hat = self.fix_sign(S_hat)

        # Compare only up to the available number of ground-truth images
        batch_size = min(S_hat.shape[0], self.S.shape[0])
        x_pred = S_hat[:batch_size]
        x_true = self.S[:batch_size]

        # Mask to keep only valid rows
        valid_mask = (
            torch.isfinite(x_pred).flatten(1).all(dim=1)
            & torch.isfinite(x_true).flatten(1).all(dim=1)
        )

        # Initialize batch metrics
        cs_batch = torch.full((batch_size,), float("nan"), device=self.device)
        cs_batch_final = torch.full((batch_size,), float("nan"), device=self.device)

        if valid_mask.any():
            # Compute metrics on valid rows
            x_pred_valid = x_pred[valid_mask]
            x_true_valid = x_true[valid_mask]

            cs = self.cs(x_pred_valid, x_true_valid)
            # Ignore sign when the matching criterion was sign-invariant.
            if self.fix_order_method == "cs_abs":
                cs = cs.abs()

            cs_batch[valid_mask] = cs

            if post_process == "abs":
                # Computes cs(S, S_hat.abs()). This is the metric for the final rec_emb passed to fi.
                # It can have a better score than cs_abs when rec_emb is not strictly 
                # non-negative or non-positive, i.e. when nv prior > 0.
                cs_final = self.cs(x_pred_valid.abs(), x_true_valid)
                cs_batch_final[valid_mask] = cs_final

        metrics_batch = {
            "cs": cs_batch.cpu().tolist(),
        }

        # Ignore NaN entries
        metrics_avg = {
            "cs": torch.nanmean(cs_batch).item(), 
        }

        if post_process is not None:
            metrics_batch["cs_final"] = cs_batch_final.cpu().tolist()
            metrics_avg["cs_final"] = torch.nanmean(cs_batch_final).item()
            
        return metrics_avg, metrics_batch, S_hat


def get_eval(inp, emb, model_type, ds_type, ds, attack, sub_inp=False, fi=False):
    if attack == "gm":
        if ds_type == "image":
            if model_type == "fc" or not sub_inp: 
                # Default fix_order_method is lpips for RGB images. If this becomes computationally heavy, switch to ssim.
                # Note that ssim may not perform as well for ImageNet reconstructions.
                fix_order_method = "lpips" if inp.shape[1] == 3 else "ssim"
                eval = ImageEval(
                    inp,
                    ds=ds,
                    fix_order_method=fix_order_method
                )
            elif fi is False:  # model_type is "conv" and sub_inp is used (hard or soft subspace on embeddings)
                eval = EmbEval(emb, fix_order_method="cs")
            else:  # fi for hard subspace
                eval = ImageEval(inp, fix_order_method=None, ds=ds)
        else:
            eval = EmbEval(inp, fix_order_method="cs")
    else:
        if ds_type == "image":
            if model_type == "fc":
                # Default fix_order_method is lpips for RGB images. If this becomes the bottleneck, switch to ssim.
                fix_order_method = "lpips" if inp.shape[1] == 3 else "ssim"
                eval = ImageEval(inp, fix_order_method=fix_order_method, fix_sign_method="best", ds=ds)
            elif fi is False:  # model_type is "conv"
                eval = EmbEval(emb, fix_order_method="cs_abs", fix_sign_method=None)
            else:  # fi
                eval = ImageEval(inp, fix_order_method=None, fix_sign_method=None, ds=ds)
        else:  # Text
            eval = EmbEval(inp, fix_order_method="cs_abs", fix_sign_method=None)

    return eval
