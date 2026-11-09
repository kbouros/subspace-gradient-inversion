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
import torch.nn.functional as F
from utils import (
    get_opt,
    get_sch,
    get_device,
    write_pickle,
    read_pickle,
    normalize,
    get_attack_exp_path,
)
from gradient_inversion_utils import CosineDistance, estimate_hull
from initialize_subspace_utils import HARCoefficientSampler
from dataset_utils import xshape_dict, ds_type_dict, ds_mean, ds_std, nclasses_dict
from abc import abstractmethod
from os.path import exists


class GradientInversionAttack:
    def __init__(self, model, grads, labels, args, batch_id):

        self.args = args
        self.model = model
        self.model.train()
        self.grads = grads
        self.labels = labels
        self.use_labels = args.use_labels
        self.device = get_device()
        self.n_iter = args.n_iter
        self.n_comp = args.batch_size_attack
        self.sigma = args.sigma * args.C / args.batch_size
        self.eps = torch.tensor(1e-20, device=self.device)
        self.ds = args.ds
        self.hard_sub = args.hard_sub

        # Hyperparameters (GM and then CPA)
        self.sub_input = args.sub_input
        self.sub_intermediate = args.sub_intermediate
        self.subspace_type = args.subspace_type
        self.tv = args.tv
        self.l1 = args.l1
        self.nv = args.nv
        self.ne = args.ne
        self.T = args.T
        self.decor = args.decor
        self.set_inp_type()
        self.mean = torch.tensor(ds_mean[self.ds], device=self.device)[None, :, None, None]
        self.std = torch.tensor(ds_std[self.ds], device=self.device)[None, :, None, None]
        self.batch_id = batch_id
        self.start_iter = 0

    def total_variation(self, x):
        dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
        dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        return dx + dy

    def make_dict(self, keys, vals):
        return {k: v.cpu().detach().item() for k, v in zip(keys, vals)}

    @abstractmethod
    def set_inp_type(self):
        pass


class CocktailPartyAttack(GradientInversionAttack):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        exp_path = get_attack_exp_path(self.args)
        if self.n_comp == self.args.batch_size: 
            self.w_pkl_file = f"{exp_path}/w_{self.n_comp}_{self.batch_id}.pkl"
        else:
            self.w_pkl_file = f"{exp_path}/w_{self.args.batch_size}_{self.n_comp}_{self.batch_id}.pkl"
        self.X = self.grads[self.model.attack_index]
        self.X_zc, self.X_mu = self.zero_center(self.X)

        if exists(self.w_pkl_file):
            w_data = read_pickle(self.w_pkl_file)
            self.X_w, self.W_w = w_data["X_w"].to(self.device), w_data["W_w"].to(
                self.device
            )
            print(f"loaded w data from {self.w_pkl_file}")
        else:
            self.X_w, self.W_w = self.whiten(self.X_zc)
            write_pickle(
                {"X_w": self.X_w.detach().cpu(), "W_w": self.W_w.detach().cpu()},
                self.w_pkl_file,
            )

        self.W_hat = torch.empty(
            [self.n_comp, self.n_comp],
            dtype=torch.float,
            requires_grad=True,
            device=self.device,
        )
        torch.nn.init.eye_(self.W_hat)
        param_list = [self.W_hat]

        self.opt = get_opt(param_list, self.args.opt, lr=self.args.lr)
        self.sch = get_sch(self.args.sch, self.opt, epochs=self.args.n_iter)
        self.a = 1

        # Keep track of best values of algorithm
        self.best_loss_dict = {"loss": float("inf")}
        self.best_W_hat = self.W_hat.detach().clone()

    def set_inp_type(self):
        if ds_type_dict[self.ds] == "image" and self.model.model_type == "fc":
            self.inp_type = "image"
            self.inp_shape = [self.n_comp] + xshape_dict[self.ds]
        else:
            self.inp_type = "emb"
            self.inp_shape = [self.n_comp, -1]

    def zero_center(self, x):
        x_mu = x.mean(dim=-1, keepdims=True)
        return x - x_mu, x_mu

    def whiten(self, x):
        cov = torch.matmul(x, x.T) / (x.shape[1] - 1)
        eig_vals, eig_vecs = torch.linalg.eig(cov)
        topk_indices = torch.topk(eig_vals.float().abs(), self.n_comp)[1]

        U = eig_vecs.float()
        lamb = eig_vals.float()[topk_indices].abs()
        lamb_inv_sqrt = torch.diag(1 / (torch.sqrt(lamb) + self.eps)).float()
        W = torch.matmul(lamb_inv_sqrt, U.T[topk_indices]).float()
        x_w = torch.matmul(W, x)
        return x_w, W

    def get_attack_state(self):
        state_dict = {}
        state_dict["best_loss_dict"] = self.best_loss_dict
        state_dict["W_hat"] = self.W_hat
        state_dict["best_W_hat"] = self.best_W_hat
        state_dict["opt"] = self.opt.state_dict()
        if self.sch is not None:
            state_dict["sch"] = self.sch.state_dict()
        return state_dict

    def set_attack_state(self, state_dict):
        self.best_loss_dict = state_dict["best_loss_dict"]
        self.W_hat.data = state_dict["W_hat"].data
        self.best_W_hat.data = state_dict["best_W_hat"].data
        self.opt.load_state_dict(state_dict["opt"])
        if self.sch is not None:
            self.sch.load_state_dict(state_dict["sch"])

    def step(self):
        loss_ne = loss_decor = loss_nv = loss_tv = loss_l1 = torch.tensor(
            0.0, device=self.device
        )

        self.opt.zero_grad()
        W_hat_norm = self.W_hat / (self.W_hat.norm(dim=-1, keepdim=True) + self.eps)

        # Neg Entropy Loss
        X_w = self.X_w
        S_hat = torch.matmul(W_hat_norm, X_w)


        if torch.isnan(S_hat).any():
            raise ValueError(f"S_hat has NaN")

        if self.ne > 0:
            loss_ne = -(
                (
                    (1 / self.a)
                    * torch.log(torch.cosh(self.a * S_hat) + self.eps).mean(dim=-1)
                )
                ** 2
            ).mean()

        # Undo centering, whitening
        S_hat = S_hat + torch.matmul(torch.matmul(W_hat_norm, self.W_w), self.X_mu)

        # Decorrelation Loss (decorrelate i-th row with j-th row, s.t. j>i)
        if self.decor > 0:
            cos_matrix = torch.matmul(W_hat_norm, W_hat_norm.T).abs()
            loss_decor = (torch.exp(cos_matrix * self.T) - 1).mean()

        # Prior Loss
        if self.tv > 0 and self.nv == 0:  # if nv > 0, tv is meant for the generator
            loss_tv = self.total_variation(S_hat.view(self.inp_shape))

        if self.nv > 0:
            loss_nv = torch.minimum(
                F.relu(-S_hat).norm(dim=-1), F.relu(S_hat).norm(dim=-1)
            ).mean()

        if self.l1 > 0:
            loss_l1 = torch.abs(S_hat).mean()

        loss = (
            loss_ne
            + (self.decor * loss_decor)
            + (self.tv * loss_tv)
            + (self.nv * loss_nv)
            + (self.l1 * loss_l1)
        )

        loss_dict = self.make_dict(
            [
                "loss",
                "loss_ne",
                "loss_decor",
                "loss_tv",
                "loss_nv",
                "loss_l1",
            ],
            [
                loss,
                loss_ne,
                loss_decor,
                loss_tv,
                loss_nv,
                loss_l1,
            ],
        )

        # Update best values
        if loss_dict["loss"] < self.best_loss_dict["loss"]:
            self.best_loss_dict = loss_dict.copy()
            self.best_W_hat = self.W_hat.detach().clone()

        loss.backward()
        self.opt.step()
        if self.sch:
            self.sch.step()

        return loss_dict

    def get_rec(self):
        with torch.no_grad():
            W_hat_norm = self.W_hat / (self.W_hat.norm(dim=-1, keepdim=True) + self.eps)
            S_hat = torch.matmul(W_hat_norm, self.X_w)
            S_hat = S_hat + torch.matmul(torch.matmul(W_hat_norm, self.W_w), self.X_mu)
            S_hat = S_hat.detach().view(self.inp_shape)
            if self.inp_type == "image":
                S_hat = normalize(S_hat, method="infer")
        return S_hat

    def get_best_rec(self):
        W_hat_norm = self.best_W_hat / (self.best_W_hat.norm(dim=-1, keepdim=True) + self.eps)
        S_hat = torch.matmul(W_hat_norm, self.X_w)
        S_hat = S_hat + torch.matmul(torch.matmul(W_hat_norm, self.W_w), self.X_mu)
        S_hat = S_hat.view(self.inp_shape)
        if self.inp_type == "image":
            S_hat = normalize(S_hat, method="infer")
        return S_hat, self.best_loss_dict


class GradientMatchingAttack(GradientInversionAttack):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Linear/affine hull from the first FC layer gradients.
        if self.hard_sub or self.sub_input > 0: 
            GW = self.grads[self.model.attack_index]
            gb = self.grads[self.model.attack_index + 1]
            try:
                mu, D = estimate_hull(GW, gb, self.n_comp, self.subspace_type)
                self.mu = mu.to(self.device).view(1, -1)
                self.D = D.to(self.device)
            except ValueError:
                if self.hard_sub:
                    # cannot continue without linear/affine hull
                    raise ValueError(f"Not enough active rows for {self.subspace_type} hull parameterization")
                # If soft subspace constraint, then continue without subspace regularizer for inputs
                self.mu = None
                self.D = None

        # Calculate linear/affine hulls of intermediate outputs of model.
        if self.sub_intermediate > 0:
            self.intermediate_hulls = self.get_intermediate_hulls()

        # Hit-and-Run sampler for hard subspace constraint when using fc model.
        if self.hard_sub and self.model.model_type == "fc":
            self.coef_sampler = HARCoefficientSampler(self.mu, self.D, self.n_comp, self.ds)

        self.restart = -1 # After next_restart() call, it will be increased to 0
        self.next_restart()

        self.criterion = torch.nn.CrossEntropyLoss()
        self.cosine_distance = CosineDistance()

        # Keep track of best values of algorithm across all iterations and restarts
        self.best_loss_dict = {"loss": float("inf")}
        self.best_Y_hat = self.Y_hat.detach().clone()
        if self.hard_sub:
            self.best_A_hat = self.A_hat.detach().clone()
        else:
            self.best_S_hat = self.S_hat.detach().clone()

    def next_restart(self):
        """Initialize restart-specific parameters, optimizer, and scheduler."""
        self.restart += 1
        self.params = []
        if self.use_labels:
            self.Y_hat = self.labels
        else:
            self.Y_hat = torch.nn.Parameter(
                torch.randn(
                    (self.n_comp, nclasses_dict[self.ds]),
                    requires_grad=True,
                    device=self.device,
                )
            )
            self.params.append(self.Y_hat)

        if self.hard_sub:
            # Optimize coefficients A
            if self.model.model_type == "conv":
                # Get random embeddings and project onto the hull
                X = torch.rand(self.n_comp, self.mu.shape[1],device=self.device)
                A_init = (X - self.mu) @ self.D
            else:
                # Use Hit-and-Run procedure
                A_init = self.coef_sampler()
                
            self.A_hat = torch.nn.Parameter(
                A_init.detach()
                      .clone()
                      .to(self.device)
            )
            self.params.append(self.A_hat)
        else:
            if self.sub_input>0 and self.model.model_type == "conv":
                # Optimize emb: use rand for initialization.
                self.S_hat = torch.nn.Parameter(
                    torch.rand(
                        self.n_comp,
                        self.grads[self.model.attack_index].shape[1], # emb dimension
                        requires_grad=True,
                        device=self.device,
                    )
                )
            else:
                # Optimize image directly 
                self.S_hat = torch.nn.Parameter(
                    torch.randn(
                        self.inp_shape,
                        requires_grad=True,
                        device=self.device,
                    )
                )
            self.params.append(self.S_hat)

        self.opt = get_opt(self.params, self.args.opt, lr=self.args.lr)
        self.sch = get_sch(self.args.sch, self.opt, epochs=self.args.n_iter)

    def get_intermediate_hulls(self):
        """
        Estimate the linear/affine hulls of the intermediate output batch points from next layer gradients.
        """
        # Indices where grads[idx] = GW and grads[idx+1] = gb
        grad_indices = self.model.intermediate_attack_indices

        # If using hard subspace constraint with conv model or testing soft subspace for embeddings, 
        # don't include "first" layer gradient (i.e. grads[self.model.attack_index]).
        if (self.hard_sub or self.sub_input>0) and self.model.model_type != "fc":
            grad_indices = grad_indices[1:]

        hulls = []
        for idx in grad_indices:
            GW = self.grads[idx]
            gb = self.grads[idx + 1]
            m, d = GW.shape
            m2 = gb.shape[0]
            assert m == m2

            if m < self.n_comp:
                # If usable rows are less than the batch size, then we can't retrieve the linear/affine hull.
                # In that case no additional reguralizer is applied for this layer.
                hulls.append(None)
                continue
            try:
                mu, D = estimate_hull(GW, gb, self.n_comp, self.subspace_type)
                hulls.append((mu, D))
            except ValueError:
                hulls.append(None)

        return hulls

    def subspace_regularizer(self, X, mu, D):
        """
        Penalize the distance of batch points X to the linear/affine hull defined by (mu, D).

        Args:
            X:  (B, C, H, W) or (B, d)
            mu: (d,) or (1, d)
            D:  (d, k) with (approximately) orthonormal columns. In our case k=B-1
        """

        # ensure mu is (1, d) for broadcasting
        if mu.dim() == 1:
            mu = mu.unsqueeze(0)   # (1, d)
        elif mu.dim() == 2 and mu.shape[0] == 1:
            pass
        else:
            raise ValueError("mu must have shape (d,) or (1, d)")

        # flatten to (B, d) if needed
        B = X.shape[0]
        X = X.reshape(B, -1)

        # center
        Xc = X - mu                # (B, d)

        # projection onto span(D)
        coeff = Xc @ D             # (B, k)
        proj = coeff @ D.T         # (B, d)

        # orthogonal residual
        resid = Xc - proj          # (B, d)

        val = (resid ** 2).sum(dim=1)   # (B,)

        return val.mean()

    def squared_cosine_decor(self, X):
        """
        X:  (B, C, H, W) or (B, d)

        Penalizes pairwise correlation between images/embeddings.
        """
        B = X.shape[0]

        if B < 2:
            return X.sum() * 0

        z = X
        z = z.reshape(B, -1)
        z = F.normalize(z, dim=1, eps=self.eps)

        cos = z @ z.T

        mask = ~torch.eye(B, device=X.device, dtype=torch.bool) # ignore correlation with self
        off_diag = cos[mask]

        return (off_diag ** 2).mean()

    def get_attack_state(self):
        state_dict = {}
        state_dict["restart"] = self.restart
        state_dict["best_loss_dict"] = self.best_loss_dict
        if self.hard_sub:
            state_dict["A_hat"] = self.A_hat
            state_dict["best_A_hat"] = self.best_A_hat
        else:
            state_dict["S_hat"] = self.S_hat
            state_dict["best_S_hat"] = self.best_S_hat

        state_dict["Y_hat"] = self.Y_hat
        state_dict["best_Y_hat"] = self.best_Y_hat
        state_dict["opt"] = self.opt.state_dict()
        if self.sch is not None:
            state_dict["sch"] = self.sch.state_dict()

        return state_dict

    def set_attack_state(self, state_dict):
        self.restart = state_dict["restart"]
        self.best_loss_dict = state_dict["best_loss_dict"]
        if self.hard_sub:
            self.A_hat.data = state_dict["A_hat"].data
            self.best_A_hat.data = state_dict["best_A_hat"].data
        else:
            self.S_hat.data = state_dict["S_hat"].data
            self.best_S_hat.data = state_dict["best_S_hat"].data

        self.Y_hat.data = state_dict["Y_hat"].data
        self.best_Y_hat.data = state_dict["best_Y_hat"].data
        self.opt.load_state_dict(state_dict["opt"])
        if self.sch is not None:
            self.sch.load_state_dict(state_dict["sch"])

    def set_inp_type(self):
        if ds_type_dict[self.ds] == "image" and (not (self.hard_sub or self.sub_input>0) or self.model.model_type == "fc"):
            self.inp_type = "image"
            self.inp_shape = [self.n_comp] + xshape_dict[self.ds]
        else:
            self.inp_type = "emb"
            self.inp_shape = [self.n_comp, -1]

    def step(self):
        loss_cd = loss_tv = torch.tensor(0.0, device=self.device) # cd loss and tv regularizer
        loss_sub = torch.tensor(0.0, device=self.device) # proposed regularizers
        loss_sub_inters = []
        loss_decor = loss_nv = loss_l1 = torch.tensor(0.0, device=self.device) # regularizers for fi 

        # Box Image
        with torch.no_grad():
            if not self.hard_sub:
                if self.sub_input > 0 and self.model.model_type == "conv":
                    # Project to 0+ orthant when testing soft subspace constraint for embeddings.
                    # Note that for the noisy scenario it is better not to project and use the nv regularizer instead.
                    # Set nv to a positive value to avoid projection.
                    if self.nv == 0:
                        self.S_hat.data = torch.max(self.S_hat, torch.zeros_like(self.S_hat))
                else:
                    # Image projection
                    self.S_hat.data = torch.max(
                        torch.min(self.S_hat, (1 - self.mean) / self.std),
                        -self.mean / self.std,
                    )

        self.opt.zero_grad()

        # Construct S_hat if we use hard subspace constraint
        if self.hard_sub:
            S_hat = self.mu + self.A_hat @ self.D.T
        else:
            S_hat = self.S_hat

        # Construct Y_hat to be passed to the criterion.
        if self.use_labels:
            Y_hat = self.Y_hat
        else:
            # Apply softmax if Y_hat are logits 
            Y_hat = F.softmax(self.Y_hat, dim=-1)

        # Gradient matching loss w.r.t. to the correct gradients. The implemented logic:
        # if model is conv and we reconstruct embeddings:
        #     Use FC part only
        # else:
        #    Use the whole model
        if self.model.model_type == "conv" and (self.hard_sub or self.sub_input>0): # reconstruct embedding / use FC part only
            if self.sub_intermediate > 0:
                pred, intermediate_outs = self.model.classify_embedding(S_hat, return_intermediate=True)
            else:
                pred = self.model.classify_embedding(S_hat)

            # Gradient matching wrt classifier parameters only
            loss_hat = self.criterion(pred, Y_hat)
            grad_hat = torch.autograd.grad(
                loss_hat, self.model.classifier.parameters(), create_graph=True
            )
            loss_cd = self.cosine_distance(grad_hat, self.grads[self.model.attack_index:])
        else:
            # Reconstruct images / use the whole model
            if self.sub_intermediate > 0:
                pred, intermediate_outs = self.model(S_hat, return_intermediate=True)
            else:
                pred = self.model(S_hat)

            loss_hat = self.criterion(pred, Y_hat)
            grad_hat = torch.autograd.grad(
                loss_hat, self.model.parameters(), create_graph=True
            )
            loss_cd = self.cosine_distance(grad_hat, self.grads)

        # Prior Losses
        if self.tv > 0 and not (self.model.model_type == "conv" and (self.hard_sub or self.sub_input>0)):
            # Reconstructing embeddings, tv is meant for the generator
            loss_tv = self.total_variation(S_hat.view(self.inp_shape))

        # Squared cosine decor
        if self.decor > 0:
            loss_decor = self.squared_cosine_decor(S_hat)

        # Soft subspace regularizer for input 
        if (
            self.sub_input > 0
            and self.mu is not None
            and self.D is not None
        ):
            loss_sub = self.subspace_regularizer(S_hat, self.mu, self.D)

        # Subspace regularizers for the intermediate outputs of fc layers (activations)
        # based on the linear/affine hulls estimated from the next layer gradients.
        # The same hyperparameter self.sub_intermediate will be used for all layers.
        if self.sub_intermediate > 0:
            if (self.hard_sub or self.sub_input>0) and self.model.model_type == "conv":
                intermediate_outs = intermediate_outs[1:]
            assert len(intermediate_outs) == len(self.intermediate_hulls)

            for out, hull in zip(intermediate_outs, self.intermediate_hulls):
                if hull is None:
                    loss_sub_inters.append(torch.tensor(0.0, device=self.device))
                    continue

                _, h_dim = out.shape
                assert h_dim == hull[0].shape[0]
                loss_sub_inters.append( self.subspace_regularizer(out, hull[0], hull[1]) )

        loss_sub_inters_total = sum(loss_sub_inters) if loss_sub_inters else torch.tensor(0.0, device=self.device)

        # Embedding priors : non-negativity, L1 norm
        if self.nv > 0:
            loss_nv = F.relu(-S_hat).norm(dim=-1).mean()

        if self.l1 > 0:
            loss_l1 = torch.abs(S_hat).mean()

        loss = (
            loss_cd
            + (self.tv * loss_tv)
            + (self.sub_input * loss_sub)
            + (self.sub_intermediate * loss_sub_inters_total)
            + (self.nv * loss_nv)
            + (self.l1 * loss_l1)
            + (self.decor * loss_decor)
        )

        subspace_key = self.subspace_type[:3] # aff if affine, lin if linear
        keys = ["loss", "loss_cd", "loss_tv", "loss_decor", f"loss_{subspace_key}"] + [
            f"loss_{subspace_key}_inter_{i}"
            for i in range(1, len(loss_sub_inters) + 1)
        ] + ["loss_nv", "loss_l1"]

        values = [loss, loss_cd, loss_tv, loss_decor, loss_sub] + loss_sub_inters + [loss_nv, loss_l1]

        loss_dict = self.make_dict(keys, values)

        # Update best values
        if loss_dict["loss"] < self.best_loss_dict["loss"]:
            self.best_loss_dict = loss_dict.copy()
            self.best_Y_hat = self.Y_hat.detach().clone()
            if self.hard_sub:
                self.best_A_hat = self.A_hat.detach().clone()
            else:
                self.best_S_hat = self.S_hat.detach().clone()

        loss.backward()
        self.opt.step()
        if self.sch:
            self.sch.step()

        return loss_dict

    def get_rec(self):
        with torch.no_grad():
            if self.hard_sub:
                S_hat = (self.mu + self.A_hat @ self.D.T).detach().view(self.inp_shape)
            else:
                S_hat = self.S_hat.detach()

            if self.inp_type == "image":
                S_hat = normalize(S_hat, method="ds", ds=self.ds)
            elif self.inp_type == "emb" and self.model.model_type == "conv": # nonnegative emb
                # final projection to nonnegative orthant
                S_hat = torch.clamp(S_hat, min=0)
        return S_hat

    def get_best_rec(self):
        if self.hard_sub:
            S_hat = (self.mu + self.best_A_hat @ self.D.T).view(self.inp_shape)
        else:
            S_hat = self.best_S_hat

        if self.inp_type == "image":
            S_hat = normalize(S_hat, method="ds", ds=self.ds)
        elif self.inp_type == "emb" and self.model.model_type == "conv": # nonnegative emb
            # final projection to nonnegative orthant
            S_hat = torch.clamp(S_hat, min=0)
        return S_hat, self.best_loss_dict


attack_dict = {"cp": CocktailPartyAttack, "gm": GradientMatchingAttack}


def get_gi(attack, model, grads, labels, args, batch_id, attack_log):
    gi = attack_dict[attack](model, grads, labels, args, batch_id)
    if attack_log.restore_required:
        gi.set_attack_state(attack_log.attack_state)
        attack_log.restore_required = False
        gi.start_iter = attack_log.iter + 1 # avoid doing additional iteration
    return gi
