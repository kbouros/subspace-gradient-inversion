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
import os
import math
import torch.nn as nn
import numpy as np
import random
from torch.backends import cudnn
from tqdm import tqdm
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
import sys
from dataset_utils import ds_mean, ds_std
import pickle

model_type_dict = {"vgg16": "conv"} # dictionary for fixed model types (practically only vgg16)
exp_path_base = "./exp"
submitit_logdir = "./exp/submititlogs"
slurm_partition = ""
wandb_entity = ""


def read_pickle(pkl_file):
    with open(pkl_file, "rb") as handle:
        data = pickle.load(handle)
    return data


def get_grads_file(ds, model, act, defense, batch_size, C, sigma):
    model_path = f"{model}/{act}" if model != "vgg16" else f"{model}" # act does not have meaning for vgg16
    exp_path = f"{exp_path_base}/{ds}/{model_path}/grads"
    if defense == "nodef":
        grads_file = f"{exp_path}/nodef/{batch_size}.pickle"
    else:
        grads_file = f"{exp_path}/dp/{batch_size}_C_{C}_sigma_{sigma}.pickle"
    return grads_file


def write_pickle(data, pkl_file):
    with open(pkl_file, "wb") as handle:
        data = pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def subsample(input_list, n_sample):
    return [item[:n_sample] for item in input_list]


def normalize(inp, method=None, ds="imagenet"):
    device = inp.device
    if method is None:
        pass
    elif method == "infer":
        orig_shape = inp.shape
        n = orig_shape[0]
        inp = inp.view([n, -1])
        inp = (inp - inp.min(dim=-1, keepdim=True)[0]) / (
            inp.max(dim=-1, keepdim=True)[0] - inp.min(dim=-1, keepdim=True)[0]
        )
        inp = inp.view(orig_shape)
    elif method == "ds":
        mean = torch.tensor(ds_mean[ds], device=device)[None, :, None, None]
        std = torch.tensor(ds_std[ds], device=device)[None, :, None, None]
        inp = torch.clamp((inp * std) + mean, 0, 1)
    else:
        raise ValueError(f"Unknown method {method}")
    return inp


class MyDataParallel(nn.DataParallel):
    """
    Allow nn.DataParallel to call model's attributes.
    """

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def setup(exp_path, log_file):
    set_seed(42)
    enable_gpu_benchmarking()
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    logger = logging.getLogger("my_logger") 
    logger.setLevel(logging.INFO)
    logger.propagate = False  

    # Clear potential previous handlers
    for h in logger.handlers[:]:
        h.close()
        logger.removeHandler(h)
        
    output_file_handler = logging.FileHandler(os.path.join(exp_path, log_file))
    stdout_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(output_file_handler)
    logger.addHandler(stdout_handler)
    
    return logger


def get_pbar(iter, disable=False, enu=False):
    t = len(iter)
    if enu:
        iter = enumerate(iter)
    if disable:
        pbar = tqdm(iter, total=t, leave=False, disable=True)
    else:
        pbar = tqdm(iter, total=t, leave=False)
    return pbar


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def wait_for_job(job):
    output = job.result()
    print(output)


def enable_gpu_benchmarking():
    if torch.cuda.is_available():
        cudnn.enabled = True
        cudnn.benchmark = True


def set_seed(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def save_model(model: nn.Module, path: str = "./"):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    try:
        torch.save(model.module.state_dict(), path)
    except AttributeError:
        torch.save(model.state_dict(), path)

    logger = logging.getLogger("my_logger") 
    logger.info(f"\nSaving model in {path}")


def train_epoch(
    model,
    train_loader,
    opt,
    criterion=nn.CrossEntropyLoss(),
    task_type="classification",
    disable_pbar=False,
):
    device = get_device()
    model = model.to(device)
    model.train()
    running_loss = correct_total = 0.0
    n_batches = len(train_loader)
    pbar = get_pbar(train_loader, disable=disable_pbar, enu=True)

    for it, (x, y) in pbar:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        pred = model(x)
        loss = criterion(pred.view(-1, pred.size(-1)), y.view(-1))
        loss.backward()
        opt.step()
        running_loss += loss.item()

        if task_type == "classification":
            pred_class = torch.argmax(pred, dim=-1)
            correct = (pred_class == y).sum().item()
            correct_total += correct
            batch_size = len(x)

            pbar.set_description(
                f"Iter {it}: train loss {loss.item():.3f} train acc {100 * correct/batch_size:.3f}"
            )
        elif task_type == "language_model":
            train_ppl = math.exp(loss.item())
            pbar.set_description(
                f"Iter {it}: train loss {loss.item():.3f} train ppl {train_ppl:.3f}"
            )

    loss = running_loss / n_batches
    acc = correct_total / len(train_loader.dataset)
    return loss, acc


def test(model, data_loader, criterion, n_iter=None, task_type="classification"):
    device = get_device()
    model = model.to(device)
    model.eval()
    correct = 0.0
    n_inputs = 0
    loss_list = []
    with torch.no_grad():
        for it, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred.view(-1, pred.size(-1)), y.view(-1))
            loss_list.append(loss.item())
            if task_type == "classification":
                pred_class = torch.argmax(pred, dim=1)
                correct += (pred_class == y).sum().item()
            n_inputs += len(x)
            if n_iter is not None and it == n_iter:
                break
        acc = correct / n_inputs
    return np.mean(loss_list), acc


def get_opt(params, opt, lr, weight_decay=0, momentum=0):
    if opt == "adam":
        return Adam(params, lr=lr, weight_decay=weight_decay)
    elif opt == "sgd":
        return SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer {opt}")


def get_sch(sch, opt, epochs):
    if sch == "none" or sch is None:
        return None
    elif sch == "cosine":

        return CosineAnnealingLR(opt, epochs, last_epoch=-1)
    else:
        raise ValueError(f"Unknown scheduler {sch}")


def get_attack_exp_path(args):
    subspace_key = args.subspace_type[:3] # aff if affine, lin if linear
    if args.exp_name is not None:
        exp_name = args.exp_name
    else:
        exp_name = (
            f"tv_{args.tv}_nv_{args.nv}_l1_{args.l1}_"
            f"{subspace_key}_inp_{args.sub_input}_{subspace_key}_intermediate_{args.sub_intermediate}"
        )
        if args.attack == "cp":
            exp_name = f"decor_{args.decor}_t_{args.T}_{exp_name}"
        else:
            exp_name = f"decor_{args.decor}_{exp_name}"
            
        if args.defense in ("dp", "clip"):
            exp_name = f"C_{args.C}_sigma_{args.sigma}_{exp_name}"

    if args.attack == "gm":
        if args.hard_sub:
            attack_name = f"gm_hard_{subspace_key}"
        elif args.sub_intermediate or args.sub_input > 0 :
            attack_name = f"gm_soft_{subspace_key}"
        else:
            attack_name = "gm"
    else:
        attack_name = "cp"
        
    fi = (
        model_type_dict.get(args.model) == "conv" # model is convolutional
        and (args.attack == "cp" or args.hard_sub or args.sub_input > 0)
    )
    if fi:
        attack_name = f"{attack_name}_{args.fi_method}"

    model_path = f"{args.model}/{args.act}" if args.model != "vgg16" else f"{args.model}" # act does not have meaning for vgg16
    exp_path = f"{exp_path_base}/{args.ds}/{model_path}/attack/{attack_name}/{args.defense}"
    
    if args.attack != "cp":
        label_name = "use_labels" if args.use_labels else "no_labels"
        exp_path = f"{exp_path}/{label_name}"
        
    exp_path = f"{exp_path}/{exp_name}"
    return exp_path
