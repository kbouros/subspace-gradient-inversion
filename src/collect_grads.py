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

import argparse
import time
import numpy as np
import torch.nn as nn
import torch
import submitit
import logging
import copy
from torch.utils.data import TensorDataset, DataLoader
from opacus import PrivacyEngine
from opacus.grad_sample import GradSampleModule
import gc

JOB_NAME = "collect_grads"

from utils import (
    get_device,
    exp_path_base,
    setup,
    write_pickle,
    submitit_logdir,
    get_grads_file,
    slurm_partition,
)
from dataset_utils import (
    get_dataloaders,
    ds_dict,
)
from models import get_model, act_dict, parse_model_name


def disable_inplace_relu(module):
    # Needed for hooks for DP when using vgg16
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.ReLU(inplace=False))
        else:
            disable_inplace_relu(child)
            

def apply_dp_noise_manual(sample_grad_list, C=1, sigma=0):
    # Manually clips and adds DP noise to gradients. Manual clipping is memory heavy for deeper models
    #
    # sample_grad_list: list of sample_wise_gradients for each parameter, where each gradient has dimensionality: batch_size x param_size
    # C: L2 norm for clipping
    # sigma: stddev of noise
    batch_size = sample_grad_list[0].shape[0]
    device = sample_grad_list[0].device
    C = torch.tensor(C, device=device)
    sample_grad_clip_list = []

    with torch.no_grad():

        for b in range(batch_size):
            # concatenate grad from each parameter into a single vector
            g_b_cat = torch.cat([g[b].view(-1) for g in sample_grad_list])

            # Compute scaling factor for "clipping"
            scaling = 1 / torch.maximum(
                torch.tensor(1.0, device=device), g_b_cat.norm() / C
            )

            # scale the gradients of all the parameters using the scaling factor
            for i, g in enumerate(sample_grad_list):
                g_b_clip = g[b] * scaling
                if b == 0:
                    sample_grad_clip_list.append([g_b_clip])
                else:
                    sample_grad_clip_list[i].append(g_b_clip)

        # Compute the sum of sample-wise clipped gradients, add noise, compute mean. Keep clipped and dp lists.
        grad_list_clip = []
        grad_list_dp = []
        for g_batch_clip in sample_grad_clip_list:
            g_sum_clip = torch.sum(torch.stack(g_batch_clip, dim=0), dim=0)
            grad_list_clip.append(g_sum_clip / batch_size) # add to clip list
            
            g_dp = (
                g_sum_clip +
                torch.randn(g_sum_clip.shape, device=device) * sigma * C
            ) / batch_size
            grad_list_dp.append(g_dp) # add to dp list

    return grad_list_clip, grad_list_dp


def apply_dp_noise_opacus(model, x, y, criterion, C=1.0, sigma=0.0, grad_sample_mode="ghost"):
    # Use opacus for gradient clipping and noise addition. Mainly useful for bigger models
    # like vgg16 where clipping is too memory heavy and "ghost" mode is needed.
    device = x.device
    batch_size = x.shape[0]
    dp_model = copy.deepcopy(model).to(device)
    dp_model.train()
    disable_inplace_relu(dp_model)
    dp_criterion = copy.deepcopy(criterion)
    
    # Dummy optimizer: for opacus
    base_optimizer = torch.optim.SGD(dp_model.parameters(), lr=0.0)
  
    # Dummy dataset for privacy engine. Ensures correct expected batch size while keeping the same x,y.
    # Note that we only care about gradients here; it doesn't matter that subsampling is not accounted.
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(
        dataset,
        batch_size=x.shape[0],
        shuffle=False,
        drop_last=False,
    )

    privacy_engine = PrivacyEngine(accountant="rdp")
    private_objects = privacy_engine.make_private(
        module=dp_model,
        optimizer=base_optimizer,
        criterion=dp_criterion,
        data_loader=train_loader,          
        noise_multiplier=sigma,
        max_grad_norm=C,
        grad_sample_mode=grad_sample_mode, 
        poisson_sampling=False,    
    )

    if grad_sample_mode=="ghost":
        dp_model, dp_optimizer, dp_criterion, _ = private_objects
    else:
        dp_model, dp_optimizer, _ = private_objects

    dp_optimizer.zero_grad()

    logits = dp_model(x)
    loss = dp_criterion(logits, y)
    loss.backward()

    # Sets DP grads
    dp_optimizer.pre_step()

    dp_grads = [p.grad.detach().clone() for p in dp_model.parameters()]

    # Clear optimizer references and delete large objects to avoid memory accumulation.
    dp_optimizer.zero_grad(set_to_none=True)

    del loss, logits, private_objects, privacy_engine
    del dp_optimizer, base_optimizer, dp_model, dp_criterion, train_loader, dataset

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return dp_grads


def compute_snr_from_clipped_grads(grad_list_clip, C, sigma, batch_size):
    """
    grad_list_clip contains clean averaged clipped gradients:
        g_sum_clip / batch_size
    """

    if len(grad_list_clip) == 0:
        return None, None

    device = grad_list_clip[0].device
    signal_sq_sum = torch.tensor(0.0, device=device)
    num_elems = 0

    for g in grad_list_clip:
        if g is None:
            continue

        g = g.detach()
        g = torch.nan_to_num(g, nan=0.0, posinf=1e4, neginf=-1e4)

        signal_sq_sum += torch.sum(g ** 2)
        num_elems += g.numel()

    if num_elems == 0:
        return None, None

    noise_std = sigma * C / batch_size
    noise_var = noise_std ** 2

    if noise_var < 1e-24:
        return None, None

    mean_signal_power = signal_sq_sum / num_elems
    snr = mean_signal_power / noise_var
    snr_db = 10 * torch.log10(snr)

    return snr.item(), snr_db.item()
    

def get_model_params(model):
    param_list = []
    for name, param in model.named_parameters():
        if "tok_emb" not in name:
            param_list.append(param)
    return param_list


def collect_grads(args):
    batch_size_list = [8, 16, 32, 64, 128, 256]

    # === Setup ===
    model_path = f"{args.model}/{args.act}" if args.model != "vgg16" else f"{args.model}" # act does not have meaning for vgg16
    exp_path = f"{exp_path_base}/{args.ds}/{model_path}/grads/{args.defense}"
    logger = setup(exp_path=exp_path, log_file="collect_grad.log")
    logger.info(
        f"Collecting gradients with {args.model} model on {args.ds} with batch_sizes:"
        f" {batch_size_list} for {args.n_batch} batches"
    )
    device = get_device()

    # === Model ===
    dataparallel = False if (args.defense == "dp") else True
    model = get_model(
        model_name=args.model,
        ds=args.ds,
        h_dim=args.h_dim,
        act=args.act,
        load_path=f"{exp_path_base}/{args.ds}/{model_path}/model.pt",
        dataparallel=dataparallel,
    )
    model_type = model.model_type
    model = model.to(device)
    if args.defense == "dp" and args.manual_dp:
        disable_inplace_relu(model) 
        model = GradSampleModule(model)
    model.train()

    # Criterion
    criterion = nn.CrossEntropyLoss()

    # === Collect grads ===
    for batch_size in batch_size_list:
        logger.info(f"\nCollecting grads with batch size: {batch_size}\n")
        _, dl_test = get_dataloaders(
            args.ds,
            batch_size=batch_size,
            load_train=False,
            shuffle_test=True,
        )

        data = {"x": [], "y": [], "z": [], "grad": [], "grad_clip": [], "grad_dp": []}
        snr_db_values = []

        for i, (x, y) in enumerate(dl_test):
            if i == args.n_batch:
                break
            data["x"].append(x.numpy())
            data["y"].append(y.numpy())
            x, y = x.to(device), y.to(device)

            # 1. Original batch gradients
            model.zero_grad()
            if model_type == "conv" or args.ds == "imdb":
                logits, z = model(x=x, return_z=True)
                data["z"].append(z.detach().cpu().numpy())
            else:
                logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            grad = [p.grad.detach().cpu().numpy() for p in model.parameters()]
            data["grad"].append(grad)

            # 2. DP gradients (manual / OPACUS)
            if args.defense == "dp":
                if args.manual_dp:
                    # Manual per-sample clipping and DP noise. Avoid for larger models like VGG16. 
                    sample_grad_list = [p.grad_sample.detach() for p in model.parameters()]
                    grad_clip, grad_dp = apply_dp_noise_manual(sample_grad_list, args.C, args.sigma)
                else:
                     # Clipped and dp gradients using "ghost" mode of OPACUS for memory efficiency.
                    grad_clip = apply_dp_noise_opacus(model, x, y, criterion, args.C, 0, grad_sample_mode="ghost")
                    grad_dp = apply_dp_noise_opacus(model, x, y, criterion, args.C, args.sigma, grad_sample_mode="ghost")
            
                # Store gradients
                data["grad_clip"].append([g.detach().cpu().numpy() for g in grad_clip])
                data["grad_dp"].append([g.detach().cpu().numpy() for g in grad_dp])
                
                # Compute SNR
                if args.sigma > 0:
                    snr, snr_db = compute_snr_from_clipped_grads(grad_clip, args.C, args.sigma, batch_size)
                    snr_db_values.append(snr_db)
                    logger.info(f"{i}) Linear SNR : {snr:.4f}, SNR (dB) : {snr_db:.4f}")  

        if args.defense == "dp" and args.sigma > 0:
            mean = np.mean(snr_db_values)
            std = np.std(snr_db_values)
            logger.info(f"\nMean SNR ± std: {mean:.3f} ± {std:.3f} dB")

        # Save data
        grads_pkl_file = get_grads_file(
            args.ds, args.model, args.act, args.defense, batch_size, args.C, args.sigma
        )
        write_pickle(data, grads_pkl_file)
        logger.info(f"\nDone!\nData saved in {grads_pkl_file}\n")


if __name__ == "__main__":
    logger = logging.getLogger("my_logger") 
    start = time.time()
    parser = argparse.ArgumentParser(description="Collect Gradients")
    parser.add_argument(
        "--ds",
        type=str,
        default="cifar10",
        help="dataset",
        choices=ds_dict.keys(),
    )
    parser.add_argument("--n_embd", type=int, default=100, help="Embedding Size")

    parser.add_argument(
        "--defense",
        type=str,
        default="nodef",
        help="Defense for gradient sharing",
        choices=["nodef", "dp"],
    )
    parser.add_argument(
        "--n_batch", type=int, default=10, help="number of batches to collect gradients"
    )

    parser.add_argument(
        "--model",
        type=parse_model_name,
        default="fc3",
        help="Model name, such as fc<L> (e.g., fc2 or fc3) or vgg16"
    )
    parser.add_argument(
        "--act",
        type=str,
        default="relu",
        help="activation function",
        choices=act_dict.keys(),
    )
    parser.add_argument("--h_dim", type=int, default=512, help="hidden dim")

    parser.add_argument("--submitit", action="store_true", help="submit to queue")
    parser.add_argument("--n_cpu", type=int, default=40, help="cpus per task")
    parser.add_argument("--n_gpu", type=int, default=8, help="gpus per node")

    # DP
    parser.add_argument("--C", type=float, default=1.0, help="Clipping Norm")
    parser.add_argument("--sigma", type=float, default=0.0, help="sigma for DP noise")
    parser.add_argument("--manual_dp", action="store_true", help="performs DP manually instead of opacus")

    args = parser.parse_args()

    # Submit it
    if args.submitit:
        executor = submitit.AutoExecutor(folder=submitit_logdir)
        executor.update_parameters(
            timeout_min=600,
            nodes=1,
            slurm_partition=slurm_partition,
            slurm_job_name=JOB_NAME,
            gpus_per_node=args.n_gpu,
            cpus_per_task=args.n_cpu,
            slurm_mem="400G",
        )
        print(JOB_NAME)
        job = executor.submit(collect_grads, args)
        print(f"Scheduled { job }.")
    else:
        collect_grads(args)

    end = time.time()
    runtime = end - start

    logger.info(f"\nRuntime: {runtime:.2f} s")
