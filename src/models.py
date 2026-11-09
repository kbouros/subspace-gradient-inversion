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
import torch.nn as nn
import numpy as np
from functools import partial
import logging
import re
import argparse
from utils import MyDataParallel

from vgg16 import vgg16
from dataset_utils import xshape_dict, nclasses_dict

# Available activation function for FC models. They don't affect vgg16.
act_dict = {"relu": nn.ReLU(), "leaky_relu": nn.LeakyReLU(0.2)}

supported_ds = {
    "fc": ["cifar10", "cifar100", "tiny_imagenet"],
    "vgg16": ["imagenet"],
}

def get_model(
    model_name,
    ds,
    h_dim=256,
    act="relu",
    dataparallel=True,
    load_path=None,
):
    logger = logging.getLogger("my_logger") 
    if act not in act_dict.keys():
        raise ValueError(f"Activation function {act} is not supported")

    # Dynamically recognize fc2, fc3, ..., fcL.
    fc_match = re.fullmatch(r"fc(\d+)", model_name.lower())

    if fc_match:
        L = int(fc_match.group(1))
        if L < 2:
            raise ValueError(
                f"FC model must have at least 2 linear layers, got L={L}"
            )
        if ds not in supported_ds["fc"]:
            raise ValueError(f"{model_name} does not support {ds}")

        model = FCL(
            L=L,
            ds=ds,
            h_dim=h_dim,
            act=act,
        )

    else:
        if model_name not in model_class_dict:
            raise ValueError(f"Unknown model: {model_name}")

        if ds not in supported_ds[model_name]:
            raise ValueError(f"{model_name} does not support {ds}")

        model = model_class_dict[model_name](
            ds=ds,
            h_dim=h_dim,
            act=act,
        )
    pretrained = model.pretrained

    if pretrained:
        logger.info(f"\nUsing Pretrained model for {model_name}")
    elif load_path is not None and pretrained == False:
        logger.info(f"\nLoading model from {load_path}")
        state_dict = torch.load(load_path)
        model.load_state_dict(state_dict)

    if dataparallel:
        device_list = [i for i in range(torch.cuda.device_count())]
        model = MyDataParallel(model, device_ids=device_list)

    return model

# This function will be used in the parser of other files that require the model name as an argument.
def parse_model_name(value):
    value = value.lower()

    match = re.fullmatch(r"fc(\d+)", value)

    if match:
        L = int(match.group(1))
        if L >= 2:
            return value

        raise argparse.ArgumentTypeError(
            f"Invalid model {value!r}: FC depth must be at least 2."
        )
        
    if value in model_class_dict:
        return value
        
    raise argparse.ArgumentTypeError(
        f"Invalid model {value!r}. Use fc<L> with L >= 2, "
        f"such as fc2 or fc5, or one of: {', '.join(model_class_dict)}"
    )

class FCL(nn.Module):
    def __init__(self, L=3, ds="cifar10", act="relu", h_dim=512, **kwargs):
        super().__init__()

        if not isinstance(L, int) or L < 2:
            raise ValueError(
                f"L must be an integer greater than or equal to 2, got {L!r}"
            )

        x_dim = int(np.prod(xshape_dict[ds]))
        n_classes = nclasses_dict[ds]

        # FC-L has L-1 hidden linear layers and one output layer.
        self.hidden_layers = nn.ModuleList()
        self.activation = act_dict[act]

        in_dim = x_dim
        for _ in range(L - 1):
            self.hidden_layers.append(nn.Linear(in_dim, h_dim))
            in_dim = h_dim

        self.final = nn.Linear(in_dim, n_classes)
        
        self.cpa_attack_layer_index = 0
        self.model_type = "fc"
        self.attack_index = 0
        self.intermediate_attack_indices = list(range(2, 2 * L, 2)) # correspondance of intermediate forward outputs and gradients
        self.pretrained = False
        
    def forward(self, x, return_intermediate=False, **kwargs):
        x = torch.flatten(x, start_dim=1)
        intermediates = []
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
            intermediates.append(x)
        out = self.final(x)
        if return_intermediate:
            return out, intermediates
        return out


model_class_dict = {
    "vgg16": partial(vgg16, pretrained=True),
}
