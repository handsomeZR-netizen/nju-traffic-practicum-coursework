#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.nn import ModuleDict

from agent_diy.conf.conf import Config


class Model(nn.Module):
    def __init__(self, device=None):
        super(Model, self).__init__()
        self.model_name = "network_diy_pielight_v1"
        self.device = device
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.unit_size = 128

        self.main_mlp = MLP([Config.DIM_OF_OBSERVATION, 256, 128], "main_mlp")
        self.label_mlp = ModuleDict(
            {
                f"label{label_index}_mlp": MLP(
                    [self.unit_size, self.label_size_list[label_index]],
                    f"label{label_index}_mlp",
                )
                for label_index in range(len(self.label_size_list))
            }
        )
        self.value_mlp = MLP([self.unit_size, 64, 1], "value_mlp")

    def forward(self, s, inference=False):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(np.array(s, dtype=np.float32), device=self.device, dtype=torch.float32)
        else:
            s = s.to(torch.float32)

        main_nn = self.main_mlp(s)
        result_list = []
        for label_index, _ in enumerate(self.label_size_list):
            result_list.append(self.label_mlp[f"label{label_index}_mlp"](main_nn))

        value_result = self.value_mlp(main_nn)
        result_list.append(value_result)

        logits = torch.flatten(torch.cat(result_list[:-1], 1), start_dim=1)
        value = result_list[-1]
        if inference:
            return [logits, value]
        return result_list

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()


def make_fc_layer(in_features: int, out_features: int, use_bias=True):
    fc_layer = nn.Linear(in_features, out_features, bias=use_bias)
    nn.init.orthogonal(fc_layer.weight)
    if use_bias:
        nn.init.zeros_(fc_layer.bias)
    return fc_layer


class MLP(nn.Module):
    def __init__(
        self,
        fc_feat_dim_list: List[int],
        name: str,
        non_linearity: nn.Module = nn.ReLU,
        non_linearity_last: bool = False,
    ):
        super(MLP, self).__init__()
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_feat_dim_list) - 1):
            fc_layer = make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1])
            self.fc_layers.add_module(f"{name}_fc{i + 1}", fc_layer)
            if i + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module(f"{name}_non_linear{i + 1}", non_linearity())

    def forward(self, data):
        return self.fc_layers(data)
