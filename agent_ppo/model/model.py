#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import torch.nn as nn
from torch.nn import ModuleDict
import numpy as np
from typing import List
from agent_ppo.conf.conf import Config


##################
## Actual model ##
##################


class Model(nn.Module):
    def __init__(self, device=None):
        super(Model, self).__init__()
        self.model_name = "network_traffic_v1"

        self.device = device
        self.label_size_list = Config.LABEL_SIZE_LIST

        self.unit_size = 128
        # build network
        # 构建网络
        # main module
        # 网络主模块
        all_dims = [Config.DIM_OF_OBSERVATION, 256, 128]
        self.main_mlp = MLP(all_dims, "main_mlp")

        # output label
        # 输出标签
        self.label_mlp = ModuleDict(
            {
                "label{0}_mlp".format(label_index): MLP(
                    [self.unit_size, self.label_size_list[label_index]],
                    "label{0}_mlp".format(label_index),
                )
                for label_index in range(len(self.label_size_list))
            }
        )

        # output value
        # 输出value
        self.value_mlp = MLP([self.unit_size, 64, 1], "value_mlp")

    def forward(self, s, inference=False):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(
                np.array(s, dtype=np.float32),
                device=self.device,
                dtype=torch.float32,
            )
        else:
            s = s.to(torch.float32)

        main_nn = self.main_mlp(s)

        result_list = []
        # output label
        # 输出标签
        for label_index, label_dim in enumerate(self.label_size_list[:]):
            label_mlp_out = self.label_mlp["label{0}_mlp".format(label_index)](main_nn)
            result_list.append(label_mlp_out)

        # output value
        # 输出value
        value_result = self.value_mlp(main_nn)
        result_list.append(value_result)

        logits = torch.flatten(torch.cat(result_list[:-1], 1), start_dim=1)
        value = result_list[-1]
        if inference:
            return [logits, value]
        else:
            return result_list

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()


#######################
## Utility functions ##
#######################


def make_fc_layer(in_features: int, out_features: int, use_bias=True):
    """
    Wrapper function to create and initialize a linear layer
    Args:
        in_features (int): ``in_features``
        out_features (int): ``out_features``
    Returns:
        nn.Linear: the initialized linear layer
    """
    """
    用于创建和初始化线性层的包装函数
    参数：
        in_features (int): ``in_features``
        out_features (int): ``out_features``
    返回值：
        nn.Linear：已初始化的线性层
    """
    fc_layer = nn.Linear(in_features, out_features, bias=use_bias)

    nn.init.orthogonal(fc_layer.weight)
    if use_bias:
        nn.init.zeros_(fc_layer.bias)

    return fc_layer


############################
## Building-block classes ##
############################


class MLP(nn.Module):
    """
    A simple multi-layer perceptron
    """

    """
    一个简单的多层感知器
    """

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
            self.fc_layers.add_module("{0}_fc{1}".format(name, i + 1), fc_layer)
            if i + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module("{0}_non_linear{1}".format(name, i + 1), non_linearity())

    def forward(self, data):
        return self.fc_layers(data)
