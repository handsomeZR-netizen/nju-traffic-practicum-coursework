###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import torch.nn as nn
import os
import time
import numpy as np
from agent_ppo.conf.conf import Config
import torch.nn.functional as F


class Algorithm:
    def __init__(self, model, optimizer, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model

        self.optimizer = optimizer
        self.parameters = [p for param_group in self.optimizer.param_groups for p in param_group["params"]]
        self.logger = logger
        self.monitor = monitor

        self.num_head = Config.NUMB_HEAD
        self._gamma = Config.GAMMA

        self.label_size_list = Config.LABEL_SIZE_LIST
        self.is_reinforce_task_list = Config.IS_REINFORCE_TASK_LIST
        self.m_var_beta = Config.BETA_START
        self.min_policy = Config.MIN_POLICY
        self.clip_param = Config.CLIP_PARAM
        self.var_beta = self.m_var_beta

        self.last_report_monitor_time = 0
        self.train_step = 0

    def learn(self, list_sample_data):
        t_data = list_sample_data

        obs = torch.stack([frame.obs for frame in t_data]).to(self.model.device)
        legal_action = torch.stack([frame.legal_action for frame in t_data]).to(self.model.device)
        sub_action = torch.stack([frame.sub_action for frame in t_data]).to(self.model.device)
        act = torch.stack([frame.act for frame in t_data]).to(self.model.device)
        prob = torch.stack([frame.prob for frame in t_data]).to(self.model.device)
        reward = torch.stack([frame.reward for frame in t_data]).to(self.model.device)
        reward_sum = torch.stack([frame.reward_sum for frame in t_data]).to(self.model.device)
        advantage = torch.stack([frame.advantage for frame in t_data]).to(self.model.device)
        value = torch.stack([frame.value for frame in t_data]).to(self.model.device)
        next_value = torch.stack([frame.next_value for frame in t_data]).to(self.model.device)
        is_train = torch.stack([frame.is_train for frame in t_data]).to(self.model.device)

        data_list = [
            obs,
            legal_action,
            sub_action,
            act,
            prob,
            reward,
            reward_sum,
            advantage,
            value,
            next_value,
            is_train,
        ]

        # model settings before prediction
        # 预测前先对model进行设置
        self.model.set_train_mode()
        self.optimizer.zero_grad()

        rst_list = self.model(obs)
        total_loss, info_list = self.calculate_loss(data_list, rst_list)
        results = {}

        results["total_loss"] = total_loss.item()

        total_loss.backward()

        # grad clip
        # 设置梯度裁剪
        if Config.USE_GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)

        self.optimizer.step()
        self.train_step += 1

        _info_list = []
        for info in info_list:
            if isinstance(info, list):
                _info = [i.item() for i in info]
            else:
                _info = info.item()
            _info_list.append(_info)

        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            _, (value_loss, policy_loss, entropy_loss) = _info_list
            results["value_loss"] = round(value_loss, 2)
            results["policy_loss"] = round(policy_loss, 2)
            results["entropy_loss"] = round(entropy_loss, 2)

            self.logger.info(
                f"policy_loss: {round(policy_loss, 2)}, value_loss: {round(value_loss, 2)}, entropy_loss: {round(entropy_loss, 2)}"
            )
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})

            self.last_report_monitor_time = now

    def calculate_loss(self, list_sample_data, model_output_data):
        (
            obs,
            legal_action,
            sub_action,
            act,
            prob,
            reward,
            reward_sum,
            advantage,
            value,
            next_value,
            is_train,
        ) = list_sample_data

        reward = reward_sum * Config.VALUE_COEF + reward * (1 - Config.VALUE_COEF)
        # parse legal_action
        # 解析合法动作
        legal_action_flag_list = torch.split(legal_action, self.label_size_list, dim=1)
        usq_label_list = list()
        for shape_index in range(len(self.label_size_list)):
            usq_label_list.append(act[:, shape_index])
        for shape_index in range(len(self.label_size_list)):
            usq_label_list[shape_index] = usq_label_list[shape_index].reshape(-1, 1).long()
        # parse prob
        # 解析决策概率数据
        sum_ls_list = [sum(self.label_size_list[0:i]) for i in range(len(self.label_size_list))]

        old_label_probability_list = list()
        for shape_index in range(len(self.label_size_list)):
            old_label_probability_list.append(
                prob[:, sum_ls_list[shape_index] : sum_ls_list[shape_index] + self.label_size_list[shape_index]]
            )
        for shape_index in range(len(self.label_size_list)):
            old_label_probability_list[shape_index] = old_label_probability_list[shape_index].reshape(
                -1, self.label_size_list[shape_index]
            )
        # parse sub_action
        # 解析sub_action
        usq_weight_list = list()
        for shape_index in range(len(self.label_size_list)):
            usq_weight_list.append(sub_action[:, shape_index])
        for shape_index in range(len(self.label_size_list)):
            usq_weight_list[shape_index] = usq_weight_list[shape_index].reshape(-1, 1)

        label_list = []
        for ele in usq_label_list:
            label_list.append(ele.squeeze(dim=1))
        weight_list = []
        for weight in usq_weight_list:
            weight_list.append(weight.squeeze(dim=1))

        label_result = model_output_data[:-1]

        value_result = model_output_data[-1]

        # loss of value net
        # 计算value loss
        fc2_value_result_squeezed = value_result.squeeze(dim=1)
        self.value_cost = 0.5 * torch.mean(torch.square(reward - fc2_value_result_squeezed), dim=0)

        # entropy loss calculate
        # 计算entropy loss
        label_logits_subtract_max_list = []
        label_sum_exp_logits_list = []
        label_probability_list = []

        epsilon = 1e-5  # 0.00001

        # policy loss: ppo clip loss
        # 计算policy loss: ppo 算法的clip loss
        self.policy_cost = torch.tensor(0.0)
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                final_log_p = torch.tensor(0.0)
                boundary = torch.pow(torch.tensor(10.0), torch.tensor(20.0))
                one_hot_actions = nn.functional.one_hot(label_list[task_index].long(), self.label_size_list[task_index])

                legal_action_flag_list_max_mask = (1 - legal_action_flag_list[task_index]) * boundary

                label_logits_subtract_max = torch.clamp(
                    label_result[task_index]
                    - torch.max(
                        label_result[task_index] - legal_action_flag_list_max_mask,
                        dim=1,
                        keepdim=True,
                    ).values,
                    -boundary,
                    1,
                )

                label_logits_subtract_max_list.append(label_logits_subtract_max)

                label_exp_logits = (
                    legal_action_flag_list[task_index] * torch.exp(label_logits_subtract_max) + self.min_policy
                )

                label_sum_exp_logits = label_exp_logits.sum(1, keepdim=True)
                label_sum_exp_logits_list.append(label_sum_exp_logits)

                label_probability = 1.0 * label_exp_logits / label_sum_exp_logits
                label_probability_list.append(label_probability)

                policy_p = (one_hot_actions * label_probability).sum(1)
                policy_log_p = torch.log(policy_p + epsilon)
                old_policy_p = (one_hot_actions * old_label_probability_list[task_index] + epsilon).sum(1)
                old_policy_log_p = torch.log(old_policy_p)
                final_log_p = final_log_p + policy_log_p - old_policy_log_p
                ratio = torch.exp(final_log_p)
                clip_ratio = ratio.clamp(0.0, 3.0)

                surr1 = clip_ratio * advantage
                surr2 = ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param) * advantage
                temp_policy_loss = -torch.sum(
                    torch.minimum(surr1, surr2) * (weight_list[task_index].float()) * 1
                ) / torch.maximum(torch.sum((weight_list[task_index].float()) * 1), torch.tensor(1.0))

                self.policy_cost = self.policy_cost + temp_policy_loss

        # cross entropy loss
        # 计算cross entropy loss
        current_entropy_loss_index = 0
        entropy_loss_list = []
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                temp_entropy_loss = -torch.sum(
                    label_probability_list[current_entropy_loss_index]
                    * legal_action_flag_list[task_index]
                    * torch.log(label_probability_list[current_entropy_loss_index] + epsilon),
                    dim=1,
                )

                temp_entropy_loss = -torch.sum(
                    (temp_entropy_loss * weight_list[task_index].float() * 1)
                ) / torch.maximum(torch.sum(weight_list[task_index].float() * 1), torch.tensor(1.0))

                entropy_loss_list.append(temp_entropy_loss)
                current_entropy_loss_index = current_entropy_loss_index + 1
            else:
                temp_entropy_loss = torch.tensor(0.0)
                entropy_loss_list.append(temp_entropy_loss)

        self.entropy_cost = torch.tensor(0.0)
        for entropy_element in entropy_loss_list:
            self.entropy_cost = self.entropy_cost + entropy_element

        self.entropy_cost_list = entropy_loss_list

        self.loss = self.value_cost + self.policy_cost + self.var_beta * self.entropy_cost

        return self.loss, [
            self.loss,
            [self.value_cost, self.policy_cost, self.entropy_cost],
        ]
