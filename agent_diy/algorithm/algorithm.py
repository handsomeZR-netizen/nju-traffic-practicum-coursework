###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import time

import torch
import torch.nn as nn

from agent_diy.conf.conf import Config


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

        self.model.set_train_mode()
        self.optimizer.zero_grad()
        rst_list = self.model(obs)
        total_loss, info_list = self.calculate_loss(data_list, rst_list)
        results = {"total_loss": total_loss.item()}

        total_loss.backward()
        if Config.USE_GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)

        self.optimizer.step()
        self.train_step += 1

        parsed_info = []
        for info in info_list:
            if isinstance(info, list):
                parsed_info.append([i.item() for i in info])
            else:
                parsed_info.append(info.item())

        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            _, (value_loss, policy_loss, entropy_loss) = parsed_info
            results["value_loss"] = round(value_loss, 2)
            results["policy_loss"] = round(policy_loss, 2)
            results["entropy_loss"] = round(entropy_loss, 2)

            if self.logger:
                self.logger.info(
                    f"policy_loss: {round(policy_loss, 2)}, "
                    f"value_loss: {round(value_loss, 2)}, "
                    f"entropy_loss: {round(entropy_loss, 2)}"
                )
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            self.last_report_monitor_time = now

    def calculate_loss(self, list_sample_data, model_output_data):
        (
            _obs,
            legal_action,
            sub_action,
            act,
            prob,
            reward,
            reward_sum,
            advantage,
            _value,
            _next_value,
            _is_train,
        ) = list_sample_data

        reward = reward_sum * Config.VALUE_COEF + reward * (1 - Config.VALUE_COEF)
        legal_action_flag_list = torch.split(legal_action, self.label_size_list, dim=1)

        usq_label_list = [act[:, shape_index].reshape(-1, 1).long() for shape_index in range(len(self.label_size_list))]
        sum_ls_list = [sum(self.label_size_list[0:i]) for i in range(len(self.label_size_list))]
        old_label_probability_list = []
        for shape_index in range(len(self.label_size_list)):
            old_label_probability_list.append(
                prob[:, sum_ls_list[shape_index] : sum_ls_list[shape_index] + self.label_size_list[shape_index]].reshape(
                    -1, self.label_size_list[shape_index]
                )
            )

        usq_weight_list = [sub_action[:, shape_index].reshape(-1, 1) for shape_index in range(len(self.label_size_list))]
        label_list = [ele.squeeze(dim=1) for ele in usq_label_list]
        weight_list = [weight.squeeze(dim=1) for weight in usq_weight_list]

        label_result = model_output_data[:-1]
        value_result = model_output_data[-1]

        fc2_value_result_squeezed = value_result.squeeze(dim=1)
        self.value_cost = 0.5 * torch.mean(torch.square(reward - fc2_value_result_squeezed), dim=0)

        epsilon = 1e-5
        self.policy_cost = torch.tensor(0.0, device=self.model.device)
        label_probability_list = []

        for task_index in range(len(self.is_reinforce_task_list)):
            if not self.is_reinforce_task_list[task_index]:
                continue

            boundary = torch.pow(torch.tensor(10.0, device=self.model.device), torch.tensor(20.0, device=self.model.device))
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

            label_exp_logits = legal_action_flag_list[task_index] * torch.exp(label_logits_subtract_max) + self.min_policy
            label_sum_exp_logits = label_exp_logits.sum(1, keepdim=True)
            label_probability = 1.0 * label_exp_logits / label_sum_exp_logits
            label_probability_list.append(label_probability)

            policy_p = (one_hot_actions * label_probability).sum(1)
            policy_log_p = torch.log(policy_p + epsilon)
            old_policy_p = (one_hot_actions * old_label_probability_list[task_index] + epsilon).sum(1)
            old_policy_log_p = torch.log(old_policy_p)

            ratio = torch.exp(policy_log_p - old_policy_log_p)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage
            self.policy_cost += (-torch.min(surr1, surr2) * weight_list[task_index]).mean()

        self.entropy_cost = torch.tensor(0.0, device=self.model.device)
        for task_index in range(len(self.is_reinforce_task_list)):
            if not self.is_reinforce_task_list[task_index]:
                continue
            label_probability = label_probability_list[task_index]
            temp_entropy = -label_probability * torch.log(label_probability + epsilon)
            self.entropy_cost += torch.sum(temp_entropy, dim=1).mean()

        total_loss = self.value_cost + self.policy_cost - self.var_beta * self.entropy_cost
        info_list = [total_loss, [self.value_cost, self.policy_cost, self.entropy_cost]]
        return total_loss, info_list
