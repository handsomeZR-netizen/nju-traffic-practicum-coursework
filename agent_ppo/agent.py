#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import numpy as np
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from kaiwu_agent.agent.base_agent import (
    predict_wrapper,
    exploit_wrapper,
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
    BaseAgent,
)
from agent_ppo.model.model import Model
from kaiwu_agent.utils.common_func import attached
from agent_ppo.feature.definition import ActData, FeatureProcess, ObsData, get_webster_lane_group, on_enter_lane
from agent_ppo.conf.conf import Config
from agent_ppo.algorithm.algorithm import Algorithm


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        torch.manual_seed(0)
        self.device = device
        self.model = Model(device).to(self.device)
        parameters = self.model.parameters()
        self.optimizer = torch.optim.Adam(
            params=parameters,
            lr=Config.INIT_LEARNING_RATE_START,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST
        self.logger = logger
        self.monitor = monitor
        self.preprocess = FeatureProcess(logger)
        self.algorithm = Algorithm(self.model, self.optimizer, self.device, self.logger, self.monitor)
        self.phase_lane_map = {
            lane_id: int(phase_id)
            for phase_id, lane_ids in get_webster_lane_group().items()
            for lane_id in lane_ids
        }
        self.reset()
        super().__init__(agent_type, device, logger, monitor)

    def reset(self):
        self.preprocess.reset()
        self.last_phase = None
        self.same_phase_count = 0
        self.phase_idle_rounds = {phase_id: 0 for phase_id in range(Config.DIM_OF_ACTION_PHASE_1)}

    def __predict_detail(self, list_obs_data, exploit_flag=False):
        feature = [obs_data.feature for obs_data in list_obs_data]
        legal_action = [obs_data.legal_action for obs_data in list_obs_data]
        logit_bias_list = [getattr(obs_data, "logit_bias", None) for obs_data in list_obs_data]
        self.model.set_eval_mode()

        s = torch.tensor(feature).view(len(feature), Config.DIM_OF_OBSERVATION).float().to(self.device)
        with torch.no_grad():
            output_list = self.model(s, inference=True)

        logits = output_list[0].detach().cpu().numpy()
        value = output_list[1].detach().cpu().numpy()

        list_act_data = []
        for i in range(len(legal_action)):
            final_logits = logits[i]
            if logit_bias_list[i] is not None:
                final_logits = final_logits + np.array(logit_bias_list[i], dtype=np.float32)

            prob, action, d_action = self._sample_masked_action(
                final_logits,
                np.array(legal_action[i], dtype=np.float32),
            )
            if exploit_flag and getattr(list_obs_data[i], "rule_action", None) is not None:
                d_action = list(list_obs_data[i].rule_action)

            list_act_data.append(
                ActData(
                    junction_id=0,
                    action=action,
                    d_action=d_action,
                    prob=prob,
                    value=value[i : i + 1],
                )
            )
        return list_act_data

    @predict_wrapper
    def predict(self, list_obs_data):
        return self.__predict_detail(list_obs_data, exploit_flag=False)

    @exploit_wrapper
    def exploit(self, observation):
        obs_data = self.observation_process(observation["obs"], observation["extra_info"])
        if not obs_data:
            return [[None, None, None]]

        if obs_data.rule_action is not None:
            act_data = ActData(
                junction_id=0,
                action=list(obs_data.rule_action),
                d_action=list(obs_data.rule_action),
                prob=[0.0] * sum(self.label_size_list),
                value=np.zeros((1, 1), dtype=np.float32),
            )
            return self.action_process(act_data, False)

        act_data = self.__predict_detail([obs_data], exploit_flag=True)
        return self.action_process(act_data[0], False)

    @learn_wrapper
    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        model_state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(model_state_dict_cpu, model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        if not os.path.exists(model_file_path):
            self.logger.info(f"model file {model_file_path} not found, skip loading")
            return

        self.model.load_state_dict(torch.load(model_file_path, map_location=self.model.device))
        self.logger.info(f"load model {model_file_path} successfully")

    def observation_process(self, raw_obs, extra_info=None):
        """
        Convert raw environment observations into compact traffic features and rule priors.
        将环境原始观测转换为交通特征和启发式先验。
        """

        self.preprocess.update_traffic_info(raw_obs, extra_info)

        frame_state = raw_obs["frame_state"]
        vehicles = frame_state["vehicles"]

        phase_metrics, scene_waiting_vehicle_count = self._collect_phase_metrics(vehicles)
        phase_scores = self._build_phase_scores(phase_metrics)
        phase_action, duration_action = self._build_signal_plan(
            phase_metrics,
            phase_scores,
            scene_waiting_vehicle_count,
        )
        observation = self._build_observation(
            phase_metrics,
            phase_scores,
            scene_waiting_vehicle_count,
        )
        legal_action = self._build_legal_action()
        logit_bias = self._build_logit_bias(phase_scores, duration_action)

        return ObsData(
            feature=observation,
            legal_action=legal_action,
            sub_action_mask=[1] * Config.NUMB_HEAD,
            rule_action=[phase_action, duration_action],
            logit_bias=logit_bias,
        )

    def action_process(self, act_data, is_stochastic=True):
        junction_id = act_data.junction_id
        action = act_data.action if is_stochastic else act_data.d_action
        action_p = int(action[0])
        action_d = Config.DURATION_TO_SECONDS[int(action[1])]
        return [junction_id, action_p, action_d]

    def _collect_phase_metrics(self, vehicles):
        metrics = {
            phase_id: {
                "vehicle_count": 0,
                "queue_count": 0,
                "wait_sum": 0.0,
                "delay_sum": 0.0,
                "speed_ratio_sum": 0.0,
                "avg_wait": 0.0,
                "avg_delay": 0.0,
            }
            for phase_id in range(Config.DIM_OF_ACTION_PHASE_1)
        }

        scene_waiting_vehicle_count = 0
        for vehicle in vehicles:
            speed = float(vehicle.get("speed", 0.0))
            if speed <= 0.1:
                scene_waiting_vehicle_count += 1

            if not on_enter_lane(vehicle):
                continue

            phase_id = self.phase_lane_map.get(vehicle["lane"])
            if phase_id is None:
                continue

            speed_limit = self._get_lane_speed_limit(vehicle["lane"])
            wait_time = float(vehicle.get("waiting_time", self.preprocess.waiting_time_store.get(vehicle["v_id"], 0.0)))
            speed_ratio = min(speed / max(speed_limit, 1e-6), 1.0)
            delay_proxy = wait_time + (1.0 - speed_ratio) * 5.0

            metrics[phase_id]["vehicle_count"] += 1
            metrics[phase_id]["wait_sum"] += wait_time
            metrics[phase_id]["delay_sum"] += delay_proxy
            metrics[phase_id]["speed_ratio_sum"] += speed_ratio
            if speed <= 0.3:
                metrics[phase_id]["queue_count"] += 1

        for phase_id, metric in metrics.items():
            vehicle_count = max(metric["vehicle_count"], 1)
            metric["avg_wait"] = metric["wait_sum"] / vehicle_count
            metric["avg_delay"] = metric["delay_sum"] / vehicle_count

        return metrics, scene_waiting_vehicle_count

    def _build_phase_scores(self, phase_metrics):
        phase_scores = {}
        for phase_id, metric in phase_metrics.items():
            idle_bonus = min(self.phase_idle_rounds[phase_id], Config.MAX_IDLE_ROUNDS) * Config.IDLE_BONUS_COEF
            score = (
                metric["queue_count"] * Config.QUEUE_PRESSURE_COEF
                + metric["avg_wait"] * Config.WAIT_PRESSURE_COEF
                + metric["vehicle_count"] * Config.VEHICLE_PRESSURE_COEF
                + metric["avg_delay"] * Config.DELAY_PRESSURE_COEF
                + idle_bonus
            )
            if metric["vehicle_count"] == 0:
                score *= 0.2
            phase_scores[phase_id] = score

        return phase_scores

    def _build_signal_plan(self, phase_metrics, phase_scores, scene_waiting_vehicle_count):
        ranked_phases = sorted(
            phase_metrics.keys(),
            key=lambda phase_id: (
                phase_scores[phase_id],
                phase_metrics[phase_id]["queue_count"],
                phase_metrics[phase_id]["avg_wait"],
            ),
            reverse=True,
        )
        active_phases = [phase_id for phase_id in ranked_phases if phase_metrics[phase_id]["vehicle_count"] > 0]

        if not active_phases:
            next_phase = 0 if self.last_phase is None else (self.last_phase + 1) % Config.DIM_OF_ACTION_PHASE_1
        else:
            best_phase = active_phases[0]
            if self.last_phase is None:
                next_phase = best_phase
            else:
                current_phase = self.last_phase
                current_score = phase_scores[current_phase]
                best_score = phase_scores[best_phase]
                current_metric = phase_metrics[current_phase]
                best_metric = phase_metrics[best_phase]

                keep_current = best_phase == current_phase
                if not keep_current and current_metric["vehicle_count"] > 0:
                    if best_score < current_score + Config.PHASE_SWITCH_MARGIN:
                        keep_current = True
                    elif (
                        best_score < current_score * Config.PHASE_SWITCH_RATIO
                        and current_metric["queue_count"] > 0
                    ):
                        keep_current = True

                starved_phase = next(
                    (
                        phase_id
                        for phase_id in active_phases
                        if phase_id != current_phase
                        and self.phase_idle_rounds[phase_id] >= Config.MAX_IDLE_ROUNDS
                        and phase_scores[phase_id] >= current_score * 0.85
                    ),
                    None,
                )
                if starved_phase is not None:
                    best_phase = starved_phase
                    keep_current = False

                if current_metric["vehicle_count"] == 0 and best_metric["vehicle_count"] > 0:
                    keep_current = False

                next_phase = current_phase if keep_current else best_phase

        duration_action = self._select_duration_action(
            next_phase,
            ranked_phases,
            phase_metrics,
            phase_scores,
            scene_waiting_vehicle_count,
        )

        if self.last_phase == next_phase:
            self.same_phase_count += 1
        else:
            self.same_phase_count = 1

        for phase_id in self.phase_idle_rounds:
            self.phase_idle_rounds[phase_id] = 0 if phase_id == next_phase else self.phase_idle_rounds[phase_id] + 1

        self.last_phase = next_phase
        return next_phase, duration_action

    def _select_duration_action(self, next_phase, ranked_phases, phase_metrics, phase_scores, scene_waiting_vehicle_count):
        selected_metric = phase_metrics[next_phase]
        second_phase = next((phase_id for phase_id in ranked_phases if phase_id != next_phase), None)
        second_score = phase_scores[second_phase] if second_phase is not None else 0.0
        score_gap = phase_scores[next_phase] - second_score

        if selected_metric["vehicle_count"] == 0:
            duration_action = 0
        elif (
            selected_metric["queue_count"] >= 8
            or selected_metric["avg_wait"] >= 12.0
            or scene_waiting_vehicle_count >= 180
        ):
            duration_action = 3
        elif (
            selected_metric["queue_count"] >= 5
            or selected_metric["avg_wait"] >= 7.0
            or score_gap >= 4.0
        ):
            duration_action = 2
        elif (
            selected_metric["queue_count"] >= 2
            or selected_metric["vehicle_count"] >= 4
            or selected_metric["avg_wait"] >= 3.0
        ):
            duration_action = 1
        else:
            duration_action = 0

        if self.last_phase is not None and next_phase != self.last_phase:
            duration_action = max(duration_action, Config.MIN_SWITCH_DURATION_ACTION)

        if (
            self.last_phase is not None
            and next_phase == self.last_phase
            and score_gap >= 6.0
            and selected_metric["queue_count"] >= 4
        ):
            duration_action = min(duration_action + 1, Config.DIM_OF_ACTION_DURATION_1 - 1)

        if any(
            self.phase_idle_rounds[phase_id] >= Config.MAX_IDLE_ROUNDS and phase_metrics[phase_id]["vehicle_count"] > 0
            for phase_id in phase_metrics
            if phase_id != next_phase
        ):
            duration_action = min(duration_action, 2)

        return duration_action

    def _build_observation(self, phase_metrics, phase_scores, scene_waiting_vehicle_count):
        observation = []

        for phase_id in range(Config.DIM_OF_ACTION_PHASE_1):
            metric = phase_metrics[phase_id]
            observation.extend(
                [
                    min(metric["queue_count"] / 20.0, 1.0),
                    min(metric["vehicle_count"] / 20.0, 1.0),
                    min(metric["avg_wait"] / 20.0, 1.0),
                    min(metric["avg_delay"] / 25.0, 1.0),
                ]
            )

        for phase_id in range(Config.DIM_OF_ACTION_PHASE_1):
            observation.append(min(phase_scores[phase_id] / 30.0, 1.0))

        current_phase_one_hot = [0.0] * Config.DIM_OF_ACTION_PHASE_1
        if self.last_phase is not None:
            current_phase_one_hot[self.last_phase] = 1.0
        observation.extend(current_phase_one_hot)

        for phase_id in range(Config.DIM_OF_ACTION_PHASE_1):
            observation.append(min(self.phase_idle_rounds[phase_id] / max(Config.MAX_IDLE_ROUNDS, 1), 1.0))

        total_queue = sum(metric["queue_count"] for metric in phase_metrics.values())
        north_south_pressure = phase_scores[0] + phase_scores[1]
        east_west_pressure = phase_scores[2] + phase_scores[3]
        pressure_gap = np.clip((north_south_pressure - east_west_pressure) / 25.0, -1.0, 1.0)

        observation.extend(
            [
                min(total_queue / 40.0, 1.0),
                min(scene_waiting_vehicle_count / 300.0, 1.0),
                min(self.same_phase_count / 4.0, 1.0),
                float(pressure_gap),
            ]
        )

        return observation

    def _build_legal_action(self):
        return [1] * sum(Config.LEGAL_ACTION_SIZE_LIST)

    def _build_logit_bias(self, phase_scores, duration_action):
        phase_values = np.array([phase_scores[phase_id] for phase_id in range(Config.DIM_OF_ACTION_PHASE_1)], dtype=np.float32)
        if np.max(np.abs(phase_values)) <= 1e-6:
            phase_bias = np.zeros_like(phase_values)
        else:
            centered_scores = phase_values - np.mean(phase_values)
            score_std = np.std(phase_values) + 1e-6
            phase_bias = np.clip(centered_scores / score_std, -2.0, 2.0) * Config.RULE_PHASE_BIAS_SCALE

        duration_bias = np.full(Config.DIM_OF_ACTION_DURATION_1, -0.15, dtype=np.float32)
        duration_bias[duration_action] = Config.RULE_DURATION_BIAS
        if duration_action > 0:
            duration_bias[duration_action - 1] = max(duration_bias[duration_action - 1], 0.2)
        if duration_action < Config.DIM_OF_ACTION_DURATION_1 - 1:
            duration_bias[duration_action + 1] = max(duration_bias[duration_action + 1], 0.2)

        return list(phase_bias) + list(duration_bias)

    def _sample_masked_action(self, logits, legal_action):
        prob_list = []
        action_list = []
        d_action_list = []
        label_split_size = [sum(self.label_size_list[: index + 1]) for index in range(len(self.label_size_list))]
        legal_actions = np.split(legal_action, label_split_size[:-1])
        logits_split = np.split(logits, label_split_size[:-1])

        for index in range(len(self.label_size_list)):
            if np.count_nonzero(legal_actions[index]) == 0:
                probs = [0] * self.label_size_list[index]
                sample_action = 0
                d_action = 0
            else:
                probs = self._legal_soft_max(logits_split[index], legal_actions[index])
                sample_action = self._legal_sample(probs, use_max=False)
                d_action = self._legal_sample(probs, use_max=True)
            action_list.append(sample_action)
            d_action_list.append(d_action)
            prob_list += list(probs)

        return prob_list, action_list, d_action_list

    def _legal_soft_max(self, input_hidden, legal_action):
        _lsm_const_w = 1e20
        tmp = input_hidden - _lsm_const_w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -_lsm_const_w, 1)
        tmp = (np.exp(tmp) + 1e-5) * legal_action
        probs = tmp / np.sum(tmp, keepdims=True)
        return probs

    def _legal_sample(self, probs, legal_action=None, use_max=False):
        if use_max:
            return int(np.argmax(probs))
        return int(np.argmax(np.random.multinomial(1, probs, size=1)))

    def _get_lane_speed_limit(self, lane_id):
        lane_cfg = self.preprocess.lane_dict.get(lane_id, {})
        lane_speed_limit = lane_cfg.get("speed_limit", 10.0)
        return float(lane_speed_limit) if lane_speed_limit else 10.0
