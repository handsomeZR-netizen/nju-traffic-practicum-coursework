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
    BaseAgent,
    exploit_wrapper,
    learn_wrapper,
    load_model_wrapper,
    predict_wrapper,
    save_model_wrapper,
)
from kaiwu_agent.utils.common_func import attached

from agent_diy.algorithm.algorithm import Algorithm
from agent_diy.conf.conf import Config
from agent_diy.feature.definition import (
    ActData,
    FeatureProcess,
    ObsData,
    get_capped_vehicle_wait_time,
    get_vehicle_status,
    get_webster_lane_group,
    is_accident_vehicle,
    on_enter_lane,
)
from agent_diy.model.model import Model


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        torch.manual_seed(0)
        self.device = device
        self.model = Model(device).to(self.device)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
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
        self.phase_lanes = {int(phase_id): list(lane_ids) for phase_id, lane_ids in get_webster_lane_group().items()}
        self.phase_lane_map = {
            lane_id: int(phase_id)
            for phase_id, lane_ids in self.phase_lanes.items()
            for lane_id in lane_ids
        }
        self.enter_lane_ids = set(self.phase_lane_map.keys())
        self.reset()
        super().__init__(agent_type, device, logger, monitor)

    def reset(self):
        self.preprocess.reset()
        self.last_phase = None
        self.same_phase_count = 0
        self.last_score_gap = 0.0
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
                action, d_action = self._apply_rule_guidance(
                    action,
                    d_action,
                    list_obs_data[i].rule_action,
                )

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
        if self.logger:
            self.logger.info(f"save model {model_file_path} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        if not os.path.exists(model_file_path):
            if self.logger:
                self.logger.info(f"model file {model_file_path} not found, skip loading")
            return

        self.model.load_state_dict(torch.load(model_file_path, map_location=self.model.device))
        if self.logger:
            self.logger.info(f"load model {model_file_path} successfully")

    def observation_process(self, raw_obs, extra_info=None):
        self.preprocess.update_traffic_info(raw_obs, extra_info)
        frame_state = raw_obs["frame_state"]

        phase_metrics, scene_waiting_vehicle_count = self._collect_phase_metrics(frame_state)
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
        logit_bias = self._build_logit_bias(
            phase_metrics,
            phase_scores,
            phase_action,
            duration_action,
        )

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

    def _apply_rule_guidance(self, action, d_action, rule_action):
        rule_duration = int(rule_action[1])
        guided_action = list(action)
        guided_d_action = list(d_action)
        guided_action[1] = self._merge_duration_action(guided_action[1], rule_duration)
        guided_d_action[1] = self._merge_duration_action(guided_d_action[1], rule_duration)
        return guided_action, guided_d_action

    def _merge_duration_action(self, model_duration_action, rule_duration_action):
        merged_duration = int(np.clip(model_duration_action, 0, Config.DIM_OF_ACTION_DURATION_1 - 1))
        rule_duration = int(np.clip(rule_duration_action, 0, Config.DIM_OF_ACTION_DURATION_1 - 1))
        if merged_duration > rule_duration + 1:
            merged_duration = rule_duration + 1
        elif merged_duration < rule_duration - 1:
            merged_duration = rule_duration - 1
        if rule_duration >= Config.MIN_SWITCH_DURATION_ACTION:
            merged_duration = max(merged_duration, rule_duration - 1)
        return int(np.clip(merged_duration, 0, Config.DIM_OF_ACTION_DURATION_1 - 1))

    def _empty_phase_metric(self):
        return {
            "vehicle_count": 0,
            "queue_count": 0,
            "close_count": 0,
            "very_close_count": 0,
            "weighted_vehicle_count": 0.0,
            "weighted_queue_count": 0.0,
            "weighted_close_count": 0.0,
            "weighted_very_close_count": 0.0,
            "accident_count": 0,
            "irregular_count": 0,
            "wait_sum": 0.0,
            "delay_sum": 0.0,
            "weighted_wait_sum": 0.0,
            "weighted_delay_sum": 0.0,
            "serviceable_pressure": 0.0,
            "risk_pressure": 0.0,
            "lane_queue_pressure": 0.0,
            "avg_congestion": 0.0,
            "open_lane_weight": 0.0,
            "open_lane_ratio": 0.0,
            "blocked_ratio": 0.0,
            "avg_wait": 0.0,
            "avg_delay": 0.0,
            "max_wait": 0.0,
            "demand_pressure": 0.0,
            "release_penalty": 0.0,
            "effective_pressure": 0.0,
        }

    def _collect_phase_metrics(self, frame_state):
        metrics = {
            phase_id: self._empty_phase_metric()
            for phase_id in range(Config.DIM_OF_ACTION_PHASE_1)
        }

        vehicles = frame_state["vehicles"]
        lane_states = {}
        for lane in frame_state.get("lanes", []):
            lane_id = int(lane.get("lane_id", lane.get("l_id", -1)))
            if lane_id in self.enter_lane_ids:
                lane_states[lane_id] = lane

        scene_waiting_vehicle_count = 0
        lane_block = {lane_id: 0.0 for lane_id in self.enter_lane_ids}
        lane_risk = {lane_id: 0.0 for lane_id in self.enter_lane_ids}
        normal_entries = []

        for vehicle in vehicles:
            speed = float(vehicle.get("speed", 0.0))
            if speed <= 0.1 and not is_accident_vehicle(vehicle):
                scene_waiting_vehicle_count += 1

            if not on_enter_lane(vehicle):
                continue

            lane_id = int(vehicle["lane"])
            phase_id = self.phase_lane_map.get(lane_id)
            if phase_id is None:
                continue

            stop_line_distance = float(vehicle.get("position_in_lane", {}).get("y", Config.CLOSE_DISTANCE + 1))
            metric = metrics[phase_id]
            approach_gain = max(Config.CLOSE_DISTANCE - stop_line_distance, 0.0) / max(Config.CLOSE_DISTANCE, 1.0)
            status = get_vehicle_status(vehicle)

            if is_accident_vehicle(vehicle):
                metric["accident_count"] += 1
                lane_block[lane_id] = max(
                    lane_block[lane_id],
                    Config.BLOCKED_LANE_ACCIDENT_NEAR_LEVEL
                    if stop_line_distance <= Config.ACCIDENT_CLOSE_DISTANCE
                    else Config.BLOCKED_LANE_ACCIDENT_FAR_LEVEL,
                )
                continue

            if status == 2:
                metric["irregular_count"] += 1
                lane_block[lane_id] = max(
                    lane_block[lane_id],
                    Config.BLOCKED_LANE_IRREGULAR_NEAR_LEVEL
                    if stop_line_distance <= Config.VERY_CLOSE_DISTANCE
                    else Config.BLOCKED_LANE_IRREGULAR_FAR_LEVEL,
                )
                lane_risk[lane_id] = max(
                    lane_risk[lane_id],
                    Config.IRREGULAR_RISK_PRESSURE_COEF * (0.5 + approach_gain),
                )
                if stop_line_distance <= Config.VERY_CLOSE_DISTANCE:
                    lane_risk[lane_id] += Config.IRREGULAR_CLOSE_PRESSURE_COEF

            wait_time = get_capped_vehicle_wait_time(self, vehicle)
            speed_limit = self._get_lane_speed_limit(lane_id)
            speed_ratio = min(speed / max(speed_limit, 1e-6), 1.0)
            delay_value = float(vehicle.get("delay", wait_time + (1.0 - speed_ratio) * 5.0))

            metric["vehicle_count"] += 1
            metric["wait_sum"] += wait_time
            metric["delay_sum"] += delay_value
            metric["max_wait"] = max(metric["max_wait"], wait_time)

            if speed <= 0.3:
                metric["queue_count"] += 1
            if stop_line_distance <= Config.CLOSE_DISTANCE:
                metric["close_count"] += 1
            if stop_line_distance <= Config.VERY_CLOSE_DISTANCE:
                metric["very_close_count"] += 1

            normal_entries.append(
                {
                    "phase_id": phase_id,
                    "lane_id": lane_id,
                    "speed": speed,
                    "stop_line_distance": stop_line_distance,
                    "wait_time": wait_time,
                    "delay": delay_value,
                    "approach_gain": approach_gain,
                }
            )

        for phase_id, metric in metrics.items():
            lane_ids = self.phase_lanes[phase_id]
            blocked_sum = 0.0
            congestion_sum = 0.0
            queue_pressure = 0.0

            for lane_id in lane_ids:
                block_level = float(np.clip(lane_block.get(lane_id, 0.0), 0.0, 1.0))
                lane_weight = self._lane_weight(block_level)
                blocked_sum += block_level
                metric["open_lane_weight"] += lane_weight
                metric["risk_pressure"] += lane_risk.get(lane_id, 0.0)

                lane_state = lane_states.get(lane_id, {})
                queue_pressure += float(lane_state.get("queue_length", 0.0)) * lane_weight
                congestion_sum += float(lane_state.get("congestion", 0.0))

            lane_norm = max(len(lane_ids), 1)
            metric["blocked_ratio"] = blocked_sum / lane_norm
            metric["open_lane_ratio"] = metric["open_lane_weight"] / lane_norm
            metric["lane_queue_pressure"] = queue_pressure
            metric["avg_congestion"] = congestion_sum / lane_norm

        for entry in normal_entries:
            metric = metrics[entry["phase_id"]]
            block_level = float(np.clip(lane_block.get(entry["lane_id"], 0.0), 0.0, 1.0))
            lane_weight = self._lane_weight(block_level)

            metric["weighted_vehicle_count"] += lane_weight
            metric["weighted_wait_sum"] += entry["wait_time"] * lane_weight
            metric["weighted_delay_sum"] += entry["delay"] * lane_weight
            metric["serviceable_pressure"] += entry["approach_gain"] * lane_weight

            if entry["speed"] <= 0.3:
                metric["weighted_queue_count"] += lane_weight
            if entry["stop_line_distance"] <= Config.CLOSE_DISTANCE:
                metric["weighted_close_count"] += lane_weight
            if entry["stop_line_distance"] <= Config.VERY_CLOSE_DISTANCE:
                metric["weighted_very_close_count"] += lane_weight

        for metric in metrics.values():
            raw_vehicle_count = max(metric["vehicle_count"], 1)
            weighted_vehicle_count = max(metric["weighted_vehicle_count"], 1.0)
            metric["avg_wait"] = (
                metric["weighted_wait_sum"] / weighted_vehicle_count
                if metric["weighted_vehicle_count"] > 0
                else metric["wait_sum"] / raw_vehicle_count
            )
            metric["avg_delay"] = (
                metric["weighted_delay_sum"] / weighted_vehicle_count
                if metric["weighted_vehicle_count"] > 0
                else metric["delay_sum"] / raw_vehicle_count
            )

        return metrics, scene_waiting_vehicle_count

    def _build_phase_scores(self, phase_metrics):
        phase_scores = {}
        for phase_id, metric in phase_metrics.items():
            idle_bonus = min(self.phase_idle_rounds[phase_id], Config.MAX_IDLE_ROUNDS) * Config.IDLE_BONUS_COEF
            keep_bonus = 0.0
            if self.last_phase == phase_id:
                keep_bonus = min(self.same_phase_count, Config.MIN_PHASE_HOLD_ROUNDS) * Config.KEEP_PHASE_BONUS

            demand_pressure = (
                metric["weighted_close_count"] * Config.CLOSE_PRESSURE_COEF
                + metric["weighted_very_close_count"] * Config.VERY_CLOSE_PRESSURE_COEF
                + metric["weighted_queue_count"] * Config.QUEUE_PRESSURE_COEF
                + metric["avg_wait"] * Config.WAIT_PRESSURE_COEF
                + metric["weighted_vehicle_count"] * Config.VEHICLE_PRESSURE_COEF
                + metric["avg_delay"] * Config.DELAY_PRESSURE_COEF
                + metric["max_wait"] * Config.MAX_WAIT_PRESSURE_COEF
                + metric["lane_queue_pressure"] * Config.LANE_QUEUE_PRESSURE_COEF
                + metric["serviceable_pressure"] * Config.APPROACH_PRESSURE_COEF
            )
            release_penalty = (
                metric["avg_congestion"] * Config.CONGESTION_PRESSURE_COEF
                + metric["blocked_ratio"] * Config.BLOCKED_PHASE_PENALTY
            )
            effective_pressure = (
                demand_pressure
                - release_penalty
                + metric["open_lane_ratio"] * Config.OPEN_LANE_BONUS_COEF
                + metric["risk_pressure"] * 0.2
            )
            metric["demand_pressure"] = demand_pressure
            metric["release_penalty"] = release_penalty
            metric["effective_pressure"] = effective_pressure

            score = effective_pressure + idle_bonus + keep_bonus
            if (
                metric["weighted_vehicle_count"] <= Config.MIN_ACTIVE_WEIGHT
                and metric["lane_queue_pressure"] < Config.PHASE_LANE_QUEUE_LOW
                and metric["demand_pressure"] < Config.PRESSURE_LOW_THRESHOLD
            ):
                score *= 0.15
            elif (
                metric["blocked_ratio"] >= Config.HARD_BLOCKED_RATIO
                and metric["demand_pressure"] < Config.PRESSURE_MID_THRESHOLD
                and metric["weighted_close_count"] < 1.5
            ):
                score *= 0.7
            phase_scores[phase_id] = score
        return phase_scores

    def _build_signal_plan(self, phase_metrics, phase_scores, scene_waiting_vehicle_count):
        ranked_phases = sorted(
            phase_metrics.keys(),
            key=lambda phase_id: (
                phase_scores[phase_id],
                phase_metrics[phase_id]["effective_pressure"],
                phase_metrics[phase_id]["avg_delay"],
                phase_metrics[phase_id]["weighted_close_count"],
                phase_metrics[phase_id]["lane_queue_pressure"],
            ),
            reverse=True,
        )
        active_phases = [
            phase_id
            for phase_id in ranked_phases
            if phase_metrics[phase_id]["demand_pressure"] >= Config.PRESSURE_LOW_THRESHOLD
            or phase_metrics[phase_id]["weighted_vehicle_count"] > Config.MIN_ACTIVE_WEIGHT
            or phase_metrics[phase_id]["lane_queue_pressure"] > 1.5
        ]

        if not active_phases:
            next_phase = 0 if self.last_phase is None else self.last_phase
            self.last_score_gap = 0.0
        else:
            best_phase = active_phases[0]
            if self.last_phase is None:
                next_phase = best_phase
                self.last_score_gap = phase_metrics[best_phase]["effective_pressure"]
            else:
                current_phase = self.last_phase
                current_metric = phase_metrics[current_phase]
                best_metric = phase_metrics[best_phase]
                current_score = phase_scores[current_phase]
                best_score = phase_scores[best_phase]
                current_pressure = current_metric["effective_pressure"]
                best_pressure = best_metric["effective_pressure"]
                pressure_gap = best_pressure - current_pressure
                self.last_score_gap = pressure_gap
                force_switch_phase = None
                alternate_phase = next(
                    (
                        phase_id
                        for phase_id in active_phases
                        if phase_id != current_phase
                        and phase_metrics[phase_id]["demand_pressure"] >= Config.PRESSURE_LOW_THRESHOLD
                    ),
                    None,
                )

                keep_current = best_phase == current_phase
                if not keep_current:
                    if (
                        self.same_phase_count < Config.MIN_PHASE_HOLD_ROUNDS
                        and current_metric["demand_pressure"] >= Config.SOTL_MIN_GREEN_PRESSURE
                    ):
                        keep_current = True
                    elif (
                        current_metric["demand_pressure"] >= Config.SOTL_KEEP_GREEN_PRESSURE
                        and pressure_gap <= Config.SOTL_RED_PRESSURE_MARGIN
                    ):
                        keep_current = True
                    elif (
                        current_metric["demand_pressure"] >= Config.SOTL_MIN_GREEN_PRESSURE
                        and best_pressure < current_pressure * Config.SOTL_FORCE_SWITCH_RATIO
                        and pressure_gap < Config.SOTL_FORCE_SWITCH_GAP
                    ):
                        keep_current = True
                    elif (
                        best_score <= current_score + Config.KEEP_SCORE_MARGIN
                        and current_metric["weighted_close_count"] > 0.75
                    ):
                        keep_current = True
                    elif (
                        current_score >= best_score * Config.KEEP_SCORE_RATIO
                        and current_metric["weighted_close_count"] > 0.75
                    ):
                        keep_current = True

                if (
                    self.same_phase_count >= Config.MAX_PHASE_HOLD_ROUNDS
                    and alternate_phase is not None
                ):
                    keep_current = False
                    force_switch_phase = alternate_phase

                if (
                    current_metric["blocked_ratio"] >= Config.HIGH_BLOCKED_RATIO
                    and best_metric["blocked_ratio"] + 0.1 < current_metric["blocked_ratio"]
                    and best_pressure > current_pressure
                ):
                    keep_current = False

                starved_phase = next(
                    (
                        phase_id
                        for phase_id in active_phases
                        if phase_id != current_phase
                        and self.phase_idle_rounds[phase_id] >= Config.FORCE_SWITCH_IDLE_ROUNDS
                        and phase_metrics[phase_id]["demand_pressure"] >= Config.PRESSURE_LOW_THRESHOLD
                        and phase_scores[phase_id] + Config.FORCE_SWITCH_PRESSURE_GAP >= current_score
                    ),
                    None,
                )
                if starved_phase is not None and self.same_phase_count >= Config.MIN_PHASE_HOLD_ROUNDS:
                    force_switch_phase = starved_phase
                    keep_current = False
                if (
                    current_metric["weighted_vehicle_count"] <= Config.MIN_ACTIVE_WEIGHT
                    and best_metric["weighted_vehicle_count"] > Config.MIN_ACTIVE_WEIGHT
                ):
                    keep_current = False
                if (
                    best_phase != current_phase
                    and self.same_phase_count >= Config.MIN_PHASE_HOLD_ROUNDS
                    and best_pressure >= current_pressure + Config.FORCE_SWITCH_PRESSURE_GAP
                ):
                    keep_current = False
                    force_switch_phase = best_phase
                if (
                    scene_waiting_vehicle_count >= Config.HIGH_SCENE_WAITING_THRESHOLD
                    and best_phase != current_phase
                    and best_pressure >= current_pressure + Config.SOTL_RED_PRESSURE_MARGIN
                    and best_metric["open_lane_ratio"] >= current_metric["open_lane_ratio"] - 0.1
                ):
                    keep_current = False

                next_phase = current_phase if keep_current else (force_switch_phase or best_phase)

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
        second_pressure = phase_metrics[second_phase]["effective_pressure"] if second_phase is not None else 0.0
        pressure_gap = selected_metric["effective_pressure"] - second_pressure
        demand_pressure = selected_metric["demand_pressure"]
        near_empty = (
            selected_metric["weighted_vehicle_count"] <= Config.MIN_ACTIVE_WEIGHT
            and selected_metric["weighted_close_count"] < 1.0
            and selected_metric["lane_queue_pressure"] < Config.PHASE_LANE_QUEUE_LOW
            and demand_pressure < Config.PRESSURE_LOW_THRESHOLD
        )

        if near_empty:
            duration_action = 0
        elif (
            selected_metric["blocked_ratio"] >= Config.HARD_BLOCKED_RATIO
            and demand_pressure < Config.PRESSURE_HIGH_THRESHOLD
        ):
            duration_action = 1 if selected_metric["weighted_close_count"] >= 1.0 else 0
        elif (
            demand_pressure >= Config.PRESSURE_HIGH_THRESHOLD
            or (
                scene_waiting_vehicle_count >= Config.HIGH_SCENE_WAITING_THRESHOLD
                and demand_pressure >= Config.PRESSURE_MID_THRESHOLD
            )
            or selected_metric["max_wait"] >= Config.EMERGENCY_MAX_WAIT_THRESHOLD
        ):
            duration_action = 3
        elif (
            demand_pressure >= Config.PRESSURE_MID_THRESHOLD
            or selected_metric["lane_queue_pressure"] >= Config.PHASE_LANE_QUEUE_MID
            or selected_metric["avg_delay"] >= 8.0
            or pressure_gap >= Config.SOTL_RED_PRESSURE_MARGIN
        ):
            duration_action = 2
        else:
            duration_action = 1

        if self.last_phase is None or next_phase != self.last_phase:
            duration_action = max(duration_action, Config.MIN_SWITCH_DURATION_ACTION)
        if (
            self.last_phase is not None
            and next_phase == self.last_phase
            and self.same_phase_count < Config.MIN_PHASE_HOLD_ROUNDS
            and demand_pressure >= Config.SOTL_MIN_GREEN_PRESSURE
        ):
            duration_action = max(duration_action, Config.MIN_SWITCH_DURATION_ACTION)
        if (
            self.last_phase is not None
            and next_phase == self.last_phase
            and demand_pressure >= Config.PRESSURE_HIGH_THRESHOLD
            and selected_metric["blocked_ratio"] < Config.HIGH_BLOCKED_RATIO
        ):
            duration_action = max(duration_action, 2)
        if (
            self.last_phase is not None
            and next_phase == self.last_phase
            and self.same_phase_count >= Config.MAX_PHASE_HOLD_ROUNDS - 1
        ):
            duration_action = min(duration_action, 2)
        if any(
            self.phase_idle_rounds[phase_id] >= Config.FORCE_SWITCH_IDLE_ROUNDS
            and phase_metrics[phase_id]["demand_pressure"] >= Config.PRESSURE_LOW_THRESHOLD
            for phase_id in phase_metrics
            if phase_id != next_phase
        ):
            duration_action = min(duration_action, 2)
        if (
            selected_metric["blocked_ratio"] >= Config.HIGH_BLOCKED_RATIO
            and demand_pressure < Config.PRESSURE_HIGH_THRESHOLD
        ):
            duration_action = min(duration_action, Config.BLOCKED_PHASE_MAX_DURATION_ACTION)
        if (
            selected_metric["irregular_count"] > 0
            and demand_pressure >= Config.PRESSURE_MID_THRESHOLD
            and selected_metric["blocked_ratio"] < Config.HIGH_BLOCKED_RATIO
        ):
            duration_action = max(duration_action, Config.IRREGULAR_RISK_MIN_DURATION_ACTION)
        return duration_action

    def _build_observation(self, phase_metrics, phase_scores, scene_waiting_vehicle_count):
        observation = []
        for phase_id in range(Config.DIM_OF_ACTION_PHASE_1):
            metric = phase_metrics[phase_id]
            observation.extend(
                [
                    self._normalize_linear(metric["weighted_close_count"], Config.OBS_CLOSE_NORM),
                    self._normalize_log(metric["lane_queue_pressure"], Config.OBS_QUEUE_NORM),
                    self._normalize_linear(metric["avg_delay"], Config.OBS_DELAY_NORM),
                    self._normalize_log(metric["weighted_vehicle_count"], Config.OBS_VEHICLE_NORM),
                ]
            )

        for phase_id in range(Config.DIM_OF_ACTION_PHASE_1):
            observation.append(self._normalize_linear(phase_scores[phase_id], Config.OBS_PHASE_SCORE_NORM))

        current_phase_one_hot = [0.0] * Config.DIM_OF_ACTION_PHASE_1
        if self.last_phase is not None:
            current_phase_one_hot[self.last_phase] = 1.0
        observation.extend(current_phase_one_hot)

        for phase_id in range(Config.DIM_OF_ACTION_PHASE_1):
            observation.append(min(self.phase_idle_rounds[phase_id] / max(Config.MAX_IDLE_ROUNDS, 1), 1.0))

        total_weighted_queue = sum(metric["weighted_queue_count"] for metric in phase_metrics.values())
        observation.extend(
            [
                self._normalize_linear(total_weighted_queue, Config.OBS_TOTAL_CLOSE_NORM),
                self._normalize_linear(scene_waiting_vehicle_count, Config.OBS_SCENE_WAITING_NORM),
                self._normalize_linear(self.same_phase_count, Config.OBS_SAME_PHASE_NORM),
                float(np.clip(self.last_score_gap / Config.OBS_SCORE_GAP_NORM, -1.0, 1.0)),
            ]
        )
        return observation

    def _build_legal_action(self):
        return [1] * sum(Config.LEGAL_ACTION_SIZE_LIST)

    def _build_logit_bias(self, phase_metrics, phase_scores, phase_action, duration_action):
        phase_values = np.array(
            [phase_scores[phase_id] for phase_id in range(Config.DIM_OF_ACTION_PHASE_1)],
            dtype=np.float32,
        )
        if np.max(np.abs(phase_values)) <= 1e-6:
            phase_bias = np.zeros_like(phase_values)
        else:
            centered_scores = phase_values - np.mean(phase_values)
            score_std = np.std(phase_values) + 1e-6
            phase_bias = np.clip(centered_scores / score_std, -2.0, 2.0) * Config.RULE_PHASE_BIAS_SCALE
        phase_bias = phase_bias - np.array(
            [
                phase_metrics[phase_id]["blocked_ratio"] * Config.BLOCKED_PHASE_BIAS_PENALTY
                for phase_id in range(Config.DIM_OF_ACTION_PHASE_1)
            ],
            dtype=np.float32,
        )
        phase_bias[phase_action] += Config.RULE_SELECTED_PHASE_BONUS

        duration_bias = np.full(Config.DIM_OF_ACTION_DURATION_1, -0.2, dtype=np.float32)
        duration_bias[duration_action] = Config.RULE_DURATION_BIAS
        if duration_action > 0:
            duration_bias[duration_action - 1] = max(duration_bias[duration_action - 1], 0.2)
        if duration_action < Config.DIM_OF_ACTION_DURATION_1 - 1:
            duration_bias[duration_action + 1] = max(duration_bias[duration_action + 1], 0.2)
        if phase_metrics[phase_action]["blocked_ratio"] >= Config.HIGH_BLOCKED_RATIO:
            duration_bias[-1] = min(duration_bias[-1], -0.35)
        return list(phase_bias) + list(duration_bias)

    def _normalize_linear(self, value, scale):
        return float(np.clip(float(value) / max(float(scale), 1e-6), 0.0, 1.0))

    def _normalize_log(self, value, scale):
        safe_value = max(float(value), 0.0)
        safe_scale = max(float(scale), 1.0)
        return float(np.clip(np.log1p(safe_value) / np.log1p(safe_scale), 0.0, 1.0))

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
        lsm_const = 1e20
        tmp = input_hidden - lsm_const * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -lsm_const, 1)
        tmp = (np.exp(tmp) + 1e-5) * legal_action
        probs = tmp / np.sum(tmp, keepdims=True)
        return probs

    def _legal_sample(self, probs, use_max=False):
        if use_max:
            return int(np.argmax(probs))
        return int(np.argmax(np.random.multinomial(1, probs, size=1)))

    def _get_lane_speed_limit(self, lane_id):
        lane_cfg = self.preprocess.lane_dict.get(lane_id, {})
        lane_speed_limit = lane_cfg.get("speed_limit", 10.0)
        return float(lane_speed_limit) if lane_speed_limit else 10.0

    def _lane_weight(self, block_level):
        return max(0.15, 1.0 - block_level)
