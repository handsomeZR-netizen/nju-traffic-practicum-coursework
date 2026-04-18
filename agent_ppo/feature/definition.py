#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import math
import numpy as np
from kaiwu_agent.utils.common_func import create_cls, attached
from agent_ppo.conf.conf import Config


def on_enter_lane(vehicle):
    """
    Determine whether the vehicle is on an entering lane.
    判断车辆是否位于进口车道上。
    """
    lane_id = vehicle["lane"]
    inlane_code = {
        11: 0,
        10: 1,
        9: 2,
        8: 3,
        129: 4,
        128: 5,
        127: 6,
        126: 7,
        23: 8,
        22: 9,
        21: 10,
        20: 11,
        163: 12,
        162: 13,
    }
    return lane_id in inlane_code and vehicle["target_junction"] != -1


def in_junction(vehicle):
    """
    Determine whether the vehicle is in the junction.
    判断车辆是否位于交叉口中。
    """
    return vehicle["junction"] != -1


def on_depart_lane(vehicle):
    """
    Determine whether the vehicle is on a departing lane.
    判断车辆是否位于出口车道上。
    """
    junction = vehicle["junction"]
    target_junction = vehicle["target_junction"]
    if (on_enter_lane(vehicle) or in_junction(vehicle)) or (junction == -1 and target_junction != -1):
        return False
    return True


def get_lane_code(vehicle):
    """
    Map lane id to the predefined entering-lane code.
    车道 id 映射到预定义的进口车道编号。
    """
    lane_id = vehicle["lane"]
    lane_code = {
        11: 0,
        10: 1,
        9: 2,
        8: 3,
        129: 4,
        128: 5,
        127: 6,
        126: 7,
        23: 8,
        22: 9,
        21: 10,
        20: 11,
        163: 12,
        162: 13,
    }
    return lane_code.get(lane_id)


def get_webster_lane_group():
    """
    Group lanes by green-light phase.
    根据绿灯相位对进口车道分组。
    """
    return {
        "0": [11, 10, 9, 23, 22, 21],
        "1": [8, 20],
        "2": [129, 128, 127, 163],
        "3": [126, 162],
    }


class FeatureProcess:
    """
    Update traffic information and provide cached statistics.
    更新交通信息并维护缓存统计量。
    """

    def __init__(self, logger):
        self.logger = logger
        self.reset()

    def reset(self):
        self.junction_dict = {}
        self.edge_dict = {}
        self.lane_dict = {}
        self.l_id_to_index = {}
        self.vehicle_configs_dict = {}

        self.vehicle_prev_junction = {}
        self.vehicle_prev_position = {}
        self.vehicle_distance_store = {}
        self.last_waiting_moment = {}
        self.waiting_time_store = {}
        self.enter_lane_time = {}

        self.lane_volume = {}
        self.last_enter_lane_vehicle_ids = set()
        self.departed_vehicle_count = 0
        self.scene_waiting_vehicle_count = 0

        self.old_waiting_time = 0
        self.last_reward_metrics = None
        self.prev_action_phase = None

    def init_road_info(self, start_info):
        junctions, signals, edges = (
            start_info["junctions"],
            start_info["signals"],
            start_info["edges"],
        )
        lane_configs, vehicle_configs = (
            start_info["lane_configs"],
            start_info["vehicle_configs"],
        )

        for junction in junctions:
            self.junction_dict[junction["j_id"]] = junction
            self.l_id_to_index[junction["j_id"]] = {}

            index = 0
            for approaching_edges in junction["enter_lanes_on_directions"]:
                for lane in approaching_edges["lanes"]:
                    self.l_id_to_index[junction["j_id"]][lane] = index
                    index += 1

            for edge in edges:
                self.edge_dict[edge["e_id"]] = edge
            for lane in lane_configs:
                self.lane_dict[lane["l_id"]] = lane
            for vehicle_config in vehicle_configs:
                self.vehicle_configs_dict[vehicle_config["v_config_id"]] = vehicle_config
            for lane in lane_configs:
                self.lane_volume[lane["l_id"]] = []

    def update_traffic_info(self, raw_obs, extra_info):
        frame_state = raw_obs["frame_state"]
        frame_no = frame_state["frame_no"]
        frame_time, vehicles = frame_state["frame_time"], frame_state["vehicles"]

        if frame_no <= 1:
            game_info = extra_info["init_state"]
            self.init_road_info(game_info)

        for lane_vehicle_ids in self.lane_volume.values():
            lane_vehicle_ids.clear()

        current_enter_lane_vehicle_ids = set()
        scene_waiting_vehicle_count = 0

        for vehicle in vehicles:
            if vehicle.get("speed", 0.0) <= 0.1:
                scene_waiting_vehicle_count += 1

            if on_enter_lane(vehicle):
                current_enter_lane_vehicle_ids.add(vehicle["v_id"])

            if vehicle["v_id"] not in self.vehicle_prev_junction:
                self.vehicle_prev_junction[vehicle["v_id"]] = vehicle["junction"]

            if (
                self.vehicle_prev_junction[vehicle["v_id"]] == -1
                and on_enter_lane(vehicle)
                and vehicle["v_id"] not in self.enter_lane_time
            ):
                self.enter_lane_time[vehicle["v_id"]] = frame_time
            elif self.vehicle_prev_junction[vehicle["v_id"]] != vehicle["junction"]:
                if self.vehicle_prev_junction[vehicle["v_id"]] != -1 and on_enter_lane(vehicle):
                    self.enter_lane_time[vehicle["v_id"]] = frame_time

            self.cal_waiting_time(frame_time, vehicle)
            self.cal_travel_distance(vehicle)
            self.cal_v_num_in_lane(vehicle)

        self.departed_vehicle_count = len(self.last_enter_lane_vehicle_ids - current_enter_lane_vehicle_ids)
        self.last_enter_lane_vehicle_ids = current_enter_lane_vehicle_ids
        self.scene_waiting_vehicle_count = scene_waiting_vehicle_count

    def cal_waiting_time(self, frame_time, vehicle):
        waiting_time = 0
        if on_enter_lane(vehicle):
            if vehicle["speed"] <= 0.1:
                if vehicle["v_id"] not in self.last_waiting_moment:
                    self.last_waiting_moment[vehicle["v_id"]] = frame_time
                    if vehicle["v_id"] not in self.waiting_time_store:
                        self.waiting_time_store[vehicle["v_id"]] = 0
                else:
                    waiting_time = frame_time - self.last_waiting_moment[vehicle["v_id"]]
                    self.waiting_time_store[vehicle["v_id"]] += waiting_time
                    self.last_waiting_moment[vehicle["v_id"]] = frame_time
            else:
                if vehicle["v_id"] in self.last_waiting_moment:
                    del self.last_waiting_moment[vehicle["v_id"]]
        else:
            if vehicle["v_id"] in self.waiting_time_store:
                del self.waiting_time_store[vehicle["v_id"]]
            if vehicle["v_id"] in self.last_waiting_moment:
                del self.last_waiting_moment[vehicle["v_id"]]

    def cal_travel_distance(self, vehicle):
        if on_enter_lane(vehicle):
            if self.vehicle_prev_junction[vehicle["v_id"]] != -1 and vehicle["v_id"] in self.vehicle_distance_store:
                del self.vehicle_distance_store[vehicle["v_id"]]
            if vehicle["v_id"] not in self.vehicle_distance_store:
                self.vehicle_distance_store[vehicle["v_id"]] = 0
                self.vehicle_prev_position[vehicle["v_id"]] = [
                    vehicle["position_in_lane"]["x"],
                    vehicle["position_in_lane"]["y"],
                ]
            else:
                if vehicle["v_id"] in self.vehicle_distance_store and vehicle["v_id"] in self.vehicle_prev_position:
                    self.vehicle_distance_store[vehicle["v_id"]] += math.sqrt(
                        math.pow(
                            vehicle["position_in_lane"]["x"] - self.vehicle_prev_position[vehicle["v_id"]][0],
                            2,
                        )
                        + math.pow(
                            vehicle["position_in_lane"]["y"] - self.vehicle_prev_position[vehicle["v_id"]][1],
                            2,
                        )
                    )
            self.vehicle_prev_position[vehicle["v_id"]] = [
                vehicle["position_in_lane"]["x"],
                vehicle["position_in_lane"]["y"],
            ]
        else:
            if vehicle["v_id"] in self.vehicle_prev_position:
                del self.vehicle_prev_position[vehicle["v_id"]]

    def cal_v_num_in_lane(self, vehicle):
        if on_enter_lane(vehicle):
            if vehicle["v_id"] not in self.lane_volume[vehicle["lane"]]:
                self.lane_volume[vehicle["lane"]].append(vehicle["v_id"])

        self.vehicle_prev_junction[vehicle["v_id"]] = vehicle["junction"]

    def get_all_junction_waiting_time(self, vehicles: list):
        res = {}
        v_num = {}
        for junction_id in self.junction_dict:
            res[junction_id] = 0
            v_num[junction_id] = 0
        for vehicle in vehicles:
            if vehicle["junction"] != -1 or vehicle["target_junction"] == -1:
                continue
            if vehicle["v_id"] in self.waiting_time_store:
                t = self.waiting_time_store[vehicle["v_id"]]
            else:
                t = 0
            res[vehicle["target_junction"]] += t
            v_num[vehicle["target_junction"]] += 1
        for junction_id in self.junction_dict:
            if v_num[junction_id] != 0:
                res[junction_id] /= v_num[junction_id]
        return res

    def get_all_junction_waiting_time_by_origin(self, vehicles: list):
        res = {}
        v_num = {}
        for junction_id in self.junction_dict:
            res[junction_id] = 0
            v_num[junction_id] = 0
        for vehicle in vehicles:
            if vehicle["junction"] != -1 or vehicle["target_junction"] == -1:
                continue
            res[vehicle["target_junction"]] += vehicle["waiting_time"]
            v_num[vehicle["target_junction"]] += 1
        for junction_id in self.junction_dict:
            if v_num[junction_id] != 0:
                res[junction_id] /= v_num[junction_id]
        return res


ObsData = create_cls(
    "ObsData",
    feature=None,
    legal_action=None,
    sub_action_mask=None,
    rule_action=None,
    logit_bias=None,
)

ActData = create_cls("ActData", junction_id=None, action=None, d_action=None, prob=None, value=None)

SampleData = create_cls(
    "SampleData",
    obs=None,
    legal_action=None,
    act=None,
    reward=None,
    reward_sum=None,
    done=None,
    value=None,
    next_value=None,
    advantage=None,
    prob=None,
    sub_action=None,
    is_train=None,
)


@attached
def sample_process(list_sample_data):
    for i in range(len(list_sample_data) - 1):
        list_sample_data[i].next_value = list_sample_data[i + 1].value

    _calc_reward(list_sample_data)

    return list_sample_data


def reward_shaping(_obs, act, agent):
    """
    Build a dense reward aligned with queue reduction, delay reduction and timeout risk control.
    构造一个兼顾排队下降、延误下降和超时风险控制的稠密奖励。
    """

    current_metrics = _build_reward_metrics(_obs, agent)
    previous_metrics = agent.preprocess.last_reward_metrics

    if previous_metrics is None:
        agent.preprocess.last_reward_metrics = current_metrics
        if act is not None and len(act) >= 2:
            agent.preprocess.prev_action_phase = act[1]
        return 0.0

    wait_improve = previous_metrics["avg_wait"] - current_metrics["avg_wait"]
    queue_improve = previous_metrics["queue_count"] - current_metrics["queue_count"]
    delay_improve = previous_metrics["avg_delay"] - current_metrics["avg_delay"]
    speed_improve = current_metrics["avg_speed_ratio"] - previous_metrics["avg_speed_ratio"]
    throughput_gain = current_metrics["throughput"]

    risk_penalty = max(0.0, current_metrics["scene_waiting"] - 220.0) / 25.0
    switch_penalty = 0.0
    if (
        act is not None
        and len(act) >= 2
        and agent.preprocess.prev_action_phase is not None
        and act[1] != agent.preprocess.prev_action_phase
    ):
        switch_penalty = 0.25

    reward = (
        1.2 * wait_improve
        + 1.6 * queue_improve
        + 1.0 * delay_improve
        + 0.6 * speed_improve
        + 0.3 * throughput_gain
        - 1.8 * risk_penalty
        - switch_penalty
    )

    if current_metrics["enter_lane_vehicle_count"] == 0 and current_metrics["queue_count"] == 0:
        reward += 0.5

    reward = float(np.clip(reward, -20.0, 20.0))
    agent.preprocess.last_reward_metrics = current_metrics
    if act is not None and len(act) >= 2:
        agent.preprocess.prev_action_phase = act[1]

    return reward


@attached
def SampleData2NumpyData(g_data):
    return np.hstack(
        (
            np.array(g_data.obs, dtype=np.float32),
            np.array(g_data.legal_action, dtype=np.float32),
            np.array(g_data.sub_action, dtype=np.float32),
            np.array(g_data.act, dtype=np.float32),
            np.array(g_data.prob, dtype=np.float32),
            np.array(g_data.reward, dtype=np.float32),
            np.array(g_data.reward_sum, dtype=np.float32),
            np.array(g_data.advantage, dtype=np.float32),
            np.array(g_data.value, dtype=np.float32),
            np.array(g_data.next_value, dtype=np.float32),
            np.array(g_data.done, dtype=np.float32),
            np.array(g_data.is_train, dtype=np.float32),
        )
    )


@attached
def NumpyData2SampleData(s_data):
    obs_dim = Config.DIM_OF_OBSERVATION
    legal_action_dim = sum(Config.LEGAL_ACTION_SIZE_LIST)
    sub_action_dim = Config.NUMB_HEAD
    act_dim = Config.NUMB_HEAD
    prob_dim = legal_action_dim

    cursor = 0
    obs = s_data[cursor : cursor + obs_dim]
    cursor += obs_dim

    legal_action = s_data[cursor : cursor + legal_action_dim]
    cursor += legal_action_dim

    sub_action = s_data[cursor : cursor + sub_action_dim]
    cursor += sub_action_dim

    act = s_data[cursor : cursor + act_dim]
    cursor += act_dim

    prob = s_data[cursor : cursor + prob_dim]
    cursor += prob_dim

    reward = s_data[cursor]
    cursor += 1
    reward_sum = s_data[cursor]
    cursor += 1
    advantage = s_data[cursor]
    cursor += 1
    value = s_data[cursor]
    cursor += 1
    next_value = s_data[cursor]
    cursor += 1
    done = s_data[cursor]
    cursor += 1
    is_train = s_data[cursor]

    return SampleData(
        obs=obs,
        legal_action=legal_action,
        sub_action=sub_action,
        act=act,
        prob=prob,
        reward=reward,
        reward_sum=reward_sum,
        advantage=advantage,
        value=value,
        next_value=next_value,
        done=done,
        is_train=is_train,
    )


def _calc_reward(list_sample_data):
    """
    Calculate cumulated reward and advantage with GAE.
    reward_sum: used for value loss
    advantage: used for policy loss
    """

    gae = 0.0
    gamma, lamda = Config.GAMMA, Config.LAMDA
    for rl_info in reversed(list_sample_data):
        delta = -rl_info.value + rl_info.reward + gamma * rl_info.next_value
        gae = gae * gamma * lamda + delta
        rl_info.advantage = gae
        rl_info.reward_sum = gae + rl_info.value


def _build_reward_metrics(raw_obs, agent):
    vehicles = raw_obs["frame_state"]["vehicles"]
    enter_lane_vehicle_ids = set()

    total_wait = 0.0
    total_delay = 0.0
    total_speed_ratio = 0.0
    queue_count = 0
    scene_waiting = 0
    enter_lane_vehicle_count = 0

    for vehicle in vehicles:
        speed = float(vehicle.get("speed", 0.0))
        if speed <= 0.1:
            scene_waiting += 1

        if not on_enter_lane(vehicle):
            continue

        lane_id = vehicle["lane"]
        speed_limit = _get_lane_speed_limit(agent, lane_id)
        wait_time = float(vehicle.get("waiting_time", 0.0))
        speed_ratio = min(speed / max(speed_limit, 1e-6), 1.0)
        delay_proxy = wait_time + (1.0 - speed_ratio) * 5.0

        enter_lane_vehicle_ids.add(vehicle["v_id"])
        enter_lane_vehicle_count += 1
        total_wait += wait_time
        total_delay += delay_proxy
        total_speed_ratio += speed_ratio
        if speed <= 0.3:
            queue_count += 1

    norm = max(enter_lane_vehicle_count, 1)
    throughput = len(agent.preprocess.last_enter_lane_vehicle_ids - enter_lane_vehicle_ids)

    return {
        "avg_wait": total_wait / norm,
        "avg_delay": total_delay / norm,
        "avg_speed_ratio": total_speed_ratio / norm,
        "queue_count": float(queue_count),
        "scene_waiting": float(scene_waiting),
        "throughput": float(throughput),
        "enter_lane_vehicle_count": float(enter_lane_vehicle_count),
    }


def _get_lane_speed_limit(agent, lane_id):
    lane_cfg = agent.preprocess.lane_dict.get(lane_id, {})
    lane_speed_limit = lane_cfg.get("speed_limit", 10.0)
    return float(lane_speed_limit) if lane_speed_limit else 10.0
