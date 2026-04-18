#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import time

from kaiwu_agent.utils.common_func import attached

from agent_diy.feature.definition import *
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf


@attached
def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    env, agent = envs[0], agents[0]
    epoch_num = 100000
    episode_num_every_epoch = 1
    last_save_model_time = 0

    monitor_data = {
        "reward": 0,
        "diy_1": 0,
        "diy_2": 0,
        "diy_3": 0,
        "diy_4": 0,
        "diy_5": 0,
    }
    last_report_monitor_time = time.time()

    usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
        return

    for epoch in range(epoch_num):
        epoch_total_rew = 0
        data_length = 0
        for g_data in run_episodes(episode_num_every_epoch, env, agent, usr_conf, logger):
            data_length += len(g_data)
            epoch_total_rew += sum(data.reward for data in g_data)
            agent.learn(g_data)
            g_data.clear()

        avg_step_reward = 0
        if data_length:
            avg_step_reward = f"{(epoch_total_rew / data_length):.2f}"

        now = time.time()
        if now - last_save_model_time >= 600:
            agent.save_model()
            last_save_model_time = now

        if now - last_report_monitor_time > 60:
            monitor_data["reward"] = avg_step_reward
            if monitor:
                monitor.put_data({os.getpid(): monitor_data})
                last_report_monitor_time = now

        logger.info(f"Avg Step Reward: {avg_step_reward}, Epoch: {epoch}, Data Length: {data_length}")


def run_episodes(n_episode, env, agent, usr_conf, logger):
    try:
        train_test_quick_stop = os.environ.get("is_train_test", "False").lower() == "true"
        for _ in range(n_episode):
            collector = []

            training_metrics = get_training_metrics()
            if training_metrics:
                logger.info(f"training_metrics is {training_metrics}")

            agent.load_model(id="latest")
            obs, extra_info = env.reset(usr_conf=usr_conf)
            agent.reset()

            if handle_disaster_recovery(extra_info, logger):
                break

            last_predict_act = None
            done = False
            while not done:
                need_to_predict = obs["legal_action"][0] != 0
                if need_to_predict:
                    if len(collector) > 0:
                        collector[-1].reward = reward_shaping(obs, last_predict_act, agent)

                    obs_data = agent.observation_process(obs, extra_info)
                    act_data = agent.predict(list_obs_data=[obs_data])[0]
                    act = agent.action_process(act_data)
                else:
                    agent.preprocess.update_traffic_info(obs, extra_info)
                    act = [None, None, None]

                frame_no, _obs, terminated, truncated, _extra_info = env.step(act)
                if handle_disaster_recovery(_extra_info, logger):
                    logger.info("_obs is None, so break")
                    break

                done = terminated or truncated or (train_test_quick_stop and len(collector) > 1)
                if truncated:
                    logger.info(f"truncated is True, frame_no is {frame_no}, so this episode timeout")
                elif terminated:
                    logger.info(f"terminated is True, frame_no is {frame_no}, so this episode reach the end")

                if need_to_predict:
                    prob, value, action = act_data.prob, act_data.value, act_data.action
                    frame = SampleData(
                        obs=np.array(obs_data.feature),
                        legal_action=np.array(obs_data.legal_action),
                        act=action,
                        reward=None,
                        done=0,
                        reward_sum=0,
                        value=value.flatten()[0],
                        next_value=0,
                        advantage=0,
                        prob=prob,
                        sub_action=np.array(obs_data.sub_action_mask),
                        is_train=1,
                    )
                    collector.append(frame)

                obs = _obs
                extra_info = _extra_info
                if need_to_predict:
                    last_predict_act = act

                if done:
                    if len(collector) > 1:
                        collector[-1].done = 1
                        collector[-1].reward = reward_shaping(obs, last_predict_act, agent)
                        collector = sample_process(collector)
                        yield collector
                        collector = []
                    break
    except Exception as e:
        logger.error(f"run_episodes error: {e}")
        raise RuntimeError("run_episodes error")


def handle_disaster_recovery(extra_info, logger):
    result_code, result_message = extra_info["result_code"], extra_info["result_message"]
    if result_code < 0:
        logger.error(f"env reset failed, please check, result_code is {result_code}, result_msg is {result_message}")
        raise RuntimeError(result_message)
    if result_code > 0:
        return True
    return False
