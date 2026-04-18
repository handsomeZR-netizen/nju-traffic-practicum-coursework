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
from agent_ppo.feature.definition import *
from kaiwu_agent.utils.common_func import Frame, attached
from tools.train_env_conf_validate import read_usr_conf
from tools.metrics_utils import get_training_metrics


@attached
def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    env, agent = envs[0], agents[0]
    epoch_num = 100000
    episode_num_every_epoch = 1
    last_save_model_time = 0

    # Initializing monitoring data
    # 监控数据初始化
    monitor_data = {
        "reward": 0,
        "diy_1": 0,
        "diy_2": 0,
        "diy_3": 0,
        "diy_4": 0,
        "diy_5": 0,
    }
    last_report_monitor_time = time.time()

    # Read and validate configuration file
    # 配置文件读取和校验
    usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error(f"usr_conf is None, please check agent_ppo/conf/train_env_conf.toml")
        return

    for epoch in range(epoch_num):
        epoch_total_rew = 0

        data_length = 0
        for g_data in run_episodes(episode_num_every_epoch, env, agent, usr_conf, logger):
            data_length += len(g_data)
            total_rew = []
            for data in g_data:
                total_rew.append(data.reward)

            total_rew = sum(total_rew)
            epoch_total_rew += total_rew
            agent.learn(g_data)
            g_data.clear()

        avg_step_reward = 0
        if data_length:
            avg_step_reward = f"{(epoch_total_rew/data_length):.2f}"

        # save model file
        # 保存model文件
        now = time.time()
        if now - last_save_model_time >= 600:
            agent.save_model()
            last_save_model_time = now

        # Reporting training progress
        # 上报训练进度
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
            collector = list()
            predict_cnt = 0

            # Retrieving training metrics
            # 获取训练中的指标
            training_metrics = get_training_metrics()
            if training_metrics:
                logger.info(f"training_metrics is {training_metrics}")

            # At the start of each environment, loading the latest model file
            # 每次对局开始时, 加载最新model文件
            agent.load_model(id="latest")

            # Reset the environment and get the initial extra_info
            # 重置环境, 并获取初始状态
            obs, extra_info = env.reset(usr_conf=usr_conf)
            agent.reset()

            # Disaster recovery
            # 容灾
            if handle_disaster_recovery(extra_info, logger):
                break

            # Record the last_predict_act
            # 记录上次预测的动作
            last_predict_act = None

            done = False
            while not done:
                need_to_predict = obs["legal_action"][0] != 0
                if need_to_predict:
                    if len(collector) > 0:
                        # Calculate reward Rewards
                        # 计算奖励
                        reward = reward_shaping(obs, last_predict_act, agent)
                        collector[-1].reward = reward

                    # Feature processing
                    # 特征处理
                    obs_data = agent.observation_process(obs, extra_info)
                    # Agent makes a prediction to get the next frame's action
                    # Agent 进行推理, 获取下一帧的预测动作
                    act_data = agent.predict(list_obs_data=[obs_data])
                    act_data = act_data[0]

                    # Unpack ActData into actions
                    # ActData 解包成动作
                    act = agent.action_process(act_data)
                    predict_cnt += 1
                else:
                    # No need to predict
                    # 不需要预测的情况
                    agent.preprocess.update_traffic_info(obs, extra_info)
                    act = [None, None, None]

                # Interact with the environment, execute actions, get the next extra_info
                # 与环境交互, 执行动作, 获取下一步的状态, 如果遇到不需要预测的帧，则env.step直到得到需要预测的帧
                (
                    frame_no,
                    _obs,
                    terminated,
                    truncated,
                    _extra_info,
                ) = env.step(act)

                # logger.info(f"current step is {predict_cnt}")

                # Disaster recovery
                # 容灾
                if handle_disaster_recovery(_extra_info, logger):
                    logger.info(f"_obs is None, so break")
                    break

                # Determine if the environment is over
                # 判断环境结束
                done = terminated or truncated or (train_test_quick_stop and len(collector) > 1)
                if truncated:
                    logger.info(f"truncated is True, frame_no is {frame_no}, so this episode timeout")
                elif terminated:
                    logger.info(f"terminated is True, frame_no is {frame_no}, so this episode reach the end")

                # Save samples only when predicting
                # 只有预测步才保存样本
                if need_to_predict:
                    # Construct environment frames to prepare for sample construction
                    # 构造环境帧，为构造样本做准备
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

                # Status update
                # 状态更新
                obs = _obs
                extra_info = _extra_info
                if need_to_predict:
                    last_predict_act = act

                # Perform sample processing and return samples for training
                # 进行样本处理并将样本返回进行训练
                if done:
                    if len(collector) > 1:
                        # Calculate reward Rewards include phase_reward and duration_reward
                        # 奖励有phase_reward和duration_reward
                        reward = reward_shaping(obs, last_predict_act, agent)
                        collector[-1].done = 1
                        collector[-1].reward = reward
                        collector = sample_process(collector)
                        yield collector
                        collector = []
                    break

    except Exception as e:
        logger.error(f"run_episodes error")
        raise RuntimeError(f"run_episodes error")


def handle_disaster_recovery(extra_info, logger):
    # Handle disaster recovery logic
    # 处理容灾逻辑
    result_code, result_message = extra_info["result_code"], extra_info["result_message"]
    if result_code < 0:
        logger.error(f"env reset failed, please check, result_code is {result_code}, result_msg is {result_message}")
        raise RuntimeError(result_message)
    elif result_code > 0:
        return True
    return False
