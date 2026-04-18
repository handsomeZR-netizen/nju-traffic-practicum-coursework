#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


class Config:

    # Size of observation
    # observation的维度，
    DIM_OF_OBSERVATION = 32
    DIM_OF_ACTION_PHASE_1 = 4
    DIM_OF_ACTION_DURATION_1 = 4
    DIM_SUB_ACTION_MASK = 24

    # Size of model output
    # 策略输出几个头，如果包含相位和时间，则为2个头
    NUMB_HEAD = 2

    GRID_WIDTH = 14
    GRID_NUM = 20
    GRID_LENGTH = 5
    MAX_GREEN_DURATION = 40
    MAX_RED_DURATION = 60
    DURATION_TO_SECONDS = [5, 10, 15, 20]

    # Rule-based controller config
    # 启发式控制器配置
    PHASE_SWITCH_MARGIN = 1.5
    PHASE_SWITCH_RATIO = 1.15
    MAX_IDLE_ROUNDS = 3
    MIN_SWITCH_DURATION_ACTION = 1
    QUEUE_PRESSURE_COEF = 4.0
    WAIT_PRESSURE_COEF = 1.6
    VEHICLE_PRESSURE_COEF = 1.0
    DELAY_PRESSURE_COEF = 1.2
    IDLE_BONUS_COEF = 0.8
    RULE_PHASE_BIAS_SCALE = 0.8
    RULE_DURATION_BIAS = 0.8

    # Algorithm config
    # PPO 算法配置
    INIT_LEARNING_RATE_START = 0.001
    BETA_START = 0.01
    LOG_EPSILON = 1e-6

    RMSPROP_DECAY = 0.9
    RMSPROP_MOMENTUM = 0.0
    RMSPROP_EPSILON = 0.01
    CLIP_PARAM = 0.2

    MIN_POLICY = 0.00001

    LABEL_SIZE_LIST = [DIM_OF_ACTION_PHASE_1, DIM_OF_ACTION_DURATION_1]
    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()
    # means each task whether need reinforce
    # 标识要进行强化学习的头
    IS_REINFORCE_TASK_LIST = [
        True,
    ] * NUMB_HEAD

    EVAL_FREQ = 5
    GAMMA = 0.995
    LAMDA = 0.95

    USE_GRAD_CLIP = False
    GRAD_CLIP_RANGE = 0.9
    VALUE_COEF = 1
