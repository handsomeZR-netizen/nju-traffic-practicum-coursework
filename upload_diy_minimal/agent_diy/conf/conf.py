#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


class Config:

    # Observation and action shapes
    DIM_OF_OBSERVATION = 32
    DIM_OF_ACTION_PHASE_1 = 4
    DIM_OF_ACTION_DURATION_1 = 4
    DIM_SUB_ACTION_MASK = 24
    NUMB_HEAD = 2

    GRID_WIDTH = 14
    GRID_NUM = 20
    GRID_LENGTH = 5
    MAX_GREEN_DURATION = 40
    MAX_RED_DURATION = 60
    DURATION_TO_SECONDS = [5, 10, 15, 20]

    # PI-eLight style rule controller
    CLOSE_DISTANCE = 120.0
    VERY_CLOSE_DISTANCE = 40.0
    CLOSE_PRESSURE_COEF = 3.2
    VERY_CLOSE_PRESSURE_COEF = 2.0
    QUEUE_PRESSURE_COEF = 2.6
    WAIT_PRESSURE_COEF = 1.2
    VEHICLE_PRESSURE_COEF = 0.7
    DELAY_PRESSURE_COEF = 0.8
    IDLE_BONUS_COEF = 0.7
    MAX_IDLE_ROUNDS = 3
    MIN_PHASE_HOLD_ROUNDS = 2
    KEEP_SCORE_MARGIN = 1.2
    KEEP_SCORE_RATIO = 0.92
    STARVATION_OVERRIDE_RATIO = 0.88
    KEEP_PHASE_BONUS = 0.45
    MIN_SWITCH_DURATION_ACTION = 1
    RULE_PHASE_BIAS_SCALE = 1.15
    RULE_DURATION_BIAS = 0.95

    # PPO training parameters
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
    IS_REINFORCE_TASK_LIST = [True] * NUMB_HEAD
    EVAL_FREQ = 5
    GAMMA = 0.995
    LAMDA = 0.95
    USE_GRAD_CLIP = False
    GRAD_CLIP_RANGE = 0.9
    VALUE_COEF = 1
