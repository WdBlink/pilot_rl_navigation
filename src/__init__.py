#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强化学习无人机定位导航系统

一个基于强化学习的无人机智能定位导航系统，集成多传感器融合、视觉定位、
自主恢复控制等先进技术，实现无人机在复杂环境下的精确导航和可靠飞行。

Author: wdblink
Date: 2024
"""

__version__ = "1.0.0"
__author__ = "wdblink"
__email__ = "wdblink@example.com"
__description__ = "基于强化学习的无人机智能定位导航系统"

# 导入核心模块
from .core import (
    RLAgent,
    PositionFusion,
    OpticalPositioning,
    RecoveryController,
    ReliabilityEvaluator
)

from .utils import (
    Position3D,
    FlightAttitude,
    SystemState,
    RLAction,
    ControlMode,
    get_logger,
    Config
)

from .interfaces import (
    SensorInterface,
    ControllerInterface
)

# 版本信息
__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__description__",
    # 核心模块
    "RLAgent",
    "PositionFusion", 
    "OpticalPositioning",
    "RecoveryController",
    "ReliabilityEvaluator",
    # 数据类型
    "Position3D",
    "FlightAttitude",
    "SystemState",
    "RLAction",
    "ControlMode",
    # 工具模块
    "get_logger",
    "Config",
    # 接口模块
    "SensorInterface",
    "ControllerInterface",
]