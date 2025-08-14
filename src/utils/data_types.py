#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强化学习无人机定位导航系统 - 数据类型定义模块

本模块定义了系统中使用的所有核心数据结构，包括位置信息、飞行状态、
传感器数据、强化学习动作等基础数据类型。

Author: wdblink
Date: 2024
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from enum import Enum
import time


@dataclass
class Position3D:
    """三维位置信息数据类
    
    用于表示无人机在三维空间中的位置坐标，包含时间戳和置信度信息。
    
    Attributes:
        x: X轴坐标 (米)
        y: Y轴坐标 (米) 
        z: Z轴坐标 (米)
        timestamp: 时间戳 (秒)
        confidence: 位置置信度 [0.0, 1.0]
    """
    x: float
    y: float
    z: float
    timestamp: float
    confidence: float = 1.0
    
    def to_array(self) -> np.ndarray:
        """转换为numpy数组格式
        
        Returns:
            包含[x, y, z]的numpy数组
        """
        return np.array([self.x, self.y, self.z])
    
    def distance_to(self, other: 'Position3D') -> float:
        """计算到另一个位置的欧几里得距离
        
        Args:
            other: 另一个位置对象
            
        Returns:
            两点间的距离 (米)
        """
        return np.linalg.norm(self.to_array() - other.to_array())


@dataclass
class FlightAttitude:
    """飞行姿态信息数据类
    
    表示无人机的三轴姿态角度信息。
    
    Attributes:
        roll: 横滚角 (弧度)
        pitch: 俯仰角 (弧度)
        yaw: 偏航角 (弧度)
        timestamp: 时间戳 (秒)
    """
    roll: float   
    pitch: float  
    yaw: float    
    timestamp: float
    
    def to_array(self) -> np.ndarray:
        """转换为numpy数组格式
        
        Returns:
            包含[roll, pitch, yaw]的numpy数组
        """
        return np.array([self.roll, self.pitch, self.yaw])


@dataclass
class VelocityVector:
    """速度向量数据类
    
    表示无人机在三维空间中的速度信息。
    
    Attributes:
        vx: X轴速度 (米/秒)
        vy: Y轴速度 (米/秒)
        vz: Z轴速度 (米/秒)
        timestamp: 时间戳 (秒)
    """
    vx: float
    vy: float
    vz: float
    timestamp: float
    
    def to_array(self) -> np.ndarray:
        """转换为numpy数组格式
        
        Returns:
            包含[vx, vy, vz]的numpy数组
        """
        return np.array([self.vx, self.vy, self.vz])
    
    def magnitude(self) -> float:
        """计算速度大小
        
        Returns:
            速度的模长 (米/秒)
        """
        return np.linalg.norm(self.to_array())


@dataclass
class OpticalMatchResult:
    """光学匹配结果数据类
    
    包含光学定位算法的匹配结果和相关质量指标。
    
    Attributes:
        position: 匹配得到的位置信息
        feature_points: 特征点数量
        match_score: 匹配得分 [0.0, 1.0]
        affine_matrix: 仿射变换矩阵
        processing_time: 处理时间 (秒)
    """
    position: Position3D
    feature_points: int
    match_score: float
    affine_matrix: np.ndarray
    processing_time: float
    
    def is_valid(self, min_features: int = 10, min_score: float = 0.5) -> bool:
        """检查匹配结果是否有效
        
        Args:
            min_features: 最小特征点数量
            min_score: 最小匹配得分
            
        Returns:
            匹配结果是否有效
        """
        return (self.feature_points >= min_features and 
                self.match_score >= min_score)


class ControlMode(Enum):
    """控制模式枚举类
    
    定义无人机的不同控制模式。
    """
    NORMAL = "normal"        # 正常模式
    RECOVERY = "recovery"    # 恢复模式
    EMERGENCY = "emergency"  # 紧急模式


class RecoveryState(Enum):
    """恢复状态枚举类
    
    定义位置丢失后的恢复状态。
    """
    NORMAL = "normal"              # 正常状态
    POSITION_LOST = "position_lost" # 位置丢失
    SEARCHING = "searching"        # 搜索中
    RECOVERING = "recovering"      # 恢复中
    FAILED = "failed"              # 恢复失败


@dataclass
class SystemState:
    """系统状态信息数据类
    
    包含无人机导航系统的完整状态信息，用于强化学习决策。
    
    Attributes:
        inertial_position: 惯导位置信息
        optical_position: 光学定位结果 (可选)
        flight_attitude: 飞行姿态
        velocity: 速度向量
        pipeline_deviation: 管道偏离距离 (米)
        battery_level: 电池电量 [0.0, 1.0]
        wind_condition: 风况信息 (速度, 方向)
        historical_error: 历史误差
        control_mode: 当前控制模式
    """
    inertial_position: Position3D
    optical_position: Optional[OpticalMatchResult]
    flight_attitude: FlightAttitude
    velocity: VelocityVector
    pipeline_deviation: float
    battery_level: float
    wind_condition: Tuple[float, float]  # (speed, direction)
    historical_error: float
    control_mode: ControlMode
    
    def to_observation_vector(self) -> np.ndarray:
        """转换为强化学习观测向量
        
        Returns:
            标准化的观测向量
        """
        obs = []
        
        # 惯导位置 (3维)
        obs.extend(self.inertial_position.to_array())
        
        # 光学位置有效性和质量 (4维)
        if self.optical_position is not None:
            obs.extend(self.optical_position.position.to_array())
            obs.append(self.optical_position.match_score)
        else:
            obs.extend([0.0, 0.0, 0.0, 0.0])
        
        # 飞行姿态 (3维)
        obs.extend(self.flight_attitude.to_array())
        
        # 速度 (3维)
        obs.extend(self.velocity.to_array())
        
        # 其他状态信息 (5维)
        obs.extend([
            self.pipeline_deviation,
            self.battery_level,
            self.wind_condition[0],  # 风速
            self.wind_condition[1],  # 风向
            self.historical_error
        ])
        
        return np.array(obs, dtype=np.float32)


@dataclass
class RLAction:
    """强化学习动作数据类
    
    定义强化学习智能体输出的动作空间。
    
    Attributes:
        fusion_weights: 融合权重 (lambda_ins, alpha_opt, bias)
        update_decision: 是否更新位置估计
        control_mode: 控制模式
        pipeline_adjustment: 管道调整量 (dx, dy)
        confidence_threshold: 置信度阈值
    """
    fusion_weights: Tuple[float, float, float]  # (lambda_ins, alpha_opt, bias)
    update_decision: bool
    control_mode: ControlMode
    pipeline_adjustment: Tuple[float, float]  # (dx, dy)
    confidence_threshold: float
    
    def to_action_vector(self) -> np.ndarray:
        """转换为动作向量
        
        Returns:
            标准化的动作向量
        """
        action = []
        
        # 融合权重 (3维)
        action.extend(self.fusion_weights)
        
        # 更新决策 (1维)
        action.append(float(self.update_decision))
        
        # 控制模式 (3维 one-hot编码)
        mode_encoding = [0.0, 0.0, 0.0]
        if self.control_mode == ControlMode.NORMAL:
            mode_encoding[0] = 1.0
        elif self.control_mode == ControlMode.RECOVERY:
            mode_encoding[1] = 1.0
        elif self.control_mode == ControlMode.EMERGENCY:
            mode_encoding[2] = 1.0
        action.extend(mode_encoding)
        
        # 管道调整 (2维)
        action.extend(self.pipeline_adjustment)
        
        # 置信度阈值 (1维)
        action.append(self.confidence_threshold)
        
        return np.array(action, dtype=np.float32)


@dataclass
class IMUData:
    """IMU传感器数据类
    
    包含惯性测量单元的原始传感器数据。
    
    Attributes:
        angular_velocity: 角速度 (弧度/秒)
        linear_acceleration: 线性加速度 (米/秒²)
        orientation: 四元数姿态
        timestamp: 时间戳 (秒)
    """
    angular_velocity: np.ndarray  # [wx, wy, wz]
    linear_acceleration: np.ndarray  # [ax, ay, az]
    orientation: np.ndarray  # [w, x, y, z] 四元数
    timestamp: float


@dataclass
class TrainingMetrics:
    """训练指标数据类
    
    记录强化学习训练过程中的关键指标。
    
    Attributes:
        episode: 训练轮次
        total_reward: 总奖励
        episode_length: 轮次长度
        position_error: 位置误差 (米)
        fusion_accuracy: 融合精度
        recovery_success_rate: 恢复成功率
        timestamp: 时间戳
    """
    episode: int
    total_reward: float
    episode_length: int
    position_error: float
    fusion_accuracy: float
    recovery_success_rate: float
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class EnvironmentConfig:
    """环境配置数据类
    
    定义仿真环境的配置参数。
    
    Attributes:
        max_episode_steps: 最大轮次步数
        position_noise_std: 位置噪声标准差
        optical_failure_rate: 光学定位失效率
        wind_speed_range: 风速范围 (最小值, 最大值)
        pipeline_length: 管道长度 (米)
        reward_weights: 奖励权重字典
    """
    max_episode_steps: int = 1000
    position_noise_std: float = 0.1
    optical_failure_rate: float = 0.1
    wind_speed_range: Tuple[float, float] = (0.0, 10.0)
    pipeline_length: float = 1000.0
    reward_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.reward_weights is None:
            self.reward_weights = {
                'position_accuracy': 1.0,
                'fusion_quality': 0.5,
                'recovery_speed': 0.3,
                'energy_efficiency': 0.2
            }