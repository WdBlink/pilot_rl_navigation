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
            速度大小 (米/秒)
        """
        return np.linalg.norm(self.to_array())


class ControlMode(Enum):
    """控制模式枚举类
    
    定义无人机的不同控制模式。
    """
    MANUAL = "manual"              # 手动控制
    STABILIZE = "stabilize"        # 自稳模式
    ALT_HOLD = "alt_hold"          # 定高模式
    LOITER = "loiter"              # 悬停模式
    AUTO = "auto"                  # 自动模式
    GUIDED = "guided"              # 引导模式
    RTL = "rtl"                    # 返航模式
    LAND = "land"                  # 降落模式
    EMERGENCY = "emergency"        # 紧急模式


@dataclass
class SystemState:
    """系统状态数据类
    
    包含无人机的完整状态信息，用于强化学习的观测空间。
    
    Attributes:
        position: 当前位置
        attitude: 当前姿态
        velocity: 当前速度
        optical_position: 光学定位结果
        optical_confidence: 光学定位置信度
        battery_voltage: 电池电压 (伏特)
        control_mode: 当前控制模式
        pipeline_deviation: 管道偏离距离 (米)
        wind_speed: 风速 (米/秒)
        wind_direction: 风向 (弧度)
        historical_error: 历史定位误差 (米)
        timestamp: 时间戳 (秒)
    """
    position: Position3D
    attitude: FlightAttitude
    velocity: VelocityVector
    optical_position: Optional[Position3D] = None
    optical_confidence: float = 0.0
    battery_voltage: float = 16.8
    control_mode: ControlMode = ControlMode.GUIDED
    pipeline_deviation: float = 0.0
    wind_speed: float = 0.0
    wind_direction: float = 0.0
    historical_error: float = 0.0
    timestamp: float = 0.0
    
    def to_observation_vector(self) -> np.ndarray:
        """转换为观测向量
        
        Returns:
            用于强化学习的观测向量
        """
        obs = []
        
        # 位置和姿态 (6维)
        obs.extend(self.position.to_array())
        obs.extend(self.attitude.to_array())
        
        # 光学定位信息 (4维)
        if self.optical_position is not None:
            obs.extend(self.optical_position.to_array())
        else:
            obs.extend([0.0, 0.0, 0.0])
        obs.append(self.optical_confidence)
        
        # 环境状态 (8维)
        obs.extend(self.velocity.to_array())
        obs.append(self.pipeline_deviation)
        obs.append(self.battery_voltage / 16.8)  # 归一化电池电压
        obs.append(self.wind_speed)
        obs.append(self.wind_direction)
        obs.append(self.historical_error)
        
        return np.array(obs, dtype=np.float32)


@dataclass
class RLAction:
    """强化学习动作数据类
    
    定义强化学习智能体输出的动作信息。
    
    Attributes:
        gps_weight: GPS权重 [0.0, 1.0]
        optical_weight: 光学定位权重 [0.0, 1.0]
        imu_weight: IMU权重 [0.0, 1.0]
        velocity_x: X轴目标速度 (米/秒)
        velocity_y: Y轴目标速度 (米/秒)
        velocity_z: Z轴目标速度 (米/秒)
        emergency_action: 是否触发紧急动作
        timestamp: 时间戳 (秒)
    """
    gps_weight: float
    optical_weight: float
    imu_weight: float
    velocity_x: float
    velocity_y: float
    velocity_z: float
    emergency_action: bool = False
    timestamp: float = 0.0
    
    def normalize_weights(self) -> None:
        """归一化融合权重"""
        total_weight = self.gps_weight + self.optical_weight + self.imu_weight
        if total_weight > 0:
            self.gps_weight /= total_weight
            self.optical_weight /= total_weight
            self.imu_weight /= total_weight
    
    def to_action_vector(self) -> np.ndarray:
        """转换为动作向量
        
        Returns:
            动作向量
        """
        return np.array([
            self.gps_weight,
            self.optical_weight,
            self.imu_weight,
            self.velocity_x,
            self.velocity_y,
            self.velocity_z,
            float(self.emergency_action)
        ], dtype=np.float32)


@dataclass
class OpticalMatchResult:
    """光学匹配结果数据类
    
    包含光学定位系统的匹配结果和质量评估信息。
    
    Attributes:
        position: 估计位置
        confidence: 匹配置信度 [0.0, 1.0]
        num_matches: 特征匹配点数量
        match_quality: 匹配质量分数 [0.0, 1.0]
        processing_time: 处理时间 (秒)
        timestamp: 时间戳 (秒)
    """
    position: Position3D
    confidence: float
    num_matches: int
    match_quality: float
    processing_time: float
    timestamp: float


@dataclass
class TrainingMetrics:
    """训练指标数据类
    
    记录强化学习训练过程中的各种指标。
    
    Attributes:
        episode: 当前回合数
        total_reward: 总奖励
        episode_length: 回合长度
        average_reward: 平均奖励
        loss: 损失值
        learning_rate: 当前学习率
        exploration_rate: 探索率
        success_rate: 成功率
        timestamp: 时间戳 (秒)
    """
    episode: int = 0
    total_reward: float = 0.0
    episode_length: int = 0
    average_reward: float = 0.0
    loss: float = 0.0
    learning_rate: float = 0.0
    exploration_rate: float = 0.0
    success_rate: float = 0.0
    timestamp: float = 0.0


@dataclass
class IMUData:
    """IMU数据类
    
    包含惯性测量单元的传感器数据。
    
    Attributes:
        acceleration: 加速度 [ax, ay, az] (米/秒²)
        angular_velocity: 角速度 [wx, wy, wz] (弧度/秒)
        magnetic_field: 磁场强度 [mx, my, mz] (特斯拉)
        temperature: 温度 (摄氏度)
        timestamp: 时间戳 (秒)
    """
    acceleration: List[float]
    angular_velocity: List[float]
    magnetic_field: Optional[List[float]] = None
    temperature: Optional[float] = None
    timestamp: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """转换为numpy数组
        
        Returns:
            包含加速度和角速度的数组
        """
        return np.array(self.acceleration + self.angular_velocity)


@dataclass
class GPSData:
    """GPS数据类
    
    包含GPS接收器的定位数据。
    
    Attributes:
        latitude: 纬度 (度)
        longitude: 经度 (度)
        altitude: 海拔高度 (米)
        accuracy: 定位精度 (米)
        num_satellites: 卫星数量
        fix_type: 定位类型 (0=无定位, 1=2D, 2=3D)
        timestamp: 时间戳 (秒)
    """
    latitude: float
    longitude: float
    altitude: float
    accuracy: float
    num_satellites: int
    fix_type: int
    timestamp: float
    
    def to_position3d(self, origin_lat: float, origin_lon: float) -> Position3D:
        """转换为Position3D格式
        
        Args:
            origin_lat: 原点纬度
            origin_lon: 原点经度
            
        Returns:
            Position3D对象
        """
        # 简化的坐标转换（实际应用中需要更精确的转换）
        x = (self.longitude - origin_lon) * 111320 * np.cos(np.radians(self.latitude))
        y = (self.latitude - origin_lat) * 110540
        z = self.altitude
        
        confidence = min(1.0, max(0.0, (10.0 - self.accuracy) / 10.0))
        
        return Position3D(
            x=x, y=y, z=z,
            timestamp=self.timestamp,
            confidence=confidence
        )


@dataclass
class SensorHealth:
    """传感器健康状态数据类
    
    监控各传感器的工作状态和健康程度。
    
    Attributes:
        gps_health: GPS健康状态 [0.0, 1.0]
        imu_health: IMU健康状态 [0.0, 1.0]
        camera_health: 相机健康状态 [0.0, 1.0]
        optical_health: 光学定位健康状态 [0.0, 1.0]
        overall_health: 整体健康状态 [0.0, 1.0]
        timestamp: 时间戳 (秒)
    """
    gps_health: float = 1.0
    imu_health: float = 1.0
    camera_health: float = 1.0
    optical_health: float = 1.0
    overall_health: float = 1.0
    timestamp: float = 0.0
    
    def update_overall_health(self) -> None:
        """更新整体健康状态"""
        self.overall_health = np.mean([
            self.gps_health,
            self.imu_health,
            self.camera_health,
            self.optical_health
        ])


# 类型别名
ObservationVector = np.ndarray
ActionVector = np.ndarray
RewardValue = float
DoneFlag = bool
InfoDict = Dict[str, Any]