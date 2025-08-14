#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强化学习无人机定位导航系统 - 传感器仿真模块

本模块实现各种传感器的仿真功能，包括：
1. IMU传感器仿真（加速度计、陀螺仪）
2. GPS传感器仿真
3. 光学传感器仿真
4. 气压计仿真
5. 磁力计仿真
6. 传感器噪声和故障模拟

Author: wdblink
Date: 2024
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import random
import math

# 导入项目模块
from ..utils.data_types import (
    Position3D, FlightAttitude, SensorData, 
    OpticalMatchResult, SystemState
)
from ..utils.logger import logger_manager, performance_monitor


class SensorType(Enum):
    """传感器类型枚举"""
    IMU = "imu"
    GPS = "gps"
    OPTICAL = "optical"
    BAROMETER = "barometer"
    MAGNETOMETER = "magnetometer"


class NoiseType(Enum):
    """噪声类型枚举"""
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    BIAS = "bias"
    DRIFT = "drift"
    OUTLIER = "outlier"


class FailureMode(Enum):
    """故障模式枚举"""
    NORMAL = "normal"
    INTERMITTENT = "intermittent"
    DEGRADED = "degraded"
    FAILED = "failed"
    STUCK = "stuck"


@dataclass
class NoiseModel:
    """噪声模型配置"""
    noise_type: NoiseType
    parameters: Dict[str, float]  # 噪声参数（如标准差、范围等）
    enabled: bool = True


@dataclass
class FailureModel:
    """故障模型配置"""
    failure_mode: FailureMode
    probability: float  # 故障概率
    duration_range: Tuple[float, float]  # 故障持续时间范围
    recovery_probability: float = 0.1  # 恢复概率


@dataclass
class SensorConfig:
    """传感器配置"""
    sensor_type: SensorType
    update_rate: float  # Hz
    noise_models: List[NoiseModel]
    failure_model: Optional[FailureModel]
    enabled: bool = True
    latency: float = 0.0  # 传感器延迟（秒）


class IMUSensor:
    """IMU传感器仿真
    
    模拟加速度计和陀螺仪的数据输出，包括：
    1. 真实物理运动的传感器响应
    2. 传感器噪声和偏差
    3. 温度漂移效应
    4. 传感器故障模拟
    """
    
    def __init__(self, config: SensorConfig):
        """
        初始化IMU传感器
        
        Args:
            config: 传感器配置
        """
        self.config = config
        self.last_update_time = 0.0
        self.bias_drift = np.array([0.0, 0.0, 0.0])  # 偏差漂移
        self.temperature_effect = 0.0
        self.failure_state = FailureMode.NORMAL
        self.failure_start_time = 0.0
        self.stuck_value = None
        
        # IMU特定参数
        self.gravity = np.array([0.0, 0.0, -9.81])  # 重力向量
        self.gyro_bias = np.array([0.0, 0.0, 0.0])  # 陀螺仪偏差
        self.accel_bias = np.array([0.0, 0.0, 0.0])  # 加速度计偏差
        
        logger_manager.info("IMU传感器仿真初始化完成")
    
    @performance_monitor
    def simulate(self, true_acceleration: np.ndarray, 
                true_angular_velocity: np.ndarray,
                attitude: FlightAttitude) -> Optional[SensorData]:
        """
        模拟IMU传感器数据
        
        Args:
            true_acceleration: 真实加速度 (m/s²)
            true_angular_velocity: 真实角速度 (rad/s)
            attitude: 飞行姿态
            
        Returns:
            IMU传感器数据
        """
        current_time = time.time()
        
        # 检查更新频率
        if current_time - self.last_update_time < 1.0 / self.config.update_rate:
            return None
        
        # 检查传感器故障
        if not self._check_sensor_health(current_time):
            return None
        
        # 计算传感器测量值
        measured_accel = self._simulate_accelerometer(true_acceleration, attitude)
        measured_gyro = self._simulate_gyroscope(true_angular_velocity)
        
        # 应用噪声
        measured_accel = self._apply_noise(measured_accel, "acceleration")
        measured_gyro = self._apply_noise(measured_gyro, "angular_velocity")
        
        # 创建传感器数据
        sensor_data = SensorData(
            timestamp=current_time,
            acceleration=measured_accel,
            angular_velocity=measured_gyro,
            confidence=self._calculate_confidence()
        )
        
        self.last_update_time = current_time
        return sensor_data
    
    def _simulate_accelerometer(self, true_acceleration: np.ndarray, 
                              attitude: FlightAttitude) -> np.ndarray:
        """
        模拟加速度计测量
        
        Args:
            true_acceleration: 真实加速度
            attitude: 飞行姿态
            
        Returns:
            加速度计测量值
        """
        # 将重力转换到机体坐标系
        gravity_body = self._rotate_vector_to_body(self.gravity, attitude)
        
        # 加速度计测量 = 真实加速度 + 重力 + 偏差
        measured_accel = true_acceleration + gravity_body + self.accel_bias
        
        # 添加温度效应
        temperature_factor = 1.0 + self.temperature_effect * 0.001
        measured_accel *= temperature_factor
        
        return measured_accel
    
    def _simulate_gyroscope(self, true_angular_velocity: np.ndarray) -> np.ndarray:
        """
        模拟陀螺仪测量
        
        Args:
            true_angular_velocity: 真实角速度
            
        Returns:
            陀螺仪测量值
        """
        # 陀螺仪测量 = 真实角速度 + 偏差 + 漂移
        measured_gyro = true_angular_velocity + self.gyro_bias + self.bias_drift
        
        # 更新偏差漂移
        self.bias_drift += np.random.normal(0, 0.0001, 3)
        
        return measured_gyro
    
    def _rotate_vector_to_body(self, vector: np.ndarray, attitude: FlightAttitude) -> np.ndarray:
        """
        将向量从世界坐标系转换到机体坐标系
        
        Args:
            vector: 世界坐标系向量
            attitude: 飞行姿态
            
        Returns:
            机体坐标系向量
        """
        # 简化的旋转矩阵计算（欧拉角到旋转矩阵）
        roll, pitch, yaw = attitude.roll, attitude.pitch, attitude.yaw
        
        # 旋转矩阵
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        R_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        R = R_z @ R_y @ R_x
        return R.T @ vector  # 转置用于世界到机体的转换


class GPSSensor:
    """GPS传感器仿真
    
    模拟GPS接收机的数据输出，包括：
    1. 位置精度模拟
    2. 多路径效应
    3. 大气延迟
    4. 卫星可见性影响
    """
    
    def __init__(self, config: SensorConfig):
        """
        初始化GPS传感器
        
        Args:
            config: 传感器配置
        """
        self.config = config
        self.last_update_time = 0.0
        self.failure_state = FailureMode.NORMAL
        self.satellite_count = 8  # 可见卫星数量
        self.hdop = 1.5  # 水平精度因子
        
        logger_manager.info("GPS传感器仿真初始化完成")
    
    @performance_monitor
    def simulate(self, true_position: Position3D) -> Optional[SensorData]:
        """
        模拟GPS传感器数据
        
        Args:
            true_position: 真实位置
            
        Returns:
            GPS传感器数据
        """
        current_time = time.time()
        
        # 检查更新频率
        if current_time - self.last_update_time < 1.0 / self.config.update_rate:
            return None
        
        # 检查传感器故障
        if not self._check_sensor_health(current_time):
            return None
        
        # 模拟卫星可见性
        self._update_satellite_visibility()
        
        # 计算GPS测量误差
        position_error = self._calculate_position_error()
        
        # 应用误差到真实位置
        measured_position = Position3D(
            x=true_position.x + position_error[0],
            y=true_position.y + position_error[1],
            z=true_position.z + position_error[2],
            timestamp=current_time,
            confidence=self._calculate_confidence()
        )
        
        # 应用噪声
        measured_position = self._apply_position_noise(measured_position)
        
        # 创建传感器数据
        sensor_data = SensorData(
            timestamp=current_time,
            position=measured_position,
            satellite_count=self.satellite_count,
            hdop=self.hdop,
            confidence=measured_position.confidence
        )
        
        self.last_update_time = current_time
        return sensor_data
    
    def _update_satellite_visibility(self) -> None:
        """
        更新卫星可见性
        """
        # 模拟卫星数量变化
        self.satellite_count += random.randint(-1, 1)
        self.satellite_count = np.clip(self.satellite_count, 4, 12)
        
        # 根据卫星数量更新HDOP
        if self.satellite_count >= 8:
            self.hdop = random.uniform(1.0, 2.0)
        elif self.satellite_count >= 6:
            self.hdop = random.uniform(1.5, 3.0)
        else:
            self.hdop = random.uniform(2.5, 5.0)
    
    def _calculate_position_error(self) -> np.ndarray:
        """
        计算GPS位置误差
        
        Returns:
            位置误差向量 [x, y, z]
        """
        # 基础精度（米）
        base_accuracy = 2.0
        
        # 根据HDOP调整精度
        horizontal_error = base_accuracy * self.hdop
        vertical_error = horizontal_error * 1.5  # 垂直精度通常更差
        
        # 生成随机误差
        error = np.array([
            np.random.normal(0, horizontal_error),
            np.random.normal(0, horizontal_error),
            np.random.normal(0, vertical_error)
        ])
        
        return error
    
    def _apply_position_noise(self, position: Position3D) -> Position3D:
        """
        应用位置噪声
        
        Args:
            position: 原始位置
            
        Returns:
            添加噪声后的位置
        """
        for noise_model in self.config.noise_models:
            if noise_model.enabled:
                if noise_model.noise_type == NoiseType.GAUSSIAN:
                    std = noise_model.parameters.get('std', 1.0)
                    position.x += np.random.normal(0, std)
                    position.y += np.random.normal(0, std)
                    position.z += np.random.normal(0, std)
        
        return position


class OpticalSensor:
    """光学传感器仿真
    
    模拟视觉定位系统的数据输出，包括：
    1. 图像匹配质量模拟
    2. 光照条件影响
    3. 特征点检测成功率
    4. 匹配置信度计算
    """
    
    def __init__(self, config: SensorConfig):
        """
        初始化光学传感器
        
        Args:
            config: 传感器配置
        """
        self.config = config
        self.last_update_time = 0.0
        self.failure_state = FailureMode.NORMAL
        self.lighting_condition = 1.0  # 光照条件因子
        
        logger_manager.info("光学传感器仿真初始化完成")
    
    @performance_monitor
    def simulate(self, true_position: Position3D, 
                environment_conditions: Dict[str, float]) -> Optional[OpticalMatchResult]:
        """
        模拟光学传感器数据
        
        Args:
            true_position: 真实位置
            environment_conditions: 环境条件
            
        Returns:
            光学匹配结果
        """
        current_time = time.time()
        
        # 检查更新频率
        if current_time - self.last_update_time < 1.0 / self.config.update_rate:
            return None
        
        # 检查传感器故障
        if not self._check_sensor_health(current_time):
            return None
        
        # 更新环境条件
        self._update_environment_conditions(environment_conditions)
        
        # 计算匹配质量
        match_score = self._calculate_match_score(true_position)
        
        # 如果匹配质量太低，返回None（匹配失败）
        if match_score < 0.3:
            return None
        
        # 计算位置估计误差
        position_error = self._calculate_optical_error(match_score)
        
        # 生成光学位置估计
        optical_position = Position3D(
            x=true_position.x + position_error[0],
            y=true_position.y + position_error[1],
            z=true_position.z + position_error[2],
            timestamp=current_time,
            confidence=match_score
        )
        
        # 创建光学匹配结果
        match_result = OpticalMatchResult(
            matched_position=optical_position,
            match_score=match_score,
            feature_count=self._simulate_feature_count(match_score),
            processing_time=random.uniform(0.05, 0.2),
            timestamp=current_time
        )
        
        self.last_update_time = current_time
        return match_result
    
    def _update_environment_conditions(self, conditions: Dict[str, float]) -> None:
        """
        更新环境条件
        
        Args:
            conditions: 环境条件字典
        """
        # 光照条件影响
        self.lighting_condition = conditions.get('lighting', 1.0)
        
        # 天气条件影响
        weather_factor = conditions.get('weather', 1.0)
        self.lighting_condition *= weather_factor
    
    def _calculate_match_score(self, position: Position3D) -> float:
        """
        计算匹配评分
        
        Args:
            position: 当前位置
            
        Returns:
            匹配评分 (0.0-1.0)
        """
        # 基础匹配质量
        base_score = random.uniform(0.6, 0.95)
        
        # 光照条件影响
        lighting_factor = np.clip(self.lighting_condition, 0.3, 1.0)
        
        # 高度影响（高度越高，匹配质量可能越差）
        altitude_factor = max(0.5, 1.0 - (position.z - 50) / 200)
        
        # 综合评分
        match_score = base_score * lighting_factor * altitude_factor
        
        # 添加随机波动
        match_score += np.random.normal(0, 0.05)
        
        return np.clip(match_score, 0.0, 1.0)
    
    def _calculate_optical_error(self, match_score: float) -> np.ndarray:
        """
        计算光学定位误差
        
        Args:
            match_score: 匹配评分
            
        Returns:
            位置误差向量
        """
        # 误差与匹配质量成反比
        base_error = 5.0 * (1.0 - match_score)
        
        error = np.array([
            np.random.normal(0, base_error),
            np.random.normal(0, base_error),
            np.random.normal(0, base_error * 0.5)  # 垂直误差较小
        ])
        
        return error
    
    def _simulate_feature_count(self, match_score: float) -> int:
        """
        模拟特征点数量
        
        Args:
            match_score: 匹配评分
            
        Returns:
            特征点数量
        """
        # 特征点数量与匹配质量相关
        base_count = int(match_score * 200)
        variation = random.randint(-20, 20)
        
        return max(10, base_count + variation)


class SensorSimulator:
    """传感器仿真器
    
    统一管理所有传感器的仿真，包括：
    1. 多传感器协调仿真
    2. 传感器故障注入
    3. 环境条件模拟
    4. 数据同步管理
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化传感器仿真器
        
        Args:
            config: 仿真配置
        """
        self.config = config
        self.sensors = {}
        self.environment_conditions = {
            'lighting': 1.0,
            'weather': 1.0,
            'temperature': 20.0,
            'wind_speed': 0.0
        }
        
        # 初始化各类传感器
        self._initialize_sensors()
        
        logger_manager.info("传感器仿真器初始化完成")
    
    def _initialize_sensors(self) -> None:
        """
        初始化所有传感器
        """
        sensor_configs = self.config.get('sensors', {})
        
        for sensor_name, sensor_config in sensor_configs.items():
            if sensor_config['type'] == 'imu':
                self.sensors[sensor_name] = IMUSensor(self._create_sensor_config(sensor_config))
            elif sensor_config['type'] == 'gps':
                self.sensors[sensor_name] = GPSSensor(self._create_sensor_config(sensor_config))
            elif sensor_config['type'] == 'optical':
                self.sensors[sensor_name] = OpticalSensor(self._create_sensor_config(sensor_config))
    
    def _create_sensor_config(self, config_dict: Dict[str, Any]) -> SensorConfig:
        """
        创建传感器配置对象
        
        Args:
            config_dict: 配置字典
            
        Returns:
            传感器配置对象
        """
        # 创建噪声模型
        noise_models = []
        for noise_config in config_dict.get('noise_models', []):
            noise_model = NoiseModel(
                noise_type=NoiseType(noise_config['type']),
                parameters=noise_config['parameters'],
                enabled=noise_config.get('enabled', True)
            )
            noise_models.append(noise_model)
        
        # 创建故障模型
        failure_model = None
        if 'failure_model' in config_dict:
            failure_config = config_dict['failure_model']
            failure_model = FailureModel(
                failure_mode=FailureMode(failure_config['mode']),
                probability=failure_config['probability'],
                duration_range=tuple(failure_config['duration_range']),
                recovery_probability=failure_config.get('recovery_probability', 0.1)
            )
        
        return SensorConfig(
            sensor_type=SensorType(config_dict['type']),
            update_rate=config_dict['update_rate'],
            noise_models=noise_models,
            failure_model=failure_model,
            enabled=config_dict.get('enabled', True),
            latency=config_dict.get('latency', 0.0)
        )
    
    @performance_monitor
    def simulate_all_sensors(self, true_state: SystemState) -> Dict[str, Any]:
        """
        模拟所有传感器数据
        
        Args:
            true_state: 真实系统状态
            
        Returns:
            所有传感器数据字典
        """
        sensor_data = {}
        
        for sensor_name, sensor in self.sensors.items():
            if isinstance(sensor, IMUSensor):
                data = sensor.simulate(
                    true_state.acceleration,
                    true_state.angular_velocity,
                    true_state.attitude
                )
            elif isinstance(sensor, GPSSensor):
                data = sensor.simulate(true_state.position)
            elif isinstance(sensor, OpticalSensor):
                data = sensor.simulate(true_state.position, self.environment_conditions)
            else:
                data = None
            
            if data is not None:
                sensor_data[sensor_name] = data
        
        return sensor_data
    
    def update_environment_conditions(self, conditions: Dict[str, float]) -> None:
        """
        更新环境条件
        
        Args:
            conditions: 环境条件字典
        """
        self.environment_conditions.update(conditions)
        logger_manager.debug(f"环境条件已更新: {self.environment_conditions}")
    
    def inject_sensor_failure(self, sensor_name: str, failure_mode: FailureMode, 
                            duration: float) -> None:
        """
        注入传感器故障
        
        Args:
            sensor_name: 传感器名称
            failure_mode: 故障模式
            duration: 故障持续时间
        """
        if sensor_name in self.sensors:
            sensor = self.sensors[sensor_name]
            sensor.failure_state = failure_mode
            sensor.failure_start_time = time.time()
            logger_manager.warning(f"传感器 {sensor_name} 故障注入: {failure_mode.value}")
    
    def get_sensor_statistics(self) -> Dict[str, Any]:
        """
        获取传感器统计信息
        
        Returns:
            传感器统计信息字典
        """
        stats = {
            'sensor_count': len(self.sensors),
            'environment_conditions': self.environment_conditions.copy()
        }
        
        for sensor_name, sensor in self.sensors.items():
            stats[f'{sensor_name}_status'] = sensor.failure_state.value
            stats[f'{sensor_name}_last_update'] = sensor.last_update_time
        
        return stats
    
    def reset_all_sensors(self) -> None:
        """
        重置所有传感器状态
        """
        for sensor in self.sensors.values():
            sensor.failure_state = FailureMode.NORMAL
            sensor.last_update_time = 0.0
            if hasattr(sensor, 'bias_drift'):
                sensor.bias_drift = np.array([0.0, 0.0, 0.0])
        
        logger_manager.info("所有传感器状态已重置")


# 为传感器基类添加通用方法
class SensorBase:
    """传感器基类"""
    
    def _check_sensor_health(self, current_time: float) -> bool:
        """
        检查传感器健康状态
        
        Args:
            current_time: 当前时间
            
        Returns:
            传感器是否正常工作
        """
        if not hasattr(self, 'failure_state'):
            return True
        
        if self.failure_state == FailureMode.FAILED:
            return False
        elif self.failure_state == FailureMode.INTERMITTENT:
            # 间歇性故障
            return random.random() > 0.3
        elif self.failure_state == FailureMode.DEGRADED:
            # 性能降级但仍可用
            return True
        
        return True
    
    def _apply_noise(self, data: np.ndarray, data_type: str) -> np.ndarray:
        """
        应用噪声模型
        
        Args:
            data: 原始数据
            data_type: 数据类型
            
        Returns:
            添加噪声后的数据
        """
        noisy_data = data.copy()
        
        for noise_model in self.config.noise_models:
            if noise_model.enabled:
                if noise_model.noise_type == NoiseType.GAUSSIAN:
                    std = noise_model.parameters.get('std', 0.1)
                    noise = np.random.normal(0, std, data.shape)
                    noisy_data += noise
                elif noise_model.noise_type == NoiseType.UNIFORM:
                    range_val = noise_model.parameters.get('range', 0.1)
                    noise = np.random.uniform(-range_val, range_val, data.shape)
                    noisy_data += noise
        
        return noisy_data
    
    def _calculate_confidence(self) -> float:
        """
        计算传感器置信度
        
        Returns:
            置信度 (0.0-1.0)
        """
        if self.failure_state == FailureMode.NORMAL:
            return random.uniform(0.8, 0.95)
        elif self.failure_state == FailureMode.DEGRADED:
            return random.uniform(0.5, 0.7)
        elif self.failure_state == FailureMode.INTERMITTENT:
            return random.uniform(0.3, 0.6)
        else:
            return 0.1


# 让所有传感器类继承基类
IMUSensor.__bases__ = (SensorBase,)
GPSSensor.__bases__ = (SensorBase,)
OpticalSensor.__bases__ = (SensorBase,)