#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强化学习无人机定位导航系统 - 位置融合模块

本模块实现智能位置融合算法，结合惯导和光学定位数据，使用强化学习
动态调整融合权重，并集成扩展卡尔曼滤波器进行状态估计。

Author: wdblink
Date: 2024
"""

import numpy as np
import time
from collections import deque
from typing import Dict, Any, Optional, Tuple, List
from scipy.spatial.transform import Rotation
from scipy.linalg import inv, cholesky

from ..utils.data_types import (
    Position3D, FlightAttitude, OpticalMatchResult, SystemState, 
    RLAction, ControlMode
)
from ..utils.logger import get_logger, performance_monitor, log_function_call
from .rl_agent import BaseRLAgent


class ExtendedKalmanFilter:
    """扩展卡尔曼滤波器类
    
    用于无人机位置和速度的状态估计，支持非线性运动模型和多传感器融合。
    
    状态向量: [x, y, z, vx, vy, vz, ax, ay, az] (9维)
    观测向量: [x_ins, y_ins, z_ins, x_opt, y_opt, z_opt] (6维)
    """
    
    def __init__(self, 
                 process_noise_std: float = 0.1,
                 measurement_noise_std: float = 0.5,
                 initial_covariance: float = 1.0):
        """初始化扩展卡尔曼滤波器
        
        Args:
            process_noise_std: 过程噪声标准差
            measurement_noise_std: 测量噪声标准差
            initial_covariance: 初始协方差
        """
        self.state_dim = 9  # [x, y, z, vx, vy, vz, ax, ay, az]
        self.obs_dim = 6    # [x_ins, y_ins, z_ins, x_opt, y_opt, z_opt]
        
        # 状态向量和协方差矩阵
        self.x = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim) * initial_covariance
        
        # 过程噪声协方差矩阵
        self.Q = np.eye(self.state_dim) * (process_noise_std ** 2)
        
        # 测量噪声协方差矩阵
        self.R = np.eye(self.obs_dim) * (measurement_noise_std ** 2)
        
        # 时间步长
        self.dt = 0.1
        self.last_update_time = None
        
        self.logger = get_logger("kalman_filter")
    
    def predict(self, dt: Optional[float] = None) -> np.ndarray:
        """预测步骤
        
        Args:
            dt: 时间步长，如果为None则使用默认值
            
        Returns:
            预测的状态向量
        """
        if dt is not None:
            self.dt = dt
        
        # 状态转移矩阵 F
        F = self._get_state_transition_matrix(self.dt)
        
        # 状态预测
        self.x = F @ self.x
        
        # 协方差预测
        self.P = F @ self.P @ F.T + self.Q
        
        return self.x.copy()
    
    def update(self, measurement: np.ndarray, measurement_covariance: Optional[np.ndarray] = None) -> np.ndarray:
        """更新步骤
        
        Args:
            measurement: 测量值 [x_ins, y_ins, z_ins, x_opt, y_opt, z_opt]
            measurement_covariance: 测量协方差矩阵，如果为None则使用默认值
            
        Returns:
            更新后的状态向量
        """
        if measurement_covariance is not None:
            R = measurement_covariance
        else:
            R = self.R
        
        # 观测矩阵 H
        H = self._get_observation_matrix()
        
        # 预测观测
        z_pred = H @ self.x
        
        # 创新（残差）
        y = measurement - z_pred
        
        # 创新协方差
        S = H @ self.P @ H.T + R
        
        # 卡尔曼增益
        try:
            K = self.P @ H.T @ inv(S)
        except np.linalg.LinAlgError:
            self.logger.log_event("warning", "singular_matrix", message="创新协方差矩阵奇异，跳过更新")
            return self.x.copy()
        
        # 状态更新
        self.x = self.x + K @ y
        
        # 协方差更新（Joseph形式，数值稳定）
        I_KH = np.eye(self.state_dim) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        
        return self.x.copy()
    
    def _get_state_transition_matrix(self, dt: float) -> np.ndarray:
        """获取状态转移矩阵
        
        Args:
            dt: 时间步长
            
        Returns:
            状态转移矩阵 F
        """
        F = np.eye(self.state_dim)
        
        # 位置 = 位置 + 速度*dt + 0.5*加速度*dt^2
        F[0:3, 3:6] = np.eye(3) * dt
        F[0:3, 6:9] = np.eye(3) * 0.5 * dt**2
        
        # 速度 = 速度 + 加速度*dt
        F[3:6, 6:9] = np.eye(3) * dt
        
        # 加速度保持不变（可以根据需要修改）
        
        return F
    
    def _get_observation_matrix(self) -> np.ndarray:
        """获取观测矩阵
        
        Returns:
            观测矩阵 H
        """
        H = np.zeros((self.obs_dim, self.state_dim))
        
        # 惯导位置观测 [x, y, z]
        H[0:3, 0:3] = np.eye(3)
        
        # 光学位置观测 [x, y, z]
        H[3:6, 0:3] = np.eye(3)
        
        return H
    
    def get_position(self) -> Position3D:
        """获取当前位置估计
        
        Returns:
            位置估计
        """
        position_covariance = np.trace(self.P[0:3, 0:3])
        confidence = max(0.0, min(1.0, 1.0 / (1.0 + position_covariance)))
        
        return Position3D(
            x=self.x[0],
            y=self.x[1],
            z=self.x[2],
            timestamp=time.time(),
            confidence=confidence
        )
    
    def get_velocity(self) -> np.ndarray:
        """获取当前速度估计
        
        Returns:
            速度向量 [vx, vy, vz]
        """
        return self.x[3:6].copy()
    
    def get_acceleration(self) -> np.ndarray:
        """获取当前加速度估计
        
        Returns:
            加速度向量 [ax, ay, az]
        """
        return self.x[6:9].copy()
    
    def reset(self, initial_state: Optional[np.ndarray] = None) -> None:
        """重置滤波器状态
        
        Args:
            initial_state: 初始状态，如果为None则重置为零
        """
        if initial_state is not None:
            self.x = initial_state.copy()
        else:
            self.x = np.zeros(self.state_dim)
        
        self.P = np.eye(self.state_dim)
        self.last_update_time = None
        
        self.logger.log_event("info", "filter_reset")


class ReliabilityEvaluator:
    """可靠性评估器类
    
    评估传感器数据和融合结果的可靠性，为权重分配提供依据。
    """
    
    def __init__(self, history_length: int = 50):
        """初始化可靠性评估器
        
        Args:
            history_length: 历史数据长度
        """
        self.history_length = history_length
        self.ins_error_history = deque(maxlen=history_length)
        self.optical_error_history = deque(maxlen=history_length)
        self.fusion_error_history = deque(maxlen=history_length)
        
        self.logger = get_logger("reliability_evaluator")
    
    def evaluate_ins_reliability(self, ins_pos: Position3D, reference_pos: Optional[Position3D] = None) -> float:
        """评估惯导可靠性
        
        Args:
            ins_pos: 惯导位置
            reference_pos: 参考位置（如果有）
            
        Returns:
            惯导可靠性 [0.0, 1.0]
        """
        # 基于置信度的基础可靠性
        base_reliability = ins_pos.confidence
        
        # 如果有参考位置，计算误差
        if reference_pos is not None:
            error = ins_pos.distance_to(reference_pos)
            self.ins_error_history.append(error)
            
            # 基于历史误差的可靠性调整
            if len(self.ins_error_history) > 5:
                avg_error = np.mean(list(self.ins_error_history))
                error_reliability = max(0.0, 1.0 - avg_error / 10.0)  # 假设10米为最大可接受误差
                base_reliability = 0.7 * base_reliability + 0.3 * error_reliability
        
        return max(0.0, min(1.0, base_reliability))
    
    def evaluate_optical_reliability(self, optical_result: OpticalMatchResult) -> float:
        """评估光学定位可靠性
        
        Args:
            optical_result: 光学匹配结果
            
        Returns:
            光学定位可靠性 [0.0, 1.0]
        """
        # 基于匹配质量的可靠性
        match_reliability = optical_result.match_quality
        
        # 基于匹配点数量的可靠性
        num_matches_reliability = min(1.0, optical_result.num_matches / 50.0)
        
        # 基于处理时间的可靠性（处理时间过长可能表示匹配困难）
        time_reliability = max(0.0, 1.0 - optical_result.processing_time / 5.0)
        
        # 综合可靠性
        reliability = (
            0.5 * match_reliability +
            0.3 * num_matches_reliability +
            0.2 * time_reliability
        )
        
        return max(0.0, min(1.0, reliability))
    
    def evaluate_fusion_quality(self, fused_pos: Position3D, ins_pos: Position3D, 
                               optical_pos: Optional[Position3D] = None) -> float:
        """评估融合质量
        
        Args:
            fused_pos: 融合位置
            ins_pos: 惯导位置
            optical_pos: 光学位置（如果有）
            
        Returns:
            融合质量 [0.0, 1.0]
        """
        # 计算融合结果与各传感器的一致性
        ins_consistency = 1.0 / (1.0 + fused_pos.distance_to(ins_pos))
        
        if optical_pos is not None:
            optical_consistency = 1.0 / (1.0 + fused_pos.distance_to(optical_pos))
            overall_consistency = 0.6 * ins_consistency + 0.4 * optical_consistency
        else:
            overall_consistency = ins_consistency
        
        return max(0.0, min(1.0, overall_consistency))


class PositionFusion:
    """位置融合主类
    
    实现智能位置融合算法，结合多传感器数据和强化学习权重分配。
    """
    
    def __init__(self, 
                 rl_agent: Optional[BaseRLAgent] = None,
                 use_kalman_filter: bool = True,
                 kalman_config: Optional[Dict[str, float]] = None):
        """初始化位置融合器
        
        Args:
            rl_agent: 强化学习智能体
            use_kalman_filter: 是否使用卡尔曼滤波器
            kalman_config: 卡尔曼滤波器配置
        """
        self.rl_agent = rl_agent
        self.use_kalman_filter = use_kalman_filter
        
        # 初始化卡尔曼滤波器
        if use_kalman_filter:
            if kalman_config is None:
                kalman_config = {
                    "process_noise_std": 0.1,
                    "measurement_noise_std": 0.5,
                    "initial_covariance": 1.0
                }
            self.kalman_filter = ExtendedKalmanFilter(**kalman_config)
        else:
            self.kalman_filter = None
        
        # 初始化可靠性评估器
        self.reliability_evaluator = ReliabilityEvaluator()
        
        # 历史数据存储
        self.position_history = deque(maxlen=100)
        self.weight_history = deque(maxlen=100)
        
        # 默认权重
        self.default_weights = {
            "gps": 0.4,
            "optical": 0.4,
            "imu": 0.2
        }
        
        self.logger = get_logger("position_fusion")
        self.logger.log_event("info", "fusion_initialized")
    
    @performance_monitor
    def fuse_positions(self, 
                      system_state: SystemState,
                      optical_result: Optional[OpticalMatchResult] = None) -> Position3D:
        """融合位置信息
        
        Args:
            system_state: 系统状态
            optical_result: 光学匹配结果
            
        Returns:
            融合后的位置
        """
        current_time = time.time()
        
        # 获取权重
        weights = self._get_fusion_weights(system_state, optical_result)
        
        # 准备位置数据
        positions = {
            "gps": system_state.position,
            "optical": optical_result.position if optical_result else None,
            "imu": system_state.position  # IMU通常与GPS结合使用
        }
        
        # 执行位置融合
        if self.use_kalman_filter and self.kalman_filter is not None:
            fused_position = self._kalman_fusion(positions, weights, current_time)
        else:
            fused_position = self._weighted_fusion(positions, weights, current_time)
        
        # 记录历史数据
        self.position_history.append(fused_position)
        self.weight_history.append(weights)
        
        # 评估融合质量
        fusion_quality = self.reliability_evaluator.evaluate_fusion_quality(
            fused_position, system_state.position, 
            optical_result.position if optical_result else None
        )
        
        self.logger.log_event("info", "position_fused", {
            "weights": weights,
            "fusion_quality": fusion_quality,
            "position": [fused_position.x, fused_position.y, fused_position.z]
        })
        
        return fused_position
    
    def _get_fusion_weights(self, 
                           system_state: SystemState,
                           optical_result: Optional[OpticalMatchResult] = None) -> Dict[str, float]:
        """获取融合权重
        
        Args:
            system_state: 系统状态
            optical_result: 光学匹配结果
            
        Returns:
            融合权重字典
        """
        if self.rl_agent is not None:
            # 使用强化学习智能体预测权重
            try:
                observation = system_state.to_observation_vector()
                action, _ = self.rl_agent.predict(observation)
                
                # 解析动作为权重
                weights = {
                    "gps": float(action[0]),
                    "optical": float(action[1]),
                    "imu": float(action[2])
                }
                
                # 归一化权重
                total_weight = sum(weights.values())
                if total_weight > 0:
                    weights = {k: v / total_weight for k, v in weights.items()}
                else:
                    weights = self.default_weights.copy()
                    
            except Exception as e:
                self.logger.log_event("warning", "rl_prediction_failed", {"error": str(e)})
                weights = self._get_adaptive_weights(system_state, optical_result)
        else:
            # 使用自适应权重分配
            weights = self._get_adaptive_weights(system_state, optical_result)
        
        return weights
    
    def _get_adaptive_weights(self, 
                             system_state: SystemState,
                             optical_result: Optional[OpticalMatchResult] = None) -> Dict[str, float]:
        """获取自适应权重
        
        Args:
            system_state: 系统状态
            optical_result: 光学匹配结果
            
        Returns:
            自适应权重字典
        """
        weights = self.default_weights.copy()
        
        # 评估GPS可靠性
        gps_reliability = self.reliability_evaluator.evaluate_ins_reliability(system_state.position)
        
        # 评估光学定位可靠性
        if optical_result is not None:
            optical_reliability = self.reliability_evaluator.evaluate_optical_reliability(optical_result)
        else:
            optical_reliability = 0.0
        
        # 根据可靠性调整权重
        total_reliability = gps_reliability + optical_reliability + 0.5  # IMU基础权重
        
        weights["gps"] = gps_reliability / total_reliability
        weights["optical"] = optical_reliability / total_reliability
        weights["imu"] = 0.5 / total_reliability
        
        # 环境因素调整
        if system_state.wind_speed > 5.0:  # 强风条件下降低光学权重
            weights["optical"] *= 0.7
            weights["gps"] += 0.15
            weights["imu"] += 0.15
        
        if system_state.battery_voltage < 14.0:  # 低电量时优先使用GPS
            weights["gps"] += 0.2
            weights["optical"] *= 0.8
        
        # 重新归一化
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _weighted_fusion(self, 
                        positions: Dict[str, Optional[Position3D]], 
                        weights: Dict[str, float],
                        timestamp: float) -> Position3D:
        """加权融合位置
        
        Args:
            positions: 位置字典
            weights: 权重字典
            timestamp: 时间戳
            
        Returns:
            融合位置
        """
        fused_x, fused_y, fused_z = 0.0, 0.0, 0.0
        total_weight = 0.0
        total_confidence = 0.0
        
        for sensor, position in positions.items():
            if position is not None and weights.get(sensor, 0.0) > 0.0:
                weight = weights[sensor]
                fused_x += position.x * weight
                fused_y += position.y * weight
                fused_z += position.z * weight
                total_weight += weight
                total_confidence += position.confidence * weight
        
        if total_weight > 0:
            fused_x /= total_weight
            fused_y /= total_weight
            fused_z /= total_weight
            total_confidence /= total_weight
        
        return Position3D(
            x=fused_x,
            y=fused_y,
            z=fused_z,
            timestamp=timestamp,
            confidence=min(1.0, total_confidence)
        )
    
    def _kalman_fusion(self, 
                      positions: Dict[str, Optional[Position3D]], 
                      weights: Dict[str, float],
                      timestamp: float) -> Position3D:
        """卡尔曼滤波融合
        
        Args:
            positions: 位置字典
            weights: 权重字典
            timestamp: 时间戳
            
        Returns:
            融合位置
        """
        # 计算时间步长
        if self.kalman_filter.last_update_time is not None:
            dt = timestamp - self.kalman_filter.last_update_time
        else:
            dt = 0.1
        
        # 预测步骤
        self.kalman_filter.predict(dt)
        
        # 准备观测数据
        measurement = np.zeros(6)
        measurement_available = False
        
        # GPS/IMU观测
        if positions["gps"] is not None:
            measurement[0:3] = positions["gps"].to_array()
            measurement_available = True
        
        # 光学观测
        if positions["optical"] is not None:
            measurement[3:6] = positions["optical"].to_array()
            measurement_available = True
        
        # 更新步骤
        if measurement_available:
            # 根据权重调整测量噪声
            R = self.kalman_filter.R.copy()
            if weights["gps"] > 0:
                R[0:3, 0:3] *= (1.0 / weights["gps"])
            if weights["optical"] > 0:
                R[3:6, 3:6] *= (1.0 / weights["optical"])
            
            self.kalman_filter.update(measurement, R)
        
        self.kalman_filter.last_update_time = timestamp
        
        return self.kalman_filter.get_position()
    
    def get_fusion_statistics(self) -> Dict[str, Any]:
        """获取融合统计信息
        
        Returns:
            统计信息字典
        """
        if len(self.position_history) == 0:
            return {}
        
        # 位置统计
        positions = np.array([[p.x, p.y, p.z] for p in self.position_history])
        position_std = np.std(positions, axis=0)
        
        # 权重统计
        if len(self.weight_history) > 0:
            weight_stats = {}
            for sensor in ["gps", "optical", "imu"]:
                weights = [w.get(sensor, 0.0) for w in self.weight_history]
                weight_stats[f"{sensor}_weight_mean"] = np.mean(weights)
                weight_stats[f"{sensor}_weight_std"] = np.std(weights)
        else:
            weight_stats = {}
        
        return {
            "num_fusions": len(self.position_history),
            "position_std_x": float(position_std[0]),
            "position_std_y": float(position_std[1]),
            "position_std_z": float(position_std[2]),
            **weight_stats
        }
    
    def reset(self) -> None:
        """重置融合器状态"""
        if self.kalman_filter is not None:
            self.kalman_filter.reset()
        
        self.position_history.clear()
        self.weight_history.clear()
        
        self.logger.log_event("info", "fusion_reset")
    
    def update_rl_agent(self, new_agent: BaseRLAgent) -> None:
        """更新强化学习智能体
        
        Args:
            new_agent: 新的智能体
        """
        self.rl_agent = new_agent
        self.logger.log_event("info", "rl_agent_updated")
    
    @property
    def info(self) -> Dict[str, Any]:
        """获取融合器信息
        
        Returns:
            融合器信息字典
        """
        return {
            "use_kalman_filter": self.use_kalman_filter,
            "has_rl_agent": self.rl_agent is not None,
            "position_history_length": len(self.position_history),
            "weight_history_length": len(self.weight_history),
            "default_weights": self.default_weights
        }