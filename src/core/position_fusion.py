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
                mean_error = np.mean(list(self.ins_error_history))
                error_reliability = max(0.0, 1.0 - mean_error / 10.0)  # 假设10米为最大可接受误差
                base_reliability = 0.7 * base_reliability + 0.3 * error_reliability
        
        return np.clip(base_reliability, 0.0, 1.0)
    
    def evaluate_optical_reliability(self, optical_result: OpticalMatchResult) -> float:
        """评估光学定位可靠性
        
        Args:
            optical_result: 光学匹配结果
            
        Returns:
            光学定位可靠性 [0.0, 1.0]
        """
        # 基于匹配得分的可靠性
        score_reliability = optical_result.match_score
        
        # 基于特征点数量的可靠性
        feature_reliability = min(1.0, optical_result.feature_points / 50.0)
        
        # 基于处理时间的可靠性（处理时间过长可能表示匹配困难）
        time_reliability = max(0.0, 1.0 - optical_result.processing_time / 1.0)
        
        # 综合可靠性
        reliability = 0.5 * score_reliability + 0.3 * feature_reliability + 0.2 * time_reliability
        
        return np.clip(reliability, 0.0, 1.0)
    
    def evaluate_fusion_quality(self, 
                               fused_pos: Position3D,
                               ins_pos: Position3D,
                               optical_result: Optional[OpticalMatchResult],
                               fusion_weights: Tuple[float, float, float]) -> float:
        """评估融合质量
        
        Args:
            fused_pos: 融合位置
            ins_pos: 惯导位置
            optical_result: 光学匹配结果
            fusion_weights: 融合权重
            
        Returns:
            融合质量 [0.0, 1.0]
        """
        # 基于权重合理性的质量评估
        lambda_ins, alpha_opt, bias = fusion_weights
        weight_quality = 1.0 - abs(lambda_ins + alpha_opt - 1.0)  # 权重和应该接近1
        
        # 基于传感器一致性的质量评估
        consistency_quality = 1.0
        if optical_result is not None:
            ins_optical_distance = ins_pos.distance_to(optical_result.position)
            consistency_quality = max(0.0, 1.0 - ins_optical_distance / 5.0)  # 5米为阈值
        
        # 基于置信度的质量评估
        confidence_quality = fused_pos.confidence
        
        # 综合质量
        quality = 0.4 * weight_quality + 0.4 * consistency_quality + 0.2 * confidence_quality
        
        return np.clip(quality, 0.0, 1.0)
    
    def get_reliability_summary(self) -> Dict[str, float]:
        """获取可靠性摘要
        
        Returns:
            可靠性摘要字典
        """
        summary = {}
        
        if self.ins_error_history:
            summary['ins_mean_error'] = np.mean(list(self.ins_error_history))
            summary['ins_std_error'] = np.std(list(self.ins_error_history))
        
        if self.optical_error_history:
            summary['optical_mean_error'] = np.mean(list(self.optical_error_history))
            summary['optical_std_error'] = np.std(list(self.optical_error_history))
        
        if self.fusion_error_history:
            summary['fusion_mean_error'] = np.mean(list(self.fusion_error_history))
            summary['fusion_std_error'] = np.std(list(self.fusion_error_history))
        
        return summary


class IntelligentPositionFusion:
    """智能位置融合器类
    
    核心功能：
    1. 动态权重分配的多传感器融合
    2. 实时可靠性评估
    3. 自适应卡尔曼滤波
    4. 强化学习驱动的融合策略优化
    """
    
    def __init__(self, 
                 rl_agent: BaseRLAgent, 
                 config: Dict[str, Any]):
        """初始化智能位置融合器
        
        Args:
            rl_agent: 强化学习智能体
            config: 融合配置参数
        """
        self.rl_agent = rl_agent
        self.config = config
        
        # 初始化卡尔曼滤波器
        self.kalman_filter = ExtendedKalmanFilter(
            process_noise_std=config.get('process_noise_std', 0.1),
            measurement_noise_std=config.get('measurement_noise_std', 0.5),
            initial_covariance=config.get('initial_covariance', 1.0)
        )
        
        # 初始化可靠性评估器
        self.reliability_evaluator = ReliabilityEvaluator(
            history_length=config.get('history_length', 50)
        )
        
        # 历史数据存储
        self.error_history = deque(maxlen=config.get('history_length', 100))
        self.fusion_history = deque(maxlen=config.get('fusion_history_length', 50))
        
        # 融合统计
        self.fusion_count = 0
        self.successful_fusions = 0
        
        self.logger = get_logger("position_fusion")
        
        self.logger.log_event("info", "fusion_initialized", config=config)
    
    @performance_monitor
    @log_function_call("position_fusion")
    def fuse_position(self, 
                     ins_pos: Position3D, 
                     optical_result: Optional[OpticalMatchResult],
                     flight_state: FlightAttitude,
                     wind_condition: Tuple[float, float] = (0.0, 0.0),
                     battery_level: float = 1.0) -> Tuple[Position3D, float]:
        """执行智能位置融合
        
        Args:
            ins_pos: 惯导位置
            optical_result: 光学定位结果
            flight_state: 飞行状态
            wind_condition: 风况信息 (速度, 方向)
            battery_level: 电池电量
            
        Returns:
            融合位置和置信度
        """
        self.fusion_count += 1
        
        # 1. 构造系统状态
        system_state = self._construct_system_state(
            ins_pos, optical_result, flight_state, wind_condition, battery_level
        )
        
        # 2. RL智能体决策
        action = self.rl_agent.predict(system_state)
        
        # 3. 执行动态权重融合
        fused_position = self._execute_fusion(ins_pos, optical_result, action)
        
        # 4. 可靠性评估
        confidence = self._evaluate_reliability(system_state, action, fused_position)
        
        # 5. 更新卡尔曼滤波器
        if action.update_decision and confidence > action.confidence_threshold:
            self._update_kalman_filter(ins_pos, optical_result, fused_position)
            self.successful_fusions += 1
        
        # 6. 记录融合历史
        self._update_history(fused_position, confidence, action)
        
        # 7. 记录日志
        self.logger.log_event(
            "debug",
            "position_fused",
            fusion_weights=action.fusion_weights,
            confidence=confidence,
            control_mode=action.control_mode.value,
            update_decision=action.update_decision
        )
        
        return fused_position, confidence
    
    def _construct_system_state(self,
                               ins_pos: Position3D,
                               optical_result: Optional[OpticalMatchResult],
                               flight_state: FlightAttitude,
                               wind_condition: Tuple[float, float],
                               battery_level: float) -> SystemState:
        """构造系统状态
        
        Args:
            ins_pos: 惯导位置
            optical_result: 光学匹配结果
            flight_state: 飞行状态
            wind_condition: 风况信息
            battery_level: 电池电量
            
        Returns:
            系统状态对象
        """
        # 计算管道偏离距离（简化计算）
        pipeline_deviation = abs(ins_pos.y)  # 假设管道沿X轴
        
        # 计算历史误差
        historical_error = np.mean(list(self.error_history)) if self.error_history else 0.0
        
        # 获取速度估计
        velocity_vector = self.kalman_filter.get_velocity()
        
        from ..utils.data_types import VelocityVector
        velocity = VelocityVector(
            vx=velocity_vector[0],
            vy=velocity_vector[1],
            vz=velocity_vector[2],
            timestamp=time.time()
        )
        
        return SystemState(
            inertial_position=ins_pos,
            optical_position=optical_result,
            flight_attitude=flight_state,
            velocity=velocity,
            pipeline_deviation=pipeline_deviation,
            battery_level=battery_level,
            wind_condition=wind_condition,
            historical_error=historical_error,
            control_mode=ControlMode.NORMAL  # 初始模式
        )
    
    def _execute_fusion(self, 
                       ins_pos: Position3D,
                       optical_result: Optional[OpticalMatchResult],
                       action: RLAction) -> Position3D:
        """执行加权融合算法
        
        Args:
            ins_pos: 惯导位置
            optical_result: 光学匹配结果
            action: RL动作
            
        Returns:
            融合位置
        """
        lambda_ins, alpha_opt, bias = action.fusion_weights
        
        if optical_result is None or not optical_result.is_valid():
            # 仅使用惯导数据
            fused_position = Position3D(
                x=ins_pos.x + bias,
                y=ins_pos.y,
                z=ins_pos.z,
                timestamp=ins_pos.timestamp,
                confidence=ins_pos.confidence * 0.5  # 降低置信度
            )
        else:
            # 加权融合
            # 归一化权重
            total_weight = lambda_ins + alpha_opt
            if total_weight > 0:
                lambda_ins_norm = lambda_ins / total_weight
                alpha_opt_norm = alpha_opt / total_weight
            else:
                lambda_ins_norm = 0.5
                alpha_opt_norm = 0.5
            
            fused_x = lambda_ins_norm * ins_pos.x + alpha_opt_norm * optical_result.position.x + bias
            fused_y = lambda_ins_norm * ins_pos.y + alpha_opt_norm * optical_result.position.y
            fused_z = lambda_ins_norm * ins_pos.z + alpha_opt_norm * optical_result.position.z
            
            # 置信度计算
            confidence = self._calculate_fusion_confidence(ins_pos, optical_result, action)
            
            fused_position = Position3D(
                x=fused_x, y=fused_y, z=fused_z,
                timestamp=max(ins_pos.timestamp, optical_result.position.timestamp),
                confidence=confidence
            )
        
        return fused_position
    
    def _calculate_fusion_confidence(self,
                                   ins_pos: Position3D,
                                   optical_result: OpticalMatchResult,
                                   action: RLAction) -> float:
        """计算融合置信度
        
        Args:
            ins_pos: 惯导位置
            optical_result: 光学匹配结果
            action: RL动作
            
        Returns:
            融合置信度
        """
        # 基础置信度
        ins_confidence = ins_pos.confidence
        optical_confidence = optical_result.match_score
        
        # 权重归一化
        lambda_ins, alpha_opt, _ = action.fusion_weights
        total_weight = lambda_ins + alpha_opt
        if total_weight > 0:
            lambda_ins_norm = lambda_ins / total_weight
            alpha_opt_norm = alpha_opt / total_weight
        else:
            lambda_ins_norm = 0.5
            alpha_opt_norm = 0.5
        
        # 加权置信度
        weighted_confidence = lambda_ins_norm * ins_confidence + alpha_opt_norm * optical_confidence
        
        # 一致性奖励
        consistency_distance = ins_pos.distance_to(optical_result.position)
        consistency_bonus = max(0.0, 1.0 - consistency_distance / 5.0) * 0.1
        
        final_confidence = min(1.0, weighted_confidence + consistency_bonus)
        
        return final_confidence
    
    def _evaluate_reliability(self,
                            system_state: SystemState,
                            action: RLAction,
                            fused_position: Position3D) -> float:
        """评估融合可靠性
        
        Args:
            system_state: 系统状态
            action: RL动作
            fused_position: 融合位置
            
        Returns:
            可靠性评估结果
        """
        # 惯导可靠性
        ins_reliability = self.reliability_evaluator.evaluate_ins_reliability(
            system_state.inertial_position
        )
        
        # 光学可靠性
        optical_reliability = 0.0
        if system_state.optical_position is not None:
            optical_reliability = self.reliability_evaluator.evaluate_optical_reliability(
                system_state.optical_position
            )
        
        # 融合质量
        fusion_quality = self.reliability_evaluator.evaluate_fusion_quality(
            fused_position,
            system_state.inertial_position,
            system_state.optical_position,
            action.fusion_weights
        )
        
        # 综合可靠性
        reliability = 0.4 * ins_reliability + 0.3 * optical_reliability + 0.3 * fusion_quality
        
        return np.clip(reliability, 0.0, 1.0)
    
    def _update_kalman_filter(self,
                             ins_pos: Position3D,
                             optical_result: Optional[OpticalMatchResult],
                             fused_pos: Position3D) -> None:
        """更新卡尔曼滤波器
        
        Args:
            ins_pos: 惯导位置
            optical_result: 光学匹配结果
            fused_pos: 融合位置
        """
        # 构造测量向量
        if optical_result is not None:
            measurement = np.array([
                ins_pos.x, ins_pos.y, ins_pos.z,
                optical_result.position.x, optical_result.position.y, optical_result.position.z
            ])
        else:
            measurement = np.array([
                ins_pos.x, ins_pos.y, ins_pos.z,
                ins_pos.x, ins_pos.y, ins_pos.z  # 使用惯导数据填充
            ])
        
        # 预测步骤
        current_time = time.time()
        if self.kalman_filter.last_update_time is not None:
            dt = current_time - self.kalman_filter.last_update_time
        else:
            dt = 0.1
        
        self.kalman_filter.predict(dt)
        
        # 更新步骤
        self.kalman_filter.update(measurement)
        self.kalman_filter.last_update_time = current_time
    
    def _update_history(self,
                       fused_position: Position3D,
                       confidence: float,
                       action: RLAction) -> None:
        """更新融合历史
        
        Args:
            fused_position: 融合位置
            confidence: 置信度
            action: RL动作
        """
        # 记录融合历史
        fusion_record = {
            'position': fused_position,
            'confidence': confidence,
            'action': action,
            'timestamp': time.time()
        }
        
        self.fusion_history.append(fusion_record)
        
        # 如果有真实位置参考，计算误差
        # 这里简化处理，实际应用中需要真实的参考位置
        # estimated_error = 0.0  # 占位符
        # self.error_history.append(estimated_error)
    
    def get_fusion_statistics(self) -> Dict[str, Any]:
        """获取融合统计信息
        
        Returns:
            融合统计字典
        """
        success_rate = self.successful_fusions / max(1, self.fusion_count)
        
        statistics = {
            'total_fusions': self.fusion_count,
            'successful_fusions': self.successful_fusions,
            'success_rate': success_rate,
            'reliability_summary': self.reliability_evaluator.get_reliability_summary()
        }
        
        if self.fusion_history:
            recent_confidences = [record['confidence'] for record in list(self.fusion_history)[-10:]]
            statistics['recent_mean_confidence'] = np.mean(recent_confidences)
            statistics['recent_std_confidence'] = np.std(recent_confidences)
        
        return statistics
    
    def reset(self) -> None:
        """重置融合器状态"""
        self.kalman_filter.reset()
        self.error_history.clear()
        self.fusion_history.clear()
        self.fusion_count = 0
        self.successful_fusions = 0
        
        self.logger.log_event("info", "fusion_reset")
    
    def get_current_position_estimate(self) -> Position3D:
        """获取当前位置估计
        
        Returns:
            当前位置估计
        """
        return self.kalman_filter.get_position()