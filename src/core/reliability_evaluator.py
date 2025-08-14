#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强化学习无人机定位导航系统 - 可靠性评估模块

本模块实现传感器数据和融合结果的可靠性评估算法，包括：
1. 传感器数据质量评估
2. 融合结果置信度计算
3. 异常检测和故障诊断
4. 动态权重调整

Author: wdblink
Date: 2024
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from collections import deque
import statistics

# 导入项目模块
from ..utils.data_types import (
    Position3D, SensorData, OpticalMatchResult, 
    SystemState, ReliabilityMetrics
)
from ..utils.logger import logger_manager, performance_monitor


class SensorType(Enum):
    """传感器类型枚举"""
    IMU = "imu"
    GPS = "gps"
    OPTICAL = "optical"
    BAROMETER = "barometer"
    MAGNETOMETER = "magnetometer"


class ReliabilityLevel(Enum):
    """可靠性等级枚举"""
    VERY_HIGH = "very_high"  # 0.9-1.0
    HIGH = "high"           # 0.7-0.9
    MEDIUM = "medium"       # 0.5-0.7
    LOW = "low"             # 0.3-0.5
    VERY_LOW = "very_low"   # 0.0-0.3


@dataclass
class SensorReliability:
    """传感器可靠性评估结果"""
    sensor_type: SensorType
    reliability_score: float  # 0.0-1.0
    reliability_level: ReliabilityLevel
    confidence: float
    noise_level: float
    drift_rate: float
    last_update_time: float
    failure_count: int
    quality_metrics: Dict[str, float]


@dataclass
class FusionReliability:
    """融合结果可靠性评估"""
    overall_score: float
    position_confidence: float
    velocity_confidence: float
    attitude_confidence: float
    sensor_weights: Dict[SensorType, float]
    anomaly_detected: bool
    consistency_score: float
    timestamp: float


class ReliabilityEvaluator:
    """可靠性评估器
    
    实现多传感器数据和融合结果的可靠性评估，包括：
    1. 传感器数据质量分析
    2. 融合一致性检查
    3. 异常检测算法
    4. 动态权重计算
    5. 故障诊断功能
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化可靠性评估器
        
        Args:
            config: 配置参数字典
        """
        self.config = config
        
        # 传感器可靠性历史
        self.sensor_reliability_history = {
            sensor_type: deque(maxlen=config.get('history_length', 100))
            for sensor_type in SensorType
        }
        
        # 融合可靠性历史
        self.fusion_reliability_history = deque(maxlen=config.get('fusion_history_length', 50))
        
        # 传感器基线参数
        self.sensor_baselines = {
            SensorType.IMU: {
                'noise_threshold': config.get('imu_noise_threshold', 0.1),
                'drift_threshold': config.get('imu_drift_threshold', 0.01),
                'update_rate': config.get('imu_update_rate', 100.0)
            },
            SensorType.GPS: {
                'noise_threshold': config.get('gps_noise_threshold', 2.0),
                'drift_threshold': config.get('gps_drift_threshold', 0.5),
                'update_rate': config.get('gps_update_rate', 10.0)
            },
            SensorType.OPTICAL: {
                'match_threshold': config.get('optical_match_threshold', 0.5),
                'feature_threshold': config.get('optical_feature_threshold', 50),
                'update_rate': config.get('optical_update_rate', 30.0)
            }
        }
        
        # 异常检测参数
        self.anomaly_thresholds = {
            'position_jump': config.get('position_jump_threshold', 10.0),  # 米
            'velocity_jump': config.get('velocity_jump_threshold', 5.0),   # m/s
            'acceleration_limit': config.get('acceleration_limit', 20.0),  # m/s²
            'consistency_threshold': config.get('consistency_threshold', 0.3)
        }
        
        # 权重计算参数
        self.weight_config = {
            'base_weights': config.get('base_sensor_weights', {
                SensorType.IMU: 0.3,
                SensorType.GPS: 0.4,
                SensorType.OPTICAL: 0.3
            }),
            'adaptation_rate': config.get('weight_adaptation_rate', 0.1),
            'min_weight': config.get('min_sensor_weight', 0.05),
            'max_weight': config.get('max_sensor_weight', 0.8)
        }
        
        # 统计信息
        self.evaluation_count = 0
        self.anomaly_count = 0
        self.last_evaluation_time = 0
        
        logger_manager.info("可靠性评估器初始化完成")
    
    @performance_monitor
    def evaluate_sensor_reliability(self, sensor_data: SensorData) -> SensorReliability:
        """
        评估单个传感器的可靠性
        
        Args:
            sensor_data: 传感器数据
            
        Returns:
            传感器可靠性评估结果
        """
        sensor_type = self._get_sensor_type(sensor_data)
        current_time = time.time()
        
        # 1. 数据质量评估
        quality_metrics = self._assess_data_quality(sensor_data, sensor_type)
        
        # 2. 噪声水平评估
        noise_level = self._calculate_noise_level(sensor_data, sensor_type)
        
        # 3. 漂移率评估
        drift_rate = self._calculate_drift_rate(sensor_data, sensor_type)
        
        # 4. 更新频率检查
        update_reliability = self._check_update_frequency(sensor_data, sensor_type)
        
        # 5. 综合可靠性评分
        reliability_score = self._calculate_reliability_score(
            quality_metrics, noise_level, drift_rate, update_reliability
        )
        
        # 6. 确定可靠性等级
        reliability_level = self._get_reliability_level(reliability_score)
        
        # 7. 计算置信度
        confidence = self._calculate_confidence(sensor_data, reliability_score)
        
        # 8. 故障计数
        failure_count = self._update_failure_count(sensor_type, reliability_score)
        
        # 创建可靠性评估结果
        reliability = SensorReliability(
            sensor_type=sensor_type,
            reliability_score=reliability_score,
            reliability_level=reliability_level,
            confidence=confidence,
            noise_level=noise_level,
            drift_rate=drift_rate,
            last_update_time=current_time,
            failure_count=failure_count,
            quality_metrics=quality_metrics
        )
        
        # 更新历史记录
        self.sensor_reliability_history[sensor_type].append(reliability)
        
        return reliability
    
    @performance_monitor
    def evaluate_fusion_reliability(self, 
                                  fusion_result: Position3D,
                                  sensor_reliabilities: Dict[SensorType, SensorReliability],
                                  system_state: SystemState) -> FusionReliability:
        """
        评估融合结果的可靠性
        
        Args:
            fusion_result: 融合位置结果
            sensor_reliabilities: 各传感器可靠性评估
            system_state: 系统状态
            
        Returns:
            融合可靠性评估结果
        """
        current_time = time.time()
        
        # 1. 传感器一致性检查
        consistency_score = self._check_sensor_consistency(sensor_reliabilities, system_state)
        
        # 2. 异常检测
        anomaly_detected = self._detect_anomalies(fusion_result, system_state)
        
        # 3. 计算传感器权重
        sensor_weights = self._calculate_dynamic_weights(sensor_reliabilities)
        
        # 4. 位置置信度评估
        position_confidence = self._evaluate_position_confidence(
            fusion_result, sensor_reliabilities, consistency_score
        )
        
        # 5. 速度置信度评估
        velocity_confidence = self._evaluate_velocity_confidence(
            system_state, sensor_reliabilities
        )
        
        # 6. 姿态置信度评估
        attitude_confidence = self._evaluate_attitude_confidence(
            system_state, sensor_reliabilities
        )
        
        # 7. 综合可靠性评分
        overall_score = self._calculate_overall_reliability(
            position_confidence, velocity_confidence, attitude_confidence, 
            consistency_score, anomaly_detected
        )
        
        # 创建融合可靠性评估结果
        fusion_reliability = FusionReliability(
            overall_score=overall_score,
            position_confidence=position_confidence,
            velocity_confidence=velocity_confidence,
            attitude_confidence=attitude_confidence,
            sensor_weights=sensor_weights,
            anomaly_detected=anomaly_detected,
            consistency_score=consistency_score,
            timestamp=current_time
        )
        
        # 更新历史记录
        self.fusion_reliability_history.append(fusion_reliability)
        self.evaluation_count += 1
        self.last_evaluation_time = current_time
        
        if anomaly_detected:
            self.anomaly_count += 1
            logger_manager.warning(f"检测到融合异常，总体可靠性: {overall_score:.3f}")
        
        return fusion_reliability
    
    def _get_sensor_type(self, sensor_data: SensorData) -> SensorType:
        """
        根据传感器数据确定传感器类型
        
        Args:
            sensor_data: 传感器数据
            
        Returns:
            传感器类型
        """
        if hasattr(sensor_data, 'acceleration') and sensor_data.acceleration is not None:
            return SensorType.IMU
        elif hasattr(sensor_data, 'latitude') and sensor_data.latitude is not None:
            return SensorType.GPS
        elif hasattr(sensor_data, 'match_score') and sensor_data.match_score is not None:
            return SensorType.OPTICAL
        else:
            return SensorType.IMU  # 默认类型
    
    def _assess_data_quality(self, sensor_data: SensorData, sensor_type: SensorType) -> Dict[str, float]:
        """
        评估传感器数据质量
        
        Args:
            sensor_data: 传感器数据
            sensor_type: 传感器类型
            
        Returns:
            质量指标字典
        """
        quality_metrics = {}
        
        if sensor_type == SensorType.IMU:
            # IMU数据质量评估
            if hasattr(sensor_data, 'acceleration'):
                acc_magnitude = np.linalg.norm(sensor_data.acceleration)
                quality_metrics['acceleration_magnitude'] = acc_magnitude
                quality_metrics['acceleration_validity'] = 1.0 if 8.0 < acc_magnitude < 12.0 else 0.5
            
            if hasattr(sensor_data, 'angular_velocity'):
                gyro_magnitude = np.linalg.norm(sensor_data.angular_velocity)
                quality_metrics['gyro_magnitude'] = gyro_magnitude
                quality_metrics['gyro_validity'] = 1.0 if gyro_magnitude < 10.0 else 0.5
        
        elif sensor_type == SensorType.GPS:
            # GPS数据质量评估
            if hasattr(sensor_data, 'hdop'):
                quality_metrics['hdop'] = sensor_data.hdop
                quality_metrics['hdop_quality'] = max(0.0, 1.0 - sensor_data.hdop / 5.0)
            
            if hasattr(sensor_data, 'satellite_count'):
                quality_metrics['satellite_count'] = sensor_data.satellite_count
                quality_metrics['satellite_quality'] = min(1.0, sensor_data.satellite_count / 8.0)
        
        elif sensor_type == SensorType.OPTICAL:
            # 光学数据质量评估
            if hasattr(sensor_data, 'match_score'):
                quality_metrics['match_score'] = sensor_data.match_score
                quality_metrics['match_quality'] = sensor_data.match_score
            
            if hasattr(sensor_data, 'feature_count'):
                quality_metrics['feature_count'] = sensor_data.feature_count
                quality_metrics['feature_quality'] = min(1.0, sensor_data.feature_count / 100.0)
        
        return quality_metrics
    
    def _calculate_noise_level(self, sensor_data: SensorData, sensor_type: SensorType) -> float:
        """
        计算传感器噪声水平
        
        Args:
            sensor_data: 传感器数据
            sensor_type: 传感器类型
            
        Returns:
            噪声水平 (0.0-1.0)
        """
        history = self.sensor_reliability_history[sensor_type]
        
        if len(history) < 5:
            return 0.5  # 默认中等噪声水平
        
        # 计算最近数据的标准差作为噪声指标
        recent_scores = [r.reliability_score for r in list(history)[-10:]]
        noise_std = statistics.stdev(recent_scores) if len(recent_scores) > 1 else 0.0
        
        # 归一化噪声水平
        baseline_threshold = self.sensor_baselines[sensor_type]['noise_threshold']
        noise_level = min(1.0, noise_std / baseline_threshold)
        
        return noise_level
    
    def _calculate_drift_rate(self, sensor_data: SensorData, sensor_type: SensorType) -> float:
        """
        计算传感器漂移率
        
        Args:
            sensor_data: 传感器数据
            sensor_type: 传感器类型
            
        Returns:
            漂移率 (0.0-1.0)
        """
        history = self.sensor_reliability_history[sensor_type]
        
        if len(history) < 10:
            return 0.0  # 数据不足，假设无漂移
        
        # 计算可靠性评分的趋势
        recent_scores = [r.reliability_score for r in list(history)[-20:]]
        
        # 使用线性回归计算趋势斜率
        x = np.arange(len(recent_scores))
        coeffs = np.polyfit(x, recent_scores, 1)
        drift_slope = abs(coeffs[0])  # 斜率的绝对值
        
        # 归一化漂移率
        baseline_threshold = self.sensor_baselines[sensor_type]['drift_threshold']
        drift_rate = min(1.0, drift_slope / baseline_threshold)
        
        return drift_rate
    
    def _check_update_frequency(self, sensor_data: SensorData, sensor_type: SensorType) -> float:
        """
        检查传感器更新频率
        
        Args:
            sensor_data: 传感器数据
            sensor_type: 传感器类型
            
        Returns:
            更新频率可靠性 (0.0-1.0)
        """
        current_time = time.time()
        expected_rate = self.sensor_baselines[sensor_type]['update_rate']
        expected_interval = 1.0 / expected_rate
        
        if hasattr(sensor_data, 'timestamp'):
            time_since_last = current_time - sensor_data.timestamp
            
            # 计算更新频率可靠性
            if time_since_last <= expected_interval * 1.5:
                return 1.0  # 正常更新
            elif time_since_last <= expected_interval * 3.0:
                return 0.7  # 稍慢
            elif time_since_last <= expected_interval * 5.0:
                return 0.4  # 较慢
            else:
                return 0.1  # 很慢或停止更新
        
        return 0.5  # 无时间戳信息
    
    def _calculate_reliability_score(self, quality_metrics: Dict[str, float], 
                                   noise_level: float, drift_rate: float, 
                                   update_reliability: float) -> float:
        """
        计算综合可靠性评分
        
        Args:
            quality_metrics: 质量指标
            noise_level: 噪声水平
            drift_rate: 漂移率
            update_reliability: 更新频率可靠性
            
        Returns:
            可靠性评分 (0.0-1.0)
        """
        # 计算质量指标平均值
        quality_scores = [v for k, v in quality_metrics.items() if 'quality' in k or 'validity' in k]
        avg_quality = statistics.mean(quality_scores) if quality_scores else 0.5
        
        # 综合评分计算
        reliability_score = (
            avg_quality * 0.4 +           # 数据质量权重40%
            (1.0 - noise_level) * 0.25 +  # 噪声水平权重25%
            (1.0 - drift_rate) * 0.15 +   # 漂移率权重15%
            update_reliability * 0.2       # 更新频率权重20%
        )
        
        return np.clip(reliability_score, 0.0, 1.0)
    
    def _get_reliability_level(self, score: float) -> ReliabilityLevel:
        """
        根据评分确定可靠性等级
        
        Args:
            score: 可靠性评分
            
        Returns:
            可靠性等级
        """
        if score >= 0.9:
            return ReliabilityLevel.VERY_HIGH
        elif score >= 0.7:
            return ReliabilityLevel.HIGH
        elif score >= 0.5:
            return ReliabilityLevel.MEDIUM
        elif score >= 0.3:
            return ReliabilityLevel.LOW
        else:
            return ReliabilityLevel.VERY_LOW
    
    def _calculate_confidence(self, sensor_data: SensorData, reliability_score: float) -> float:
        """
        计算传感器数据置信度
        
        Args:
            sensor_data: 传感器数据
            reliability_score: 可靠性评分
            
        Returns:
            置信度 (0.0-1.0)
        """
        # 基础置信度基于可靠性评分
        base_confidence = reliability_score
        
        # 根据传感器特定信息调整置信度
        if hasattr(sensor_data, 'confidence'):
            sensor_confidence = sensor_data.confidence
            # 加权平均
            confidence = base_confidence * 0.7 + sensor_confidence * 0.3
        else:
            confidence = base_confidence
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _update_failure_count(self, sensor_type: SensorType, reliability_score: float) -> int:
        """
        更新传感器故障计数
        
        Args:
            sensor_type: 传感器类型
            reliability_score: 可靠性评分
            
        Returns:
            故障计数
        """
        history = self.sensor_reliability_history[sensor_type]
        
        if not history:
            return 0
        
        # 统计最近的低可靠性事件
        recent_reliabilities = [r.reliability_score for r in list(history)[-10:]]
        failure_count = sum(1 for score in recent_reliabilities if score < 0.3)
        
        return failure_count
    
    def _check_sensor_consistency(self, sensor_reliabilities: Dict[SensorType, SensorReliability],
                                system_state: SystemState) -> float:
        """
        检查传感器间一致性
        
        Args:
            sensor_reliabilities: 传感器可靠性评估
            system_state: 系统状态
            
        Returns:
            一致性评分 (0.0-1.0)
        """
        # 获取各传感器位置估计
        positions = []
        
        if system_state.inertial_position:
            positions.append(system_state.inertial_position)
        if system_state.gps_position:
            positions.append(system_state.gps_position)
        if system_state.optical_position:
            positions.append(system_state.optical_position)
        
        if len(positions) < 2:
            return 1.0  # 单一传感器，假设一致
        
        # 计算位置差异
        max_distance = 0.0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distance = np.sqrt(
                    (positions[i].x - positions[j].x) ** 2 +
                    (positions[i].y - positions[j].y) ** 2 +
                    (positions[i].z - positions[j].z) ** 2
                )
                max_distance = max(max_distance, distance)
        
        # 一致性评分（距离越小，一致性越高）
        consistency_threshold = self.anomaly_thresholds['consistency_threshold']
        consistency_score = max(0.0, 1.0 - max_distance / (consistency_threshold * 10))
        
        return consistency_score
    
    def _detect_anomalies(self, fusion_result: Position3D, system_state: SystemState) -> bool:
        """
        检测融合结果异常
        
        Args:
            fusion_result: 融合位置结果
            system_state: 系统状态
            
        Returns:
            是否检测到异常
        """
        if not self.fusion_reliability_history:
            return False
        
        last_fusion = self.fusion_reliability_history[-1]
        
        # 检查位置跳跃
        if hasattr(system_state, 'last_position') and system_state.last_position:
            position_jump = np.sqrt(
                (fusion_result.x - system_state.last_position.x) ** 2 +
                (fusion_result.y - system_state.last_position.y) ** 2 +
                (fusion_result.z - system_state.last_position.z) ** 2
            )
            
            if position_jump > self.anomaly_thresholds['position_jump']:
                logger_manager.warning(f"检测到位置跳跃异常: {position_jump:.2f}m")
                return True
        
        # 检查速度跳跃
        if hasattr(system_state, 'velocity') and system_state.velocity:
            velocity_magnitude = np.linalg.norm(system_state.velocity)
            if velocity_magnitude > self.anomaly_thresholds['velocity_jump']:
                logger_manager.warning(f"检测到速度异常: {velocity_magnitude:.2f}m/s")
                return True
        
        # 检查可靠性突然下降
        if (len(self.fusion_reliability_history) > 1 and
            last_fusion.overall_score - self.fusion_reliability_history[-2].overall_score < -0.3):
            logger_manager.warning("检测到可靠性突然下降")
            return True
        
        return False
    
    def _calculate_dynamic_weights(self, sensor_reliabilities: Dict[SensorType, SensorReliability]) -> Dict[SensorType, float]:
        """
        计算动态传感器权重
        
        Args:
            sensor_reliabilities: 传感器可靠性评估
            
        Returns:
            传感器权重字典
        """
        base_weights = self.weight_config['base_weights']
        adaptation_rate = self.weight_config['adaptation_rate']
        min_weight = self.weight_config['min_weight']
        max_weight = self.weight_config['max_weight']
        
        # 计算可靠性权重
        reliability_weights = {}
        total_reliability = 0.0
        
        for sensor_type, reliability in sensor_reliabilities.items():
            if sensor_type in base_weights:
                reliability_weights[sensor_type] = reliability.reliability_score
                total_reliability += reliability.reliability_score
        
        # 归一化可靠性权重
        if total_reliability > 0:
            for sensor_type in reliability_weights:
                reliability_weights[sensor_type] /= total_reliability
        
        # 动态调整权重
        dynamic_weights = {}
        for sensor_type in base_weights:
            if sensor_type in reliability_weights:
                # 基础权重与可靠性权重的加权平均
                new_weight = (
                    base_weights[sensor_type] * (1 - adaptation_rate) +
                    reliability_weights[sensor_type] * adaptation_rate
                )
                # 应用权重限制
                dynamic_weights[sensor_type] = np.clip(new_weight, min_weight, max_weight)
            else:
                dynamic_weights[sensor_type] = min_weight
        
        # 重新归一化权重
        total_weight = sum(dynamic_weights.values())
        if total_weight > 0:
            for sensor_type in dynamic_weights:
                dynamic_weights[sensor_type] /= total_weight
        
        return dynamic_weights
    
    def _evaluate_position_confidence(self, fusion_result: Position3D,
                                    sensor_reliabilities: Dict[SensorType, SensorReliability],
                                    consistency_score: float) -> float:
        """
        评估位置置信度
        
        Args:
            fusion_result: 融合位置结果
            sensor_reliabilities: 传感器可靠性评估
            consistency_score: 一致性评分
            
        Returns:
            位置置信度
        """
        # 基于传感器可靠性的加权平均
        weighted_reliability = 0.0
        total_weight = 0.0
        
        for sensor_type, reliability in sensor_reliabilities.items():
            if sensor_type in [SensorType.GPS, SensorType.OPTICAL, SensorType.IMU]:
                weight = self.weight_config['base_weights'].get(sensor_type, 0.0)
                weighted_reliability += reliability.reliability_score * weight
                total_weight += weight
        
        if total_weight > 0:
            avg_reliability = weighted_reliability / total_weight
        else:
            avg_reliability = 0.5
        
        # 结合一致性评分
        position_confidence = avg_reliability * 0.7 + consistency_score * 0.3
        
        return np.clip(position_confidence, 0.0, 1.0)
    
    def _evaluate_velocity_confidence(self, system_state: SystemState,
                                    sensor_reliabilities: Dict[SensorType, SensorReliability]) -> float:
        """
        评估速度置信度
        
        Args:
            system_state: 系统状态
            sensor_reliabilities: 传感器可靠性评估
            
        Returns:
            速度置信度
        """
        # 主要基于IMU和GPS的可靠性
        imu_reliability = sensor_reliabilities.get(SensorType.IMU)
        gps_reliability = sensor_reliabilities.get(SensorType.GPS)
        
        if imu_reliability and gps_reliability:
            velocity_confidence = (imu_reliability.reliability_score * 0.6 + 
                                 gps_reliability.reliability_score * 0.4)
        elif imu_reliability:
            velocity_confidence = imu_reliability.reliability_score * 0.8
        elif gps_reliability:
            velocity_confidence = gps_reliability.reliability_score * 0.6
        else:
            velocity_confidence = 0.3
        
        return np.clip(velocity_confidence, 0.0, 1.0)
    
    def _evaluate_attitude_confidence(self, system_state: SystemState,
                                    sensor_reliabilities: Dict[SensorType, SensorReliability]) -> float:
        """
        评估姿态置信度
        
        Args:
            system_state: 系统状态
            sensor_reliabilities: 传感器可靠性评估
            
        Returns:
            姿态置信度
        """
        # 主要基于IMU的可靠性
        imu_reliability = sensor_reliabilities.get(SensorType.IMU)
        
        if imu_reliability:
            attitude_confidence = imu_reliability.reliability_score
        else:
            attitude_confidence = 0.3
        
        return np.clip(attitude_confidence, 0.0, 1.0)
    
    def _calculate_overall_reliability(self, position_confidence: float,
                                     velocity_confidence: float,
                                     attitude_confidence: float,
                                     consistency_score: float,
                                     anomaly_detected: bool) -> float:
        """
        计算总体可靠性评分
        
        Args:
            position_confidence: 位置置信度
            velocity_confidence: 速度置信度
            attitude_confidence: 姿态置信度
            consistency_score: 一致性评分
            anomaly_detected: 是否检测到异常
            
        Returns:
            总体可靠性评分
        """
        # 加权平均计算总体评分
        overall_score = (
            position_confidence * 0.4 +
            velocity_confidence * 0.25 +
            attitude_confidence * 0.2 +
            consistency_score * 0.15
        )
        
        # 异常检测惩罚
        if anomaly_detected:
            overall_score *= 0.5
        
        return np.clip(overall_score, 0.0, 1.0)
    
    def get_reliability_statistics(self) -> Dict[str, Any]:
        """
        获取可靠性统计信息
        
        Returns:
            可靠性统计信息字典
        """
        stats = {
            'evaluation_count': self.evaluation_count,
            'anomaly_count': self.anomaly_count,
            'anomaly_rate': self.anomaly_count / max(1, self.evaluation_count),
            'last_evaluation_time': self.last_evaluation_time
        }
        
        # 各传感器统计
        for sensor_type in SensorType:
            history = self.sensor_reliability_history[sensor_type]
            if history:
                recent_scores = [r.reliability_score for r in list(history)[-10:]]
                stats[f'{sensor_type.value}_avg_reliability'] = statistics.mean(recent_scores)
                stats[f'{sensor_type.value}_reliability_std'] = statistics.stdev(recent_scores) if len(recent_scores) > 1 else 0.0
                stats[f'{sensor_type.value}_failure_rate'] = sum(1 for s in recent_scores if s < 0.3) / len(recent_scores)
        
        # 融合统计
        if self.fusion_reliability_history:
            recent_fusion = list(self.fusion_reliability_history)[-10:]
            stats['fusion_avg_reliability'] = statistics.mean([f.overall_score for f in recent_fusion])
            stats['fusion_consistency_avg'] = statistics.mean([f.consistency_score for f in recent_fusion])
        
        return stats
    
    def reset_statistics(self) -> None:
        """
        重置统计信息
        """
        self.evaluation_count = 0
        self.anomaly_count = 0
        self.last_evaluation_time = 0
        
        for sensor_type in SensorType:
            self.sensor_reliability_history[sensor_type].clear()
        
        self.fusion_reliability_history.clear()
        
        logger_manager.info("可靠性评估统计信息已重置")