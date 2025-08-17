#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多元化奖励函数模块

该模块实现了针对无人机强化学习导航的多元化奖励函数，包括：
1. 循迹能力奖励 - 评估沿预定航线飞行的精确度
2. 寻回定位能力奖励 - 评估偏离后的自主寻回能力
3. 紧急决策能力奖励 - 评估紧急情况下的最优决策能力
4. 安全性奖励 - 评估飞行安全性
5. 能效奖励 - 评估能源使用效率

Author: AI Assistant
Date: 2024
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import math

from ..utils.data_types import Position3D, FlightAttitude, VelocityVector


class TaskPhase(Enum):
    """任务阶段枚举"""
    NORMAL = "normal"      # 正常飞行阶段
    RECOVERY = "recovery"  # 恢复阶段
    EMERGENCY = "emergency" # 紧急阶段


class DecisionType(Enum):
    """决策类型枚举"""
    CONTINUE = "continue"                    # 继续原计划
    DIRECT_TO_TARGET = "direct_to_target"    # 直飞到目标
    EMERGENCY_LAND = "emergency_land"        # 紧急降落
    RETURN_TO_LAUNCH = "return_to_launch"    # 返回起飞点


@dataclass
class FlightStatus:
    """飞行状态信息"""
    planned_trajectory: List[Position3D]  # 预定轨迹
    target_position: Position3D           # 目标位置
    trajectory_tolerance: float = 5.0     # 轨迹容忍度(米)
    deviation_distance: float = 0.0       # 偏离距离
    recovery_time: float = 0.0            # 寻回时间
    recovery_success: bool = True         # 寻回成功标志
    battery_level: float = 1.0            # 电池电量(0-1)
    is_collision: bool = False            # 碰撞标志
    altitude_limit: Tuple[float, float] = (5.0, 50.0)  # 高度限制(最低, 最高)
    speed_limit: float = 15.0             # 速度限制
    wind_speed: float = 0.0               # 风速
    visibility: float = 1.0               # 能见度(0-1)


@dataclass
class RewardComponents:
    """奖励组件"""
    tracking: float = 0.0      # 循迹奖励
    recovery: float = 0.0      # 寻回奖励
    emergency: float = 0.0     # 紧急决策奖励
    safety: float = 0.0        # 安全性奖励
    efficiency: float = 0.0    # 能效奖励
    total: float = 0.0         # 总奖励


class MultiDimensionalRewardFunction:
    """多元化奖励函数类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化奖励函数
        
        Args:
            config: 奖励函数配置参数
        """
        self.config = config or self._get_default_config()
        
        # 历史状态记录
        self.position_history: List[Position3D] = []
        self.deviation_start_time: Optional[float] = None
        self.last_battery_level: float = 1.0
        self.cumulative_energy_consumption: float = 0.0
        
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'weights': {
                'tracking': 0.3,
                'recovery': 0.2,
                'emergency': 0.2,
                'safety': 0.2,
                'efficiency': 0.1
            },
            'tracking': {
                'max_deviation': 10.0,
                'heading_weight': 0.3,
                'speed_weight': 0.2,
                'distance_weight': 0.5
            },
            'recovery': {
                'max_deviation': 100.0,
                'max_recovery_time': 60.0,
                'success_bonus': 1.0,
                'failure_penalty': -1.0
            },
            'emergency': {
                'critical_battery_threshold': 0.2,
                'energy_efficiency_weight': 0.3,
                'decision_quality_weight': 0.7
            },
            'safety': {
                'collision_penalty': -10.0,
                'altitude_violation_penalty': -5.0,
                'speed_violation_penalty': -3.0,
                'safe_flight_bonus': 0.1
            },
            'efficiency': {
                'energy_consumption_weight': 0.6,
                'time_efficiency_weight': 0.4
            }
        }
    
    def calculate_reward(self, 
                       current_state: Dict[str, Any],
                       action: Dict[str, Any],
                       next_state: Dict[str, Any],
                       flight_status: FlightStatus,
                       task_phase: TaskPhase = TaskPhase.NORMAL) -> RewardComponents:
        """
        计算综合奖励函数
        
        Args:
            current_state: 当前状态
            action: 执行的动作
            next_state: 下一状态
            flight_status: 飞行状态信息
            task_phase: 任务阶段
            
        Returns:
            RewardComponents: 奖励组件对象
        """
        # 更新历史记录
        self._update_history(next_state, flight_status)
        
        # 计算各个奖励组件
        r_tracking = self._calculate_tracking_reward(next_state, flight_status)
        r_recovery = self._calculate_recovery_reward(flight_status)
        r_emergency = self._calculate_emergency_reward(next_state, action, flight_status)
        r_safety = self._calculate_safety_reward(next_state, flight_status)
        r_efficiency = self._calculate_efficiency_reward(current_state, next_state, action)
        
        # 获取动态权重
        weights = self._get_dynamic_weights(task_phase, flight_status)
        
        # 计算总奖励
        total_reward = (
            weights['tracking'] * r_tracking +
            weights['recovery'] * r_recovery +
            weights['emergency'] * r_emergency +
            weights['safety'] * r_safety +
            weights['efficiency'] * r_efficiency
        )
        
        return RewardComponents(
            tracking=r_tracking,
            recovery=r_recovery,
            emergency=r_emergency,
            safety=r_safety,
            efficiency=r_efficiency,
            total=total_reward
        )
    
    def _calculate_tracking_reward(self, state: Dict[str, Any], flight_status: FlightStatus) -> float:
        """
        计算循迹能力奖励
        
        Args:
            state: 当前状态
            flight_status: 飞行状态信息
            
        Returns:
            float: 循迹奖励值
        """
        if not flight_status.planned_trajectory:
            return 0.0
        
        current_pos = Position3D(
            x=state['position'][0],
            y=state['position'][1], 
            z=state['position'][2]
        )
        
        # 1. 计算到最近轨迹点的距离
        min_distance = float('inf')
        closest_point_idx = 0
        
        for i, traj_point in enumerate(flight_status.planned_trajectory):
            distance = self._calculate_distance(current_pos, traj_point)
            if distance < min_distance:
                min_distance = distance
                closest_point_idx = i
        
        # 2. 计算轨迹偏离奖励
        max_deviation = self.config['tracking']['max_deviation']
        if min_distance <= flight_status.trajectory_tolerance:
            distance_reward = 1.0 - (min_distance / flight_status.trajectory_tolerance)
        else:
            distance_reward = -0.5 * min(min_distance / max_deviation, 2.0)
        
        # 3. 计算航向一致性奖励
        heading_consistency = self._calculate_heading_consistency(
            current_pos, flight_status.planned_trajectory, closest_point_idx, state
        )
        
        # 4. 计算速度一致性奖励
        speed_consistency = self._calculate_speed_consistency(state, flight_status)
        
        # 综合计算
        weights = self.config['tracking']
        tracking_reward = (
            weights['distance_weight'] * distance_reward +
            weights['heading_weight'] * heading_consistency +
            weights['speed_weight'] * speed_consistency
        )
        
        return np.clip(tracking_reward, -2.0, 1.0)
    
    def _calculate_recovery_reward(self, flight_status: FlightStatus) -> float:
        """
        计算寻回定位能力奖励
        
        Args:
            flight_status: 飞行状态信息
            
        Returns:
            float: 寻回奖励值
        """
        config = self.config['recovery']
        
        if not flight_status.recovery_success:
            return config['failure_penalty']
        
        if flight_status.deviation_distance == 0.0:
            return 0.0  # 没有偏离，不需要寻回
        
        # 1. 基于偏离距离的奖励
        max_deviation = config['max_deviation']
        deviation_penalty = min(flight_status.deviation_distance / max_deviation, 1.0)
        
        # 2. 基于寻回时间的奖励
        max_recovery_time = config['max_recovery_time']
        time_efficiency = max(0, 1.0 - flight_status.recovery_time / max_recovery_time)
        
        # 3. 寻回策略质量评估
        strategy_quality = self._evaluate_recovery_strategy_quality(flight_status)
        
        recovery_reward = (
            0.4 * (1.0 - deviation_penalty) +
            0.4 * time_efficiency +
            0.2 * strategy_quality
        )
        
        return np.clip(recovery_reward, -1.0, 1.0)
    
    def _calculate_emergency_reward(self, state: Dict[str, Any], action: Dict[str, Any], 
                                  flight_status: FlightStatus) -> float:
        """
        计算紧急决策能力奖励
        
        Args:
            state: 当前状态
            action: 执行的动作
            flight_status: 飞行状态信息
            
        Returns:
            float: 紧急决策奖励值
        """
        config = self.config['emergency']
        
        current_pos = Position3D(
            x=state['position'][0],
            y=state['position'][1],
            z=state['position'][2]
        )
        
        distance_to_target = self._calculate_distance(current_pos, flight_status.target_position)
        estimated_energy_needed = self._estimate_energy_consumption(current_pos, flight_status.target_position)
        
        # 1. 电池临界状态判断
        critical_threshold = config['critical_battery_threshold']
        is_critical = flight_status.battery_level < critical_threshold
        
        # 2. 决策合理性评估
        decision_type = action.get('decision_type', DecisionType.CONTINUE)
        
        if is_critical:
            if (decision_type == DecisionType.DIRECT_TO_TARGET and 
                flight_status.battery_level > estimated_energy_needed):
                decision_reward = 1.0  # 最优决策
            elif (decision_type == DecisionType.EMERGENCY_LAND and 
                  flight_status.battery_level < estimated_energy_needed):
                decision_reward = 0.8  # 合理决策
            elif decision_type == DecisionType.CONTINUE:
                decision_reward = -0.5  # 危险决策
            else:
                decision_reward = 0.0
        else:
            if decision_type == DecisionType.CONTINUE:
                decision_reward = 0.5  # 正常情况下继续原计划
            else:
                decision_reward = 0.0  # 非必要的紧急决策
        
        # 3. 能效比奖励
        energy_efficiency = self._calculate_energy_efficiency(decision_type, current_pos, flight_status)
        
        emergency_reward = (
            config['decision_quality_weight'] * decision_reward +
            config['energy_efficiency_weight'] * energy_efficiency
        )
        
        return np.clip(emergency_reward, -1.0, 1.0)
    
    def _calculate_safety_reward(self, state: Dict[str, Any], flight_status: FlightStatus) -> float:
        """
        计算安全性奖励
        
        Args:
            state: 当前状态
            flight_status: 飞行状态信息
            
        Returns:
            float: 安全性奖励值
        """
        config = self.config['safety']
        safety_reward = 0.0
        
        # 1. 碰撞检测
        if flight_status.is_collision:
            safety_reward += config['collision_penalty']
        
        # 2. 高度违规检测
        altitude = -state['position'][2]  # AirSim中z轴向下为正
        min_alt, max_alt = flight_status.altitude_limit
        if altitude < min_alt or altitude > max_alt:
            safety_reward += config['altitude_violation_penalty']
        
        # 3. 速度违规检测
        velocity = np.linalg.norm(state['velocity'])
        if velocity > flight_status.speed_limit:
            safety_reward += config['speed_violation_penalty']
        
        # 4. 安全飞行奖励
        if not flight_status.is_collision and min_alt <= altitude <= max_alt and velocity <= flight_status.speed_limit:
            safety_reward += config['safe_flight_bonus']
        
        return safety_reward
    
    def _calculate_efficiency_reward(self, current_state: Dict[str, Any], 
                                   next_state: Dict[str, Any], action: Dict[str, Any]) -> float:
        """
        计算能效奖励
        
        Args:
            current_state: 当前状态
            next_state: 下一状态
            action: 执行的动作
            
        Returns:
            float: 能效奖励值
        """
        config = self.config['efficiency']
        
        # 1. 能耗效率计算
        energy_consumption = self._calculate_step_energy_consumption(current_state, next_state, action)
        self.cumulative_energy_consumption += energy_consumption
        
        # 归一化能耗（假设最大单步能耗为0.01）
        normalized_energy = min(energy_consumption / 0.01, 1.0)
        energy_efficiency_reward = 1.0 - normalized_energy
        
        # 2. 时间效率计算
        distance_moved = self._calculate_distance(
            Position3D(x=current_state['position'][0], y=current_state['position'][1], z=current_state['position'][2]),
            Position3D(x=next_state['position'][0], y=next_state['position'][1], z=next_state['position'][2])
        )
        
        # 鼓励有效移动，惩罚原地不动
        time_efficiency_reward = min(distance_moved / 5.0, 1.0)  # 假设期望每步移动5米
        
        efficiency_reward = (
            config['energy_consumption_weight'] * energy_efficiency_reward +
            config['time_efficiency_weight'] * time_efficiency_reward
        )
        
        return np.clip(efficiency_reward, 0.0, 1.0)
    
    def _get_dynamic_weights(self, task_phase: TaskPhase, flight_status: FlightStatus) -> Dict[str, float]:
        """
        根据任务阶段和飞行状态动态调整奖励权重
        
        Args:
            task_phase: 任务阶段
            flight_status: 飞行状态信息
            
        Returns:
            Dict[str, float]: 动态权重字典
        """
        base_weights = self.config['weights'].copy()
        
        # 根据任务阶段调整权重
        if task_phase == TaskPhase.NORMAL:
            base_weights['tracking'] = 0.4
            base_weights['efficiency'] = 0.2
        elif task_phase == TaskPhase.RECOVERY:
            base_weights['recovery'] = 0.4
            base_weights['tracking'] = 0.2
        elif task_phase == TaskPhase.EMERGENCY:
            base_weights['safety'] = 0.4
            base_weights['emergency'] = 0.3
            base_weights['tracking'] = 0.1
        
        # 根据电池电量调整权重
        if flight_status.battery_level < 0.3:
            base_weights['emergency'] += 0.1
            base_weights['efficiency'] += 0.1
            base_weights['tracking'] -= 0.1
            base_weights['recovery'] -= 0.1
        
        # 确保权重和为1
        total_weight = sum(base_weights.values())
        for key in base_weights:
            base_weights[key] /= total_weight
        
        return base_weights
    
    # 辅助方法
    def _update_history(self, state: Dict[str, Any], flight_status: FlightStatus):
        """更新历史记录"""
        current_pos = Position3D(
            x=state['position'][0],
            y=state['position'][1],
            z=state['position'][2]
        )
        self.position_history.append(current_pos)
        
        # 保持历史记录长度
        if len(self.position_history) > 100:
            self.position_history.pop(0)
        
        # 更新电池记录
        self.last_battery_level = flight_status.battery_level
    
    def _calculate_distance(self, pos1: Position3D, pos2: Position3D) -> float:
        """计算两点间距离"""
        return math.sqrt(
            (pos1.x - pos2.x) ** 2 +
            (pos1.y - pos2.y) ** 2 +
            (pos1.z - pos2.z) ** 2
        )
    
    def _calculate_heading_consistency(self, current_pos: Position3D, trajectory: List[Position3D], 
                                     closest_idx: int, state: Dict[str, Any]) -> float:
        """计算航向一致性"""
        if closest_idx >= len(trajectory) - 1:
            return 0.0
        
        # 计算期望航向
        next_point = trajectory[closest_idx + 1]
        expected_heading = math.atan2(
            next_point.y - current_pos.y,
            next_point.x - current_pos.x
        )
        
        # 获取当前航向
        current_heading = state['attitude'][2]  # yaw角
        
        # 计算航向差异
        heading_diff = abs(expected_heading - current_heading)
        heading_diff = min(heading_diff, 2 * math.pi - heading_diff)  # 取较小角度
        
        # 归一化到[0, 1]
        return 1.0 - (heading_diff / math.pi)
    
    def _calculate_speed_consistency(self, state: Dict[str, Any], flight_status: FlightStatus) -> float:
        """计算速度一致性"""
        current_speed = np.linalg.norm(state['velocity'])
        expected_speed = 5.0  # 默认期望速度
        
        # 从轨迹中获取期望速度（如果可用）
        if hasattr(flight_status, 'expected_speed'):
            expected_speed = flight_status.expected_speed
        
        speed_diff = abs(current_speed - expected_speed)
        return max(0.0, 1.0 - speed_diff / expected_speed)
    
    def _evaluate_recovery_strategy_quality(self, flight_status: FlightStatus) -> float:
        """评估寻回策略质量"""
        # 简化实现：基于寻回时间和偏离距离的比值
        if flight_status.recovery_time == 0 or flight_status.deviation_distance == 0:
            return 1.0
        
        efficiency_ratio = flight_status.deviation_distance / flight_status.recovery_time
        # 归一化到[0, 1]，假设最优效率比为10米/秒
        return min(efficiency_ratio / 10.0, 1.0)
    
    def _estimate_energy_consumption(self, current_pos: Position3D, target_pos: Position3D) -> float:
        """估算能耗"""
        distance = self._calculate_distance(current_pos, target_pos)
        # 简化模型：假设每米消耗0.001单位能量
        return distance * 0.001
    
    def _calculate_energy_efficiency(self, decision_type: DecisionType, 
                                   current_pos: Position3D, flight_status: FlightStatus) -> float:
        """计算能效比"""
        if decision_type == DecisionType.DIRECT_TO_TARGET:
            # 直飞最节能
            return 1.0
        elif decision_type == DecisionType.CONTINUE:
            # 继续原路线，中等能效
            return 0.6
        elif decision_type == DecisionType.EMERGENCY_LAND:
            # 紧急降落，能效取决于当前高度
            altitude = abs(current_pos.z)
            return max(0.2, 1.0 - altitude / 50.0)  # 高度越高能效越低
        else:
            return 0.5
    
    def _calculate_step_energy_consumption(self, current_state: Dict[str, Any], 
                                         next_state: Dict[str, Any], action: Dict[str, Any]) -> float:
        """计算单步能耗"""
        # 基于速度和加速度的简化能耗模型
        velocity = np.linalg.norm(next_state['velocity'])
        
        # 计算加速度
        current_vel = np.array(current_state['velocity'])
        next_vel = np.array(next_state['velocity'])
        acceleration = np.linalg.norm(next_vel - current_vel)
        
        # 简化能耗模型：基础能耗 + 速度相关 + 加速度相关
        base_consumption = 0.001
        velocity_consumption = velocity * 0.0001
        acceleration_consumption = acceleration * 0.0005
        
        return base_consumption + velocity_consumption + acceleration_consumption
    
    def reset(self):
        """重置奖励函数状态"""
        self.position_history.clear()
        self.deviation_start_time = None
        self.last_battery_level = 1.0
        self.cumulative_energy_consumption = 0.0