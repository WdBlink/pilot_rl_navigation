#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多元化奖励函数模块

该模块实现了强化学习无人机导航系统的多元化奖励函数，包括：
1. 循迹能力奖励 - 评估无人机按照预定航线飞行的能力
2. 寻回定位能力奖励 - 评估无人机偏离航线后返回的能力
3. 紧急决策能力奖励 - 评估无人机在紧急情况下的决策质量
4. 安全性奖励 - 评估飞行安全性
5. 能效奖励 - 评估能源使用效率
"""

import numpy as np
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

# 任务阶段枚举
class TaskPhase(Enum):
    NORMAL = "normal"           # 正常飞行阶段
    RECOVERY = "recovery"       # 寻回阶段
    EMERGENCY = "emergency"     # 紧急阶段
    LANDING = "landing"         # 降落阶段

# 决策类型枚举
class DecisionType(Enum):
    FOLLOW_TRAJECTORY = "follow_trajectory"     # 跟随轨迹
    RETURN_TO_TRAJECTORY = "return_to_trajectory" # 返回轨迹
    DIRECT_TO_TARGET = "direct_to_target"       # 直接飞向目标
    EMERGENCY_LANDING = "emergency_landing"     # 紧急降落
    HOVER = "hover"                             # 悬停

# 飞行状态枚举
class FlightStatus(Enum):
    NORMAL = "normal"               # 正常状态
    DEVIATION = "deviation"         # 偏离状态
    LOW_BATTERY = "low_battery"     # 低电量状态
    EMERGENCY = "emergency"         # 紧急状态
    CRITICAL = "critical"           # 危急状态

@dataclass
class FlightState:
    """飞行状态数据类"""
    position: np.ndarray            # 位置 [x, y, z]
    velocity: np.ndarray            # 速度 [vx, vy, vz]
    orientation: np.ndarray         # 姿态 [roll, pitch, yaw]
    battery_level: float            # 电池电量 (0-1)
    timestamp: float                # 时间戳
    
    # 可选的额外状态信息
    acceleration: Optional[np.ndarray] = None
    angular_velocity: Optional[np.ndarray] = None
    sensor_data: Optional[Dict] = None

class BaseReward(ABC):
    """奖励函数基类"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.flight_plan: Optional[List[np.ndarray]] = None
        self.reward_history: List[float] = []
    
    @abstractmethod
    def calculate(self, state: FlightState, action: np.ndarray, 
                 next_state: FlightState, info: Dict) -> float:
        """计算奖励值"""
        pass
    
    def set_flight_plan(self, flight_plan: List[np.ndarray]):
        """设置飞行计划"""
        self.flight_plan = flight_plan
    
    def reset(self):
        """重置奖励函数状态"""
        self.reward_history.clear()
    
    def get_closest_waypoint(self, position: np.ndarray) -> Tuple[int, np.ndarray, float]:
        """获取最近的航点"""
        if not self.flight_plan:
            return 0, np.array([0, 0, 0]), float('inf')
        
        min_distance = float('inf')
        closest_idx = 0
        closest_waypoint = self.flight_plan[0]
        
        for i, waypoint in enumerate(self.flight_plan):
            distance = np.linalg.norm(position - waypoint)
            if distance < min_distance:
                min_distance = distance
                closest_idx = i
                closest_waypoint = waypoint
        
        return closest_idx, closest_waypoint, min_distance

class TrackingReward(BaseReward):
    """循迹能力奖励"""
    
    def calculate(self, state: FlightState, action: np.ndarray, 
                 next_state: FlightState, info: Dict) -> float:
        """计算循迹奖励"""
        if not self.flight_plan:
            return 0.0
        
        # 获取最近航点和偏离距离
        closest_idx, closest_waypoint, deviation = self.get_closest_waypoint(state.position)
        
        # 距离奖励 - 基于偏离距离
        max_deviation = self.config.get('max_deviation', 15.0)
        distance_reward = max(0, 1 - deviation / max_deviation)
        
        # 航向一致性奖励
        heading_reward = self._calculate_heading_consistency(state, closest_idx)
        
        # 速度一致性奖励
        speed_reward = self._calculate_speed_consistency(state)
        
        # 加权组合
        distance_weight = self.config.get('distance_weight', 0.5)
        heading_weight = self.config.get('heading_weight', 0.3)
        speed_weight = self.config.get('speed_weight', 0.2)
        
        total_reward = (distance_weight * distance_reward + 
                       heading_weight * heading_reward + 
                       speed_weight * speed_reward)
        
        self.reward_history.append(total_reward)
        return total_reward
    
    def _calculate_heading_consistency(self, state: FlightState, waypoint_idx: int) -> float:
        """计算航向一致性奖励"""
        if waypoint_idx >= len(self.flight_plan) - 1:
            return 1.0  # 已到达最后一个航点
        
        # 计算期望航向（从当前航点到下一个航点）
        current_waypoint = self.flight_plan[waypoint_idx]
        next_waypoint = self.flight_plan[waypoint_idx + 1]
        expected_direction = next_waypoint - current_waypoint
        expected_direction = expected_direction / (np.linalg.norm(expected_direction) + 1e-8)
        
        # 计算实际航向
        actual_direction = state.velocity / (np.linalg.norm(state.velocity) + 1e-8)
        
        # 计算航向一致性（余弦相似度）
        consistency = np.dot(expected_direction, actual_direction)
        return max(0, consistency)
    
    def _calculate_speed_consistency(self, state: FlightState) -> float:
        """计算速度一致性奖励"""
        expected_speed = self.config.get('expected_speed', 8.0)
        actual_speed = np.linalg.norm(state.velocity)
        
        speed_error = abs(actual_speed - expected_speed) / expected_speed
        return max(0, 1 - speed_error)

class RecoveryReward(BaseReward):
    """寻回定位能力奖励"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.recovery_start_time: Optional[float] = None
        self.recovery_start_position: Optional[np.ndarray] = None
        self.initial_deviation: Optional[float] = None
    
    def start_recovery(self, state: FlightState):
        """开始寻回过程"""
        self.recovery_start_time = state.timestamp
        self.recovery_start_position = state.position.copy()
        _, _, self.initial_deviation = self.get_closest_waypoint(state.position)
    
    def calculate(self, state: FlightState, action: np.ndarray, 
                 next_state: FlightState, info: Dict) -> float:
        """计算寻回奖励"""
        if not self.flight_plan:
            return 0.0
        
        task_phase = info.get('task_phase', TaskPhase.NORMAL)
        
        # 只在寻回阶段计算寻回奖励
        if task_phase != TaskPhase.RECOVERY:
            return 0.0
        
        # 如果还没开始寻回，初始化寻回状态
        if self.recovery_start_time is None:
            self.start_recovery(state)
        
        # 计算寻回进度奖励
        progress_reward = self._calculate_recovery_progress(next_state)
        
        # 计算时间效率奖励
        time_reward = self._calculate_time_efficiency(next_state)
        
        # 检查是否成功寻回
        success_reward = self._calculate_success_reward(next_state)
        
        total_reward = progress_reward + time_reward + success_reward
        
        self.reward_history.append(total_reward)
        return total_reward
    
    def _calculate_recovery_progress(self, state: FlightState) -> float:
        """计算寻回进度奖励"""
        if self.initial_deviation is None:
            return 0.0
        
        _, _, current_deviation = self.get_closest_waypoint(state.position)
        
        # 计算寻回进度（偏离距离的减少程度）
        progress = (self.initial_deviation - current_deviation) / self.initial_deviation
        return max(0, progress)
    
    def _calculate_time_efficiency(self, state: FlightState) -> float:
        """计算时间效率奖励"""
        if self.recovery_start_time is None:
            return 0.0
        
        recovery_time = state.timestamp - self.recovery_start_time
        max_recovery_time = self.config.get('max_recovery_time', 120.0)
        
        # 时间效率奖励（越快寻回奖励越高）
        time_efficiency = max(0, 1 - recovery_time / max_recovery_time)
        time_penalty_factor = self.config.get('time_penalty_factor', 0.01)
        
        return time_efficiency - time_penalty_factor * recovery_time
    
    def _calculate_success_reward(self, state: FlightState) -> float:
        """计算成功寻回奖励"""
        _, _, deviation = self.get_closest_waypoint(state.position)
        max_deviation = self.config.get('max_deviation', 50.0)
        
        # 如果成功返回航线（偏离距离小于阈值）
        if deviation < max_deviation * 0.3:  # 30%的最大偏离距离作为成功阈值
            success_bonus = self.config.get('success_bonus', 2.0)
            self.reset_recovery()  # 重置寻回状态
            return success_bonus
        
        return 0.0
    
    def reset_recovery(self):
        """重置寻回状态"""
        self.recovery_start_time = None
        self.recovery_start_position = None
        self.initial_deviation = None
    
    def reset(self):
        """重置奖励函数状态"""
        super().reset()
        self.reset_recovery()

class EmergencyReward(BaseReward):
    """紧急决策能力奖励"""
    
    def calculate(self, state: FlightState, action: np.ndarray, 
                 next_state: FlightState, info: Dict) -> float:
        """计算紧急决策奖励"""
        task_phase = info.get('task_phase', TaskPhase.NORMAL)
        
        # 只在紧急阶段计算紧急决策奖励
        if task_phase != TaskPhase.EMERGENCY:
            return 0.0
        
        # 评估决策质量
        decision_quality = self._evaluate_decision_quality(state, action, info)
        
        # 评估能效优化
        energy_efficiency = self._evaluate_energy_efficiency(state, action, next_state)
        
        # 加权组合
        decision_weight = self.config.get('decision_quality_weight', 0.7)
        energy_weight = self.config.get('energy_efficiency_weight', 0.3)
        
        total_reward = (decision_weight * decision_quality + 
                       energy_weight * energy_efficiency)
        
        self.reward_history.append(total_reward)
        return total_reward
    
    def _evaluate_decision_quality(self, state: FlightState, action: np.ndarray, info: Dict) -> float:
        """评估决策质量"""
        decision_type = info.get('decision_type', DecisionType.FOLLOW_TRAJECTORY)
        battery_level = state.battery_level
        critical_threshold = self.config.get('critical_battery_threshold', 0.2)
        
        # 根据电池电量和决策类型评估决策质量
        if battery_level < critical_threshold:
            # 低电量情况下，直接飞向目标是最优决策
            if decision_type == DecisionType.DIRECT_TO_TARGET:
                return self.config.get('optimal_decision_bonus', 1.5)
            elif decision_type == DecisionType.EMERGENCY_LANDING:
                return self.config.get('optimal_decision_bonus', 1.5) * 0.8
            else:
                return self.config.get('poor_decision_penalty', -1.0)
        else:
            # 电量充足时，继续跟随轨迹或寻回是合理的
            if decision_type in [DecisionType.FOLLOW_TRAJECTORY, DecisionType.RETURN_TO_TRAJECTORY]:
                return 0.5
            else:
                return 0.0
    
    def _evaluate_energy_efficiency(self, state: FlightState, action: np.ndarray, next_state: FlightState) -> float:
        """评估能效优化"""
        # 计算速度变化（加速度消耗更多能量）
        velocity_change = np.linalg.norm(next_state.velocity - state.velocity)
        
        # 计算高度变化（爬升消耗更多能量）
        altitude_change = abs(next_state.position[2] - state.position[2])
        
        # 能效评分（速度和高度变化越小越好）
        energy_score = max(0, 1 - 0.1 * velocity_change - 0.05 * altitude_change)
        
        return energy_score

class SafetyReward(BaseReward):
    """安全性奖励"""
    
    def calculate(self, state: FlightState, action: np.ndarray, 
                 next_state: FlightState, info: Dict) -> float:
        """计算安全性奖励"""
        safety_reward = 0.0
        
        # 碰撞检测
        if info.get('collision_detected', False):
            safety_reward += self.config.get('collision_penalty', -20.0)
        
        # 高度违规检测
        if info.get('altitude_violation', False):
            safety_reward += self.config.get('altitude_violation_penalty', -8.0)
        
        # 速度违规检测
        if info.get('speed_violation', False):
            safety_reward += self.config.get('speed_violation_penalty', -5.0)
        
        # 接近碰撞检测
        if info.get('near_collision', False):
            safety_reward += self.config.get('near_collision_penalty', -3.0)
        
        # 安全飞行奖励
        if not any([info.get('collision_detected', False),
                   info.get('altitude_violation', False),
                   info.get('speed_violation', False),
                   info.get('near_collision', False)]):
            safety_reward += self.config.get('safe_flight_bonus', 0.2)
        
        self.reward_history.append(safety_reward)
        return safety_reward

class EfficiencyReward(BaseReward):
    """能效奖励"""
    
    def calculate(self, state: FlightState, action: np.ndarray, 
                 next_state: FlightState, info: Dict) -> float:
        """计算能效奖励"""
        # 能耗评估
        energy_consumption = info.get('energy_consumption', 0.001)
        energy_score = self._evaluate_energy_consumption(energy_consumption)
        
        # 时间效率评估
        time_efficiency = info.get('time_efficiency', 0.8)
        time_score = self._evaluate_time_efficiency(time_efficiency)
        
        # 加权组合
        energy_weight = self.config.get('energy_consumption_weight', 0.6)
        time_weight = self.config.get('time_efficiency_weight', 0.4)
        
        total_reward = energy_weight * energy_score + time_weight * time_score
        
        self.reward_history.append(total_reward)
        return total_reward
    
    def _evaluate_energy_consumption(self, energy_consumption: float) -> float:
        """评估能耗效率"""
        # 假设正常能耗为0.001，低于此值给予奖励，高于此值给予惩罚
        normal_consumption = 0.001
        
        if energy_consumption < normal_consumption:
            return self.config.get('optimal_energy_bonus', 0.5)
        elif energy_consumption > normal_consumption * 2:
            return self.config.get('wasteful_penalty', -0.3)
        else:
            return 0.0
    
    def _evaluate_time_efficiency(self, time_efficiency: float) -> float:
        """评估时间效率"""
        # time_efficiency 应该是0-1之间的值，1表示最高效率
        return time_efficiency - 0.5  # 中性点为0.5

class MultiDimensionalRewardFunction:
    """多元化奖励函数主类"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.reward_config = config.get('reward_function', {})
        
        # 初始化各个奖励组件
        self.tracking_reward = TrackingReward(self.reward_config.get('tracking', {}))
        self.recovery_reward = RecoveryReward(self.reward_config.get('recovery', {}))
        self.emergency_reward = EmergencyReward(self.reward_config.get('emergency', {}))
        self.safety_reward = SafetyReward(self.reward_config.get('safety', {}))
        self.efficiency_reward = EfficiencyReward(self.reward_config.get('efficiency', {}))
        
        # 奖励权重
        self.base_weights = self.reward_config.get('weights', {
            'tracking': 0.3,
            'recovery': 0.2,
            'emergency': 0.2,
            'safety': 0.2,
            'efficiency': 0.1
        })
        
        # 奖励历史和统计
        self.reward_history: List[Dict[str, float]] = []
        self.flight_plan: Optional[List[np.ndarray]] = None
    
    def set_flight_plan(self, flight_plan: List[np.ndarray]):
        """设置飞行计划"""
        self.flight_plan = flight_plan
        self.tracking_reward.set_flight_plan(flight_plan)
        self.recovery_reward.set_flight_plan(flight_plan)
        self.emergency_reward.set_flight_plan(flight_plan)
        self.safety_reward.set_flight_plan(flight_plan)
        self.efficiency_reward.set_flight_plan(flight_plan)
    
    def calculate_reward(self, state: FlightState, action: np.ndarray, 
                       next_state: FlightState, info: Dict) -> float:
        """计算综合奖励"""
        # 计算各个奖励组件
        tracking_reward = self.tracking_reward.calculate(state, action, next_state, info)
        recovery_reward = self.recovery_reward.calculate(state, action, next_state, info)
        emergency_reward = self.emergency_reward.calculate(state, action, next_state, info)
        safety_reward = self.safety_reward.calculate(state, action, next_state, info)
        efficiency_reward = self.efficiency_reward.calculate(state, action, next_state, info)
        
        # 获取动态权重
        task_phase = info.get('task_phase', TaskPhase.NORMAL)
        flight_status = info.get('flight_status', FlightStatus.NORMAL)
        battery_level = state.battery_level
        
        weights = self.get_dynamic_weights(task_phase, flight_status, battery_level)
        
        # 加权求和
        total_reward = (
            weights['tracking'] * tracking_reward +
            weights['recovery'] * recovery_reward +
            weights['emergency'] * emergency_reward +
            weights['safety'] * safety_reward +
            weights['efficiency'] * efficiency_reward
        )
        
        # 记录奖励分解
        reward_breakdown = {
            'total': total_reward,
            'tracking': tracking_reward,
            'recovery': recovery_reward,
            'emergency': emergency_reward,
            'safety': safety_reward,
            'efficiency': efficiency_reward,
            'weights': weights.copy()
        }
        
        self.reward_history.append(reward_breakdown)
        
        return total_reward
    
    def get_dynamic_weights(self, task_phase: TaskPhase, flight_status: FlightStatus, 
                          battery_level: float) -> Dict[str, float]:
        """获取动态调整的权重"""
        weights = self.base_weights.copy()
        
        # 根据任务阶段调整权重
        if task_phase == TaskPhase.RECOVERY:
            weights['recovery'] += 0.2
            weights['tracking'] -= 0.1
            weights['efficiency'] -= 0.1
        elif task_phase == TaskPhase.EMERGENCY:
            weights['emergency'] += 0.2
            weights['safety'] += 0.1
            weights['tracking'] -= 0.15
            weights['recovery'] -= 0.15
        
        # 根据电池电量调整权重
        if battery_level < 0.3:
            weights['emergency'] += 0.1
            weights['efficiency'] += 0.05
            weights['tracking'] -= 0.075
            weights['recovery'] -= 0.075
        
        # 根据飞行状态调整权重
        if flight_status == FlightStatus.CRITICAL:
            weights['safety'] += 0.2
            weights['emergency'] += 0.1
            weights['tracking'] -= 0.15
            weights['recovery'] -= 0.15
        
        # 确保权重和为1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def get_last_reward_breakdown(self) -> Optional[Dict[str, float]]:
        """获取最后一次奖励的详细分解"""
        if self.reward_history:
            return self.reward_history[-1]
        return None
    
    def get_reward_statistics(self, window_size: int = 100) -> Dict[str, float]:
        """获取奖励统计信息"""
        if not self.reward_history:
            return {}
        
        recent_rewards = self.reward_history[-window_size:]
        
        stats = {}
        for component in ['total', 'tracking', 'recovery', 'emergency', 'safety', 'efficiency']:
            values = [r[component] for r in recent_rewards]
            stats[f'{component}_mean'] = np.mean(values)
            stats[f'{component}_std'] = np.std(values)
            stats[f'{component}_min'] = np.min(values)
            stats[f'{component}_max'] = np.max(values)
        
        return stats
    
    def reset(self):
        """重置奖励函数状态"""
        self.tracking_reward.reset()
        self.recovery_reward.reset()
        self.emergency_reward.reset()
        self.safety_reward.reset()
        self.efficiency_reward.reset()
        
        self.reward_history.clear()
        self.flight_plan = None
    
    def save_reward_history(self, filepath: str):
        """保存奖励历史到文件"""
        import json
        
        # 转换numpy数组为列表以便JSON序列化
        serializable_history = []
        for record in self.reward_history:
            serializable_record = {}
            for key, value in record.items():
                if isinstance(value, np.ndarray):
                    serializable_record[key] = value.tolist()
                else:
                    serializable_record[key] = value
            serializable_history.append(serializable_record)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_history, f, indent=2)
    
    def load_reward_history(self, filepath: str):
        """从文件加载奖励历史"""
        import json
        
        with open(filepath, 'r') as f:
            self.reward_history = json.load(f)