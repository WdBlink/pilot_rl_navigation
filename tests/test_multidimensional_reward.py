#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多元化奖励函数系统测试脚本

该脚本用于测试多元化奖励函数的各个组件是否正常工作，
包括循迹、寻回、紧急决策、安全性和能效等奖励计算。
"""

import sys
import os
import numpy as np
import unittest
from unittest.mock import Mock, patch
from dataclasses import dataclass
from typing import Dict, List, Tuple

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.environment.reward_function import (
    MultiDimensionalRewardFunction,
    FlightState,
    TaskPhase,
    DecisionType,
    FlightStatus,
    TrackingReward,
    RecoveryReward,
    EmergencyReward,
    SafetyReward,
    EfficiencyReward
)

class TestMultidimensionalReward(unittest.TestCase):
    """多元化奖励函数测试类"""
    
    def setUp(self):
        """测试初始化"""
        # 创建测试配置
        self.config = {
            'reward_function': {
                'weights': {
                    'tracking': 0.3,
                    'recovery': 0.2,
                    'emergency': 0.2,
                    'safety': 0.2,
                    'efficiency': 0.1
                },
                'tracking': {
                    'max_deviation': 15.0,
                    'heading_weight': 0.3,
                    'speed_weight': 0.2,
                    'distance_weight': 0.5,
                    'expected_speed': 8.0
                },
                'recovery': {
                    'max_deviation': 50.0,
                    'max_recovery_time': 120.0,
                    'success_bonus': 2.0,
                    'failure_penalty': -2.0,
                    'time_penalty_factor': 0.01
                },
                'emergency': {
                    'critical_battery_threshold': 0.2,
                    'energy_efficiency_weight': 0.3,
                    'decision_quality_weight': 0.7,
                    'optimal_decision_bonus': 1.5,
                    'poor_decision_penalty': -1.0
                },
                'safety': {
                    'collision_penalty': -20.0,
                    'altitude_violation_penalty': -8.0,
                    'speed_violation_penalty': -5.0,
                    'safe_flight_bonus': 0.2,
                    'near_collision_penalty': -3.0
                },
                'efficiency': {
                    'energy_consumption_weight': 0.6,
                    'time_efficiency_weight': 0.4,
                    'optimal_energy_bonus': 0.5,
                    'wasteful_penalty': -0.3
                }
            }
        }
        
        # 创建奖励函数实例
        self.reward_function = MultiDimensionalRewardFunction(self.config)
        
        # 创建测试用的飞行计划
        self.flight_plan = [
            np.array([0, 0, -10]),
            np.array([25, 0, -10]),
            np.array([50, 0, -10]),
            np.array([75, 0, -10]),
            np.array([100, 0, -10])
        ]
        
        # 创建测试状态
        self.create_test_states()
    
    def create_test_states(self):
        """创建测试用的飞行状态"""
        # 正常循迹状态
        self.normal_state = FlightState(
            position=np.array([12.5, 2.0, -10.0]),
            velocity=np.array([8.0, 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 0.0]),
            battery_level=0.8,
            timestamp=0.0
        )
        
        # 偏离航线状态
        self.deviated_state = FlightState(
            position=np.array([25.0, 20.0, -10.0]),
            velocity=np.array([5.0, -3.0, 0.0]),
            orientation=np.array([0.0, 0.0, -0.5]),
            battery_level=0.7,
            timestamp=10.0
        )
        
        # 低电量紧急状态
        self.emergency_state = FlightState(
            position=np.array([60.0, 5.0, -10.0]),
            velocity=np.array([10.0, 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 0.0]),
            battery_level=0.15,
            timestamp=50.0
        )
        
        # 危险状态（高度过低）
        self.unsafe_state = FlightState(
            position=np.array([30.0, 0.0, -2.0]),
            velocity=np.array([8.0, 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 0.0]),
            battery_level=0.6,
            timestamp=20.0
        )
    
    def test_tracking_reward_normal(self):
        """测试正常循迹奖励计算"""
        tracking_reward = TrackingReward(self.config['reward_function']['tracking'])
        
        # 设置飞行计划
        tracking_reward.set_flight_plan(self.flight_plan)
        
        # 计算奖励
        reward = tracking_reward.calculate(
            state=self.normal_state,
            action=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            next_state=self.normal_state,
            info={'task_phase': TaskPhase.NORMAL}
        )
        
        # 验证奖励为正值（因为偏离距离较小）
        self.assertGreater(reward, 0, "正常循迹应该获得正奖励")
        print(f"正常循迹奖励: {reward:.4f}")
    
    def test_tracking_reward_deviated(self):
        """测试偏离航线时的循迹奖励"""
        tracking_reward = TrackingReward(self.config['reward_function']['tracking'])
        tracking_reward.set_flight_plan(self.flight_plan)
        
        reward = tracking_reward.calculate(
            state=self.deviated_state,
            action=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            next_state=self.deviated_state,
            info={'task_phase': TaskPhase.NORMAL}
        )
        
        # 验证奖励为负值或很小的正值（因为偏离距离较大）
        self.assertLess(reward, 0.5, "偏离航线应该获得较低奖励")
        print(f"偏离航线奖励: {reward:.4f}")
    
    def test_recovery_reward(self):
        """测试寻回奖励计算"""
        recovery_reward = RecoveryReward(self.config['reward_function']['recovery'])
        recovery_reward.set_flight_plan(self.flight_plan)
        
        # 模拟寻回过程
        recovery_reward.start_recovery(self.deviated_state)
        
        # 创建寻回中的状态（向航线靠近）
        recovering_state = FlightState(
            position=np.array([25.0, 15.0, -10.0]),  # 比之前更接近航线
            velocity=np.array([0.0, -5.0, 0.0]),     # 向航线移动
            orientation=np.array([0.0, 0.0, -1.57]), # 转向航线
            battery_level=0.65,
            timestamp=15.0
        )
        
        reward = recovery_reward.calculate(
            state=self.deviated_state,
            action=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            next_state=recovering_state,
            info={'task_phase': TaskPhase.RECOVERY}
        )
        
        # 验证寻回进度获得正奖励
        self.assertGreater(reward, 0, "寻回进度应该获得正奖励")
        print(f"寻回奖励: {reward:.4f}")
    
    def test_emergency_reward(self):
        """测试紧急决策奖励"""
        emergency_reward = EmergencyReward(self.config['reward_function']['emergency'])
        emergency_reward.set_flight_plan(self.flight_plan)
        
        # 模拟紧急决策动作（直接飞向目标）
        emergency_action = np.array([1.0, 0.0, 1.0, 0.0, 0.0])  # 高效模式
        
        reward = emergency_reward.calculate(
            state=self.emergency_state,
            action=emergency_action,
            next_state=self.emergency_state,
            info={
                'task_phase': TaskPhase.EMERGENCY,
                'decision_type': DecisionType.DIRECT_TO_TARGET
            }
        )
        
        # 在紧急情况下做出正确决策应该获得奖励
        self.assertGreater(reward, 0, "紧急情况下的正确决策应该获得奖励")
        print(f"紧急决策奖励: {reward:.4f}")
    
    def test_safety_reward(self):
        """测试安全性奖励"""
        safety_reward = SafetyReward(self.config['reward_function']['safety'])
        
        # 测试安全飞行
        safe_reward = safety_reward.calculate(
            state=self.normal_state,
            action=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            next_state=self.normal_state,
            info={'collision_detected': False, 'altitude_violation': False}
        )
        
        # 测试不安全飞行
        unsafe_reward = safety_reward.calculate(
            state=self.unsafe_state,
            action=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            next_state=self.unsafe_state,
            info={'collision_detected': False, 'altitude_violation': True}
        )
        
        # 验证安全飞行获得正奖励，不安全飞行获得负奖励
        self.assertGreater(safe_reward, unsafe_reward, "安全飞行应该比不安全飞行获得更高奖励")
        self.assertLess(unsafe_reward, 0, "高度违规应该获得负奖励")
        print(f"安全飞行奖励: {safe_reward:.4f}, 不安全飞行奖励: {unsafe_reward:.4f}")
    
    def test_efficiency_reward(self):
        """测试能效奖励"""
        efficiency_reward = EfficiencyReward(self.config['reward_function']['efficiency'])
        
        # 测试高效飞行（合理速度，低能耗）
        efficient_reward = efficiency_reward.calculate(
            state=self.normal_state,
            action=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            next_state=self.normal_state,
            info={'energy_consumption': 0.001, 'time_efficiency': 0.9}
        )
        
        # 测试低效飞行（过高速度，高能耗）
        high_speed_state = FlightState(
            position=np.array([12.5, 2.0, -10.0]),
            velocity=np.array([20.0, 0.0, 0.0]),  # 过高速度
            orientation=np.array([0.0, 0.0, 0.0]),
            battery_level=0.7,  # 电量消耗更快
            timestamp=0.0
        )
        
        inefficient_reward = efficiency_reward.calculate(
            state=high_speed_state,
            action=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            next_state=high_speed_state,
            info={'energy_consumption': 0.005, 'time_efficiency': 0.6}
        )
        
        # 验证高效飞行获得更高奖励
        self.assertGreater(efficient_reward, inefficient_reward, "高效飞行应该获得更高奖励")
        print(f"高效飞行奖励: {efficient_reward:.4f}, 低效飞行奖励: {inefficient_reward:.4f}")
    
    def test_multidimensional_reward_integration(self):
        """测试多元化奖励函数整体集成"""
        # 设置飞行计划
        self.reward_function.set_flight_plan(self.flight_plan)
        
        # 测试正常飞行状态
        normal_reward = self.reward_function.calculate_reward(
            state=self.normal_state,
            action=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            next_state=self.normal_state,
            info={
                'task_phase': TaskPhase.NORMAL,
                'flight_status': FlightStatus.NORMAL,
                'collision_detected': False,
                'altitude_violation': False,
                'energy_consumption': 0.001,
                'time_efficiency': 0.8
            }
        )
        
        # 测试紧急状态
        emergency_reward = self.reward_function.calculate_reward(
            state=self.emergency_state,
            action=np.array([1.0, 0.0, 1.0, 0.0, 0.0]),
            next_state=self.emergency_state,
            info={
                'task_phase': TaskPhase.EMERGENCY,
                'flight_status': FlightStatus.LOW_BATTERY,
                'decision_type': DecisionType.DIRECT_TO_TARGET,
                'collision_detected': False,
                'altitude_violation': False,
                'energy_consumption': 0.002,
                'time_efficiency': 0.9
            }
        )
        
        # 验证奖励计算正常
        self.assertIsInstance(normal_reward, float, "正常状态奖励应该是浮点数")
        self.assertIsInstance(emergency_reward, float, "紧急状态奖励应该是浮点数")
        
        print(f"正常状态综合奖励: {normal_reward:.4f}")
        print(f"紧急状态综合奖励: {emergency_reward:.4f}")
        
        # 获取奖励组件分解
        reward_breakdown = self.reward_function.get_last_reward_breakdown()
        print(f"奖励组件分解: {reward_breakdown}")
        
        # 验证奖励组件
        self.assertIn('tracking', reward_breakdown, "应该包含循迹奖励组件")
        self.assertIn('safety', reward_breakdown, "应该包含安全奖励组件")
        self.assertIn('efficiency', reward_breakdown, "应该包含能效奖励组件")
    
    def test_dynamic_weight_adjustment(self):
        """测试动态权重调整"""
        # 测试正常阶段权重
        normal_weights = self.reward_function.get_dynamic_weights(
            TaskPhase.NORMAL, 
            FlightStatus.NORMAL, 
            0.8
        )
        
        # 测试紧急阶段权重
        emergency_weights = self.reward_function.get_dynamic_weights(
            TaskPhase.EMERGENCY, 
            FlightStatus.LOW_BATTERY, 
            0.15
        )
        
        # 验证紧急状态下紧急决策权重增加
        self.assertGreater(
            emergency_weights['emergency'], 
            normal_weights['emergency'],
            "紧急状态下紧急决策权重应该增加"
        )
        
        print(f"正常阶段权重: {normal_weights}")
        print(f"紧急阶段权重: {emergency_weights}")
    
    def test_reward_function_reset(self):
        """测试奖励函数重置"""
        # 设置一些状态
        self.reward_function.set_flight_plan(self.flight_plan)
        self.reward_function.calculate_reward(
            state=self.normal_state,
            action=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            next_state=self.normal_state,
            info={'task_phase': TaskPhase.NORMAL}
        )
        
        # 重置
        self.reward_function.reset()
        
        # 验证重置后状态
        self.assertIsNone(self.reward_function.flight_plan, "重置后飞行计划应该为None")
        self.assertEqual(len(self.reward_function.reward_history), 0, "重置后奖励历史应该为空")
        
        print("奖励函数重置测试通过")

def run_performance_test():
    """运行性能测试"""
    import time
    
    print("\n=== 性能测试 ===")
    
    # 创建测试环境
    config = {
        'reward_function': {
            'weights': {'tracking': 0.3, 'recovery': 0.2, 'emergency': 0.2, 'safety': 0.2, 'efficiency': 0.1},
            'tracking': {'max_deviation': 15.0, 'heading_weight': 0.3, 'speed_weight': 0.2, 'distance_weight': 0.5, 'expected_speed': 8.0},
            'recovery': {'max_deviation': 50.0, 'max_recovery_time': 120.0, 'success_bonus': 2.0, 'failure_penalty': -2.0, 'time_penalty_factor': 0.01},
            'emergency': {'critical_battery_threshold': 0.2, 'energy_efficiency_weight': 0.3, 'decision_quality_weight': 0.7, 'optimal_decision_bonus': 1.5, 'poor_decision_penalty': -1.0},
            'safety': {'collision_penalty': -20.0, 'altitude_violation_penalty': -8.0, 'speed_violation_penalty': -5.0, 'safe_flight_bonus': 0.2, 'near_collision_penalty': -3.0},
            'efficiency': {'energy_consumption_weight': 0.6, 'time_efficiency_weight': 0.4, 'optimal_energy_bonus': 0.5, 'wasteful_penalty': -0.3}
        }
    }
    
    reward_function = MultiDimensionalRewardFunction(config)
    
    # 创建测试数据
    flight_plan = [np.array([i*25, 0, -10]) for i in range(5)]
    reward_function.set_flight_plan(flight_plan)
    
    test_state = FlightState(
        position=np.array([12.5, 2.0, -10.0]),
        velocity=np.array([8.0, 0.0, 0.0]),
        orientation=np.array([0.0, 0.0, 0.0]),
        battery_level=0.8,
        timestamp=0.0
    )
    
    # 性能测试
    n_iterations = 1000
    start_time = time.time()
    
    for i in range(n_iterations):
        reward = reward_function.calculate_reward(
            state=test_state,
            action=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            next_state=test_state,
            info={
                'task_phase': TaskPhase.NORMAL,
                'flight_status': FlightStatus.NORMAL,
                'collision_detected': False,
                'altitude_violation': False,
                'energy_consumption': 0.001,
                'time_efficiency': 0.8
            }
        )
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / n_iterations
    
    print(f"总计算时间: {total_time:.4f}秒")
    print(f"平均每次计算时间: {avg_time*1000:.4f}毫秒")
    print(f"每秒可计算次数: {1/avg_time:.0f}次")
    
    # 验证性能要求（每次计算应该在10ms以内）
    assert avg_time < 0.01, f"奖励计算性能不达标: {avg_time*1000:.4f}ms > 10ms"
    print("性能测试通过！")

def main():
    """主测试函数"""
    print("开始多元化奖励函数系统测试...")
    print("=" * 50)
    
    # 运行单元测试
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # 运行性能测试
    try:
        run_performance_test()
    except Exception as e:
        print(f"性能测试失败: {e}")
    
    print("\n测试完成！")

if __name__ == '__main__':
    main()