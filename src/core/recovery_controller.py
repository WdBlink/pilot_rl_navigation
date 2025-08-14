#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强化学习无人机定位导航系统 - 自主恢复控制模块

本模块实现位置丢失后的智能恢复策略，包括螺旋搜索、紧急返航等功能。
当无人机失去位置信息时，该模块能够自动执行恢复策略，确保飞行安全。

Author: wdblink
Date: 2024
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from collections import deque

# 导入项目模块
from ..utils.data_types import Position3D, FlightAttitude, SystemState, RLAction, ControlMode
from ..utils.logger import logger_manager, performance_monitor


class RecoveryState(Enum):
    """恢复状态枚举"""
    NORMAL = "normal"
    POSITION_LOST = "position_lost"
    SEARCHING = "searching"
    RETURNING = "returning"
    EMERGENCY = "emergency"


class SearchStrategy(Enum):
    """搜索策略枚举"""
    SPIRAL = "spiral"
    GRID = "grid"
    RANDOM = "random"
    RETURN_HOME = "return_home"


@dataclass
class RecoveryCommand:
    """恢复控制指令"""
    target_position: Optional[Position3D]
    velocity: Tuple[float, float, float]  # (vx, vy, vz)
    duration: float
    strategy: SearchStrategy
    priority: int  # 优先级，数值越高优先级越高
    safety_check: bool = True


class AutonomousRecoveryController:
    """自主恢复控制模块
    
    实现位置丢失后的智能恢复策略，包括：
    1. 位置丢失检测
    2. 螺旋搜索策略
    3. 网格搜索策略
    4. 紧急返航策略
    5. 安全性检查
    """
    
    def __init__(self, rl_agent, flight_interface, config: Dict[str, Any]):
        """
        初始化自主恢复控制器
        
        Args:
            rl_agent: 强化学习智能体实例
            flight_interface: 飞行控制接口
            config: 配置参数字典
        """
        self.rl_agent = rl_agent
        self.flight_interface = flight_interface
        self.config = config
        
        # 恢复状态管理
        self.recovery_state = RecoveryState.NORMAL
        self.last_known_position = None
        self.recovery_start_time = None
        self.position_history = deque(maxlen=config.get('position_history_length', 100))
        
        # 搜索参数
        self.search_radius = config.get('search_radius', 50.0)  # 搜索半径(米)
        self.search_altitude = config.get('search_altitude', 100.0)  # 搜索高度(米)
        self.max_search_time = config.get('max_search_time', 300.0)  # 最大搜索时间(秒)
        self.spiral_turns = config.get('spiral_turns', 3)  # 螺旋圈数
        self.points_per_turn = config.get('points_per_turn', 8)  # 每圈点数
        
        # 安全参数
        self.min_altitude = config.get('min_altitude', 50.0)  # 最小飞行高度
        self.max_velocity = config.get('max_velocity', 10.0)  # 最大飞行速度
        self.battery_threshold = config.get('battery_threshold', 0.2)  # 电池阈值
        
        # 恢复策略映射
        self.recovery_strategies = {
            ControlMode.NORMAL: self._normal_operation,
            ControlMode.RECOVERY: self._recovery_operation,
            ControlMode.EMERGENCY: self._emergency_operation
        }
        
        # 搜索策略映射
        self.search_strategies = {
            SearchStrategy.SPIRAL: self._spiral_search_strategy,
            SearchStrategy.GRID: self._grid_search_strategy,
            SearchStrategy.RANDOM: self._random_search_strategy,
            SearchStrategy.RETURN_HOME: self._return_home_strategy
        }
        
        logger_manager.info("自主恢复控制器初始化完成")
    
    @performance_monitor
    def execute_recovery(self, system_state: SystemState) -> RecoveryCommand:
        """
        执行恢复控制策略
        
        Args:
            system_state: 当前系统状态
            
        Returns:
            恢复控制指令
        """
        # 1. 更新位置历史
        self._update_position_history(system_state)
        
        # 2. 检测位置丢失状态
        loss_detected = self._detect_position_loss(system_state)
        
        # 3. 更新恢复状态
        self._update_recovery_state(loss_detected, system_state)
        
        # 4. RL智能体决策
        action = self.rl_agent.predict(system_state)
        
        # 5. 执行对应的恢复策略
        recovery_command = self.recovery_strategies[action.control_mode](system_state, action)
        
        # 6. 安全性检查
        safe_command = self._safety_check(recovery_command, system_state)
        
        # 7. 记录恢复状态
        self._log_recovery_status(system_state, safe_command)
        
        return safe_command
    
    def _detect_position_loss(self, system_state: SystemState) -> bool:
        """
        检测位置丢失状态
        
        Args:
            system_state: 当前系统状态
            
        Returns:
            是否检测到位置丢失
        """
        # 检查光学定位是否可用
        optical_lost = (system_state.optical_position is None or 
                       system_state.optical_position.match_score < 0.3)
        
        # 检查惯导累积误差
        inertial_error_high = system_state.historical_error > self.config.get('max_inertial_error', 10.0)
        
        # 检查管道偏离程度
        pipeline_deviation_high = abs(system_state.pipeline_deviation) > self.config.get('max_pipeline_deviation', 20.0)
        
        # 综合判断
        position_lost = optical_lost and (inertial_error_high or pipeline_deviation_high)
        
        return position_lost
    
    def _update_recovery_state(self, loss_detected: bool, system_state: SystemState) -> None:
        """
        更新恢复状态
        
        Args:
            loss_detected: 是否检测到位置丢失
            system_state: 当前系统状态
        """
        current_time = time.time()
        
        if loss_detected and self.recovery_state == RecoveryState.NORMAL:
            # 进入位置丢失状态
            self.recovery_state = RecoveryState.POSITION_LOST
            self.recovery_start_time = current_time
            self.last_known_position = system_state.inertial_position
            logger_manager.warning("检测到位置丢失，启动恢复程序")
            
        elif not loss_detected and self.recovery_state != RecoveryState.NORMAL:
            # 恢复正常状态
            self.recovery_state = RecoveryState.NORMAL
            self.recovery_start_time = None
            logger_manager.info("位置恢复正常")
            
        elif (self.recovery_state in [RecoveryState.POSITION_LOST, RecoveryState.SEARCHING] and
              current_time - self.recovery_start_time > self.max_search_time):
            # 搜索超时，进入紧急状态
            self.recovery_state = RecoveryState.EMERGENCY
            logger_manager.error("搜索超时，进入紧急模式")
            
        elif system_state.battery_level < self.battery_threshold:
            # 电池电量不足，进入紧急状态
            self.recovery_state = RecoveryState.EMERGENCY
            logger_manager.error("电池电量不足，进入紧急模式")
    
    def _normal_operation(self, system_state: SystemState, action: RLAction) -> RecoveryCommand:
        """
        正常运行模式
        
        Args:
            system_state: 系统状态
            action: RL动作
            
        Returns:
            恢复控制指令
        """
        # 正常模式下，根据管道调整进行微调
        dx, dy = action.pipeline_adjustment
        current_pos = system_state.inertial_position
        
        target_position = Position3D(
            x=current_pos.x + dx,
            y=current_pos.y + dy,
            z=current_pos.z,
            timestamp=current_pos.timestamp
        )
        
        return RecoveryCommand(
            target_position=target_position,
            velocity=(2.0, 2.0, 0.0),
            duration=1.0,
            strategy=SearchStrategy.SPIRAL,
            priority=1
        )
    
    def _recovery_operation(self, system_state: SystemState, action: RLAction) -> RecoveryCommand:
        """
        恢复运行模式
        
        Args:
            system_state: 系统状态
            action: RL动作
            
        Returns:
            恢复控制指令
        """
        self.recovery_state = RecoveryState.SEARCHING
        
        # 选择搜索策略
        if self.last_known_position is not None:
            # 使用螺旋搜索
            waypoints = self._spiral_search_strategy(
                self.last_known_position, 
                self.search_radius
            )
            
            if waypoints:
                target_position = waypoints[0]  # 取第一个搜索点
                return RecoveryCommand(
                    target_position=target_position,
                    velocity=(5.0, 5.0, 0.0),
                    duration=3.0,
                    strategy=SearchStrategy.SPIRAL,
                    priority=3
                )
        
        # 默认悬停
        return RecoveryCommand(
            target_position=None,
            velocity=(0.0, 0.0, 0.0),
            duration=5.0,
            strategy=SearchStrategy.SPIRAL,
            priority=2
        )
    
    def _emergency_operation(self, system_state: SystemState, action: RLAction) -> RecoveryCommand:
        """
        紧急运行模式
        
        Args:
            system_state: 系统状态
            action: RL动作
            
        Returns:
            恢复控制指令
        """
        self.recovery_state = RecoveryState.EMERGENCY
        
        # 紧急返航
        home_position = self.config.get('home_position', Position3D(x=0, y=0, z=100, timestamp=0))
        
        return RecoveryCommand(
            target_position=home_position,
            velocity=(8.0, 8.0, 2.0),
            duration=10.0,
            strategy=SearchStrategy.RETURN_HOME,
            priority=5
        )
    
    def _spiral_search_strategy(self, center_pos: Position3D, radius: float) -> List[Position3D]:
        """
        螺旋搜索轨迹生成
        
        Args:
            center_pos: 搜索中心位置
            radius: 搜索半径
            
        Returns:
            螺旋搜索轨迹点列表
        """
        waypoints = []
        
        for i in range(self.spiral_turns * self.points_per_turn):
            angle = 2 * np.pi * i / self.points_per_turn
            current_radius = radius * (i / (self.spiral_turns * self.points_per_turn))
            
            x = center_pos.x + current_radius * np.cos(angle)
            y = center_pos.y + current_radius * np.sin(angle)
            z = max(center_pos.z, self.search_altitude)  # 确保搜索高度
            
            waypoints.append(Position3D(x=x, y=y, z=z, timestamp=0))
        
        return waypoints
    
    def _grid_search_strategy(self, center_pos: Position3D, radius: float) -> List[Position3D]:
        """
        网格搜索轨迹生成
        
        Args:
            center_pos: 搜索中心位置
            radius: 搜索半径
            
        Returns:
            网格搜索轨迹点列表
        """
        waypoints = []
        grid_size = self.config.get('grid_size', 5)
        step = 2 * radius / grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                x = center_pos.x - radius + i * step
                y = center_pos.y - radius + j * step
                z = max(center_pos.z, self.search_altitude)
                
                waypoints.append(Position3D(x=x, y=y, z=z, timestamp=0))
        
        return waypoints
    
    def _random_search_strategy(self, center_pos: Position3D, radius: float) -> List[Position3D]:
        """
        随机搜索轨迹生成
        
        Args:
            center_pos: 搜索中心位置
            radius: 搜索半径
            
        Returns:
            随机搜索轨迹点列表
        """
        waypoints = []
        num_points = self.config.get('random_search_points', 10)
        
        for _ in range(num_points):
            # 在圆形区域内随机生成点
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, radius)
            
            x = center_pos.x + r * np.cos(angle)
            y = center_pos.y + r * np.sin(angle)
            z = max(center_pos.z, self.search_altitude)
            
            waypoints.append(Position3D(x=x, y=y, z=z, timestamp=0))
        
        return waypoints
    
    def _return_home_strategy(self, center_pos: Position3D, radius: float) -> List[Position3D]:
        """
        返航策略
        
        Args:
            center_pos: 当前位置
            radius: 未使用
            
        Returns:
            返航轨迹点列表
        """
        home_position = self.config.get('home_position', Position3D(x=0, y=0, z=100, timestamp=0))
        return [home_position]
    
    def _safety_check(self, command: RecoveryCommand, system_state: SystemState) -> RecoveryCommand:
        """
        安全性检查
        
        Args:
            command: 原始恢复指令
            system_state: 系统状态
            
        Returns:
            安全检查后的恢复指令
        """
        if not command.safety_check:
            return command
        
        # 检查目标位置高度
        if command.target_position and command.target_position.z < self.min_altitude:
            command.target_position.z = self.min_altitude
            logger_manager.warning(f"目标高度过低，调整为最小高度: {self.min_altitude}m")
        
        # 检查速度限制
        vx, vy, vz = command.velocity
        max_v = self.max_velocity
        
        if abs(vx) > max_v:
            vx = max_v if vx > 0 else -max_v
        if abs(vy) > max_v:
            vy = max_v if vy > 0 else -max_v
        if abs(vz) > max_v:
            vz = max_v if vz > 0 else -max_v
        
        command.velocity = (vx, vy, vz)
        
        # 检查电池电量
        if system_state.battery_level < self.battery_threshold:
            # 强制返航
            home_position = self.config.get('home_position', Position3D(x=0, y=0, z=100, timestamp=0))
            command.target_position = home_position
            command.strategy = SearchStrategy.RETURN_HOME
            command.priority = 5
            logger_manager.error("电池电量不足，强制返航")
        
        return command
    
    def _update_position_history(self, system_state: SystemState) -> None:
        """
        更新位置历史记录
        
        Args:
            system_state: 系统状态
        """
        self.position_history.append({
            'timestamp': time.time(),
            'position': system_state.inertial_position,
            'optical_available': system_state.optical_position is not None,
            'confidence': system_state.inertial_position.confidence if system_state.inertial_position else 0.0
        })
    
    def _log_recovery_status(self, system_state: SystemState, command: RecoveryCommand) -> None:
        """
        记录恢复状态日志
        
        Args:
            system_state: 系统状态
            command: 恢复指令
        """
        logger_manager.info(
            f"恢复状态: {self.recovery_state.value}, "
            f"策略: {command.strategy.value}, "
            f"优先级: {command.priority}, "
            f"电池: {system_state.battery_level:.2f}"
        )
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """
        获取恢复统计信息
        
        Returns:
            恢复统计信息字典
        """
        return {
            'current_state': self.recovery_state.value,
            'recovery_start_time': self.recovery_start_time,
            'last_known_position': self.last_known_position,
            'position_history_length': len(self.position_history),
            'search_radius': self.search_radius,
            'max_search_time': self.max_search_time
        }
    
    def reset_recovery_state(self) -> None:
        """
        重置恢复状态
        """
        self.recovery_state = RecoveryState.NORMAL
        self.recovery_start_time = None
        self.last_known_position = None
        self.position_history.clear()
        logger_manager.info("恢复状态已重置")