#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强化学习无人机定位导航系统 - 控制器接口

本模块实现无人机控制器接口，提供位置控制、姿态控制和速度控制功能。
支持多种控制模式和安全保护机制。

Author: wdblink
Date: 2024
"""

import numpy as np
import time
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

from ..utils.data_types import Position3D, FlightAttitude, VelocityVector, ControlMode
from ..utils.logger import get_logger, performance_monitor, log_function_call


class ControlType(Enum):
    """控制类型枚举"""
    POSITION = "position"
    VELOCITY = "velocity"
    ATTITUDE = "attitude"
    THRUST = "thrust"
    MANUAL = "manual"


@dataclass
class ControlCommand:
    """控制命令类
    
    包含无人机控制的所有参数和约束条件。
    
    Attributes:
        control_type: 控制类型
        target_position: 目标位置
        target_velocity: 目标速度
        target_attitude: 目标姿态
        target_thrust: 目标推力
        max_velocity: 最大速度限制
        max_acceleration: 最大加速度限制
        max_angular_velocity: 最大角速度限制
        timeout: 命令超时时间
        priority: 命令优先级
        timestamp: 时间戳
    """
    control_type: ControlType
    target_position: Optional[Position3D] = None
    target_velocity: Optional[VelocityVector] = None
    target_attitude: Optional[FlightAttitude] = None
    target_thrust: Optional[float] = None
    max_velocity: float = 5.0
    max_acceleration: float = 2.0
    max_angular_velocity: float = 1.0
    timeout: float = 10.0
    priority: int = 0
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
    
    def is_expired(self) -> bool:
        """检查命令是否过期
        
        Returns:
            是否过期
        """
        return time.time() - self.timestamp > self.timeout
    
    def validate(self) -> bool:
        """验证命令有效性
        
        Returns:
            命令是否有效
        """
        if self.control_type == ControlType.POSITION and self.target_position is None:
            return False
        if self.control_type == ControlType.VELOCITY and self.target_velocity is None:
            return False
        if self.control_type == ControlType.ATTITUDE and self.target_attitude is None:
            return False
        if self.control_type == ControlType.THRUST and self.target_thrust is None:
            return False
        
        return True


class SafetyChecker:
    """安全检查器类
    
    实现飞行安全检查和保护机制。
    """
    
    def __init__(self, 
                 max_altitude: float = 100.0,
                 min_altitude: float = 0.5,
                 max_distance: float = 500.0,
                 max_velocity: float = 10.0,
                 max_acceleration: float = 5.0,
                 geofence_enabled: bool = True):
        """初始化安全检查器
        
        Args:
            max_altitude: 最大飞行高度 (m)
            min_altitude: 最小飞行高度 (m)
            max_distance: 最大飞行距离 (m)
            max_velocity: 最大飞行速度 (m/s)
            max_acceleration: 最大加速度 (m/s²)
            geofence_enabled: 是否启用地理围栏
        """
        self.max_altitude = max_altitude
        self.min_altitude = min_altitude
        self.max_distance = max_distance
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.geofence_enabled = geofence_enabled
        
        # 地理围栏边界 (经纬度)
        self.geofence_bounds = {
            'min_lat': -90.0,
            'max_lat': 90.0,
            'min_lon': -180.0,
            'max_lon': 180.0
        }
        
        # 起飞点位置
        self.home_position: Optional[Position3D] = None
        
        self.logger = get_logger("safety_checker")
    
    def set_home_position(self, position: Position3D) -> None:
        """设置起飞点位置
        
        Args:
            position: 起飞点位置
        """
        self.home_position = position
        self.logger.log_event("info", "home_position_set", {
            "position": [position.x, position.y, position.z]
        })
    
    def set_geofence(self, min_lat: float, max_lat: float, 
                    min_lon: float, max_lon: float) -> None:
        """设置地理围栏
        
        Args:
            min_lat: 最小纬度
            max_lat: 最大纬度
            min_lon: 最小经度
            max_lon: 最大经度
        """
        self.geofence_bounds = {
            'min_lat': min_lat,
            'max_lat': max_lat,
            'min_lon': min_lon,
            'max_lon': max_lon
        }
        self.logger.log_event("info", "geofence_updated", self.geofence_bounds)
    
    def check_position_safety(self, position: Position3D) -> Tuple[bool, str]:
        """检查位置安全性
        
        Args:
            position: 目标位置
            
        Returns:
            安全性检查结果和原因
        """
        # 高度检查
        if position.z > self.max_altitude:
            return False, f"高度超限: {position.z:.1f}m > {self.max_altitude}m"
        
        if position.z < self.min_altitude:
            return False, f"高度过低: {position.z:.1f}m < {self.min_altitude}m"
        
        # 距离检查
        if self.home_position is not None:
            distance = position.distance_to(self.home_position)
            if distance > self.max_distance:
                return False, f"距离超限: {distance:.1f}m > {self.max_distance}m"
        
        # 地理围栏检查
        if self.geofence_enabled:
            if not (self.geofence_bounds['min_lat'] <= position.y <= self.geofence_bounds['max_lat']):
                return False, f"纬度超出围栏: {position.y}"
            
            if not (self.geofence_bounds['min_lon'] <= position.x <= self.geofence_bounds['max_lon']):
                return False, f"经度超出围栏: {position.x}"
        
        return True, "位置安全"
    
    def check_velocity_safety(self, velocity: VelocityVector) -> Tuple[bool, str]:
        """检查速度安全性
        
        Args:
            velocity: 目标速度
            
        Returns:
            安全性检查结果和原因
        """
        speed = velocity.magnitude()
        if speed > self.max_velocity:
            return False, f"速度超限: {speed:.1f}m/s > {self.max_velocity}m/s"
        
        return True, "速度安全"
    
    def check_command_safety(self, command: ControlCommand, 
                           current_position: Position3D) -> Tuple[bool, str]:
        """检查控制命令安全性
        
        Args:
            command: 控制命令
            current_position: 当前位置
            
        Returns:
            安全性检查结果和原因
        """
        # 检查命令有效性
        if not command.validate():
            return False, "命令参数无效"
        
        # 检查命令是否过期
        if command.is_expired():
            return False, "命令已过期"
        
        # 根据控制类型进行安全检查
        if command.control_type == ControlType.POSITION and command.target_position:
            return self.check_position_safety(command.target_position)
        
        elif command.control_type == ControlType.VELOCITY and command.target_velocity:
            return self.check_velocity_safety(command.target_velocity)
        
        return True, "命令安全"


class BaseController(ABC):
    """控制器基类
    
    定义控制器的基本接口和通用功能。
    """
    
    def __init__(self, name: str):
        """初始化控制器
        
        Args:
            name: 控制器名称
        """
        self.name = name
        self.enabled = False
        self.logger = get_logger(f"controller_{name}")
    
    @abstractmethod
    def compute_control(self, 
                       command: ControlCommand,
                       current_state: Dict[str, Any]) -> np.ndarray:
        """计算控制输出
        
        Args:
            command: 控制命令
            current_state: 当前状态
            
        Returns:
            控制输出
        """
        pass
    
    def enable(self) -> None:
        """启用控制器"""
        self.enabled = True
        self.logger.log_event("info", "controller_enabled")
    
    def disable(self) -> None:
        """禁用控制器"""
        self.enabled = False
        self.logger.log_event("info", "controller_disabled")
    
    @abstractmethod
    def reset(self) -> None:
        """重置控制器状态"""
        pass


class PIDController(BaseController):
    """PID控制器类
    
    实现比例-积分-微分控制算法。
    """
    
    def __init__(self, 
                 name: str,
                 kp: float = 1.0,
                 ki: float = 0.0,
                 kd: float = 0.0,
                 output_limit: float = 10.0,
                 integral_limit: float = 5.0):
        """初始化PID控制器
        
        Args:
            name: 控制器名称
            kp: 比例增益
            ki: 积分增益
            kd: 微分增益
            output_limit: 输出限制
            integral_limit: 积分限制
        """
        super().__init__(name)
        
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit
        
        # 内部状态
        self.integral = 0.0
        self.previous_error = 0.0
        self.last_time = None
    
    def compute_control(self, 
                       command: ControlCommand,
                       current_state: Dict[str, Any]) -> np.ndarray:
        """计算PID控制输出
        
        Args:
            command: 控制命令
            current_state: 当前状态
            
        Returns:
            控制输出
        """
        if not self.enabled:
            return np.array([0.0])
        
        current_time = time.time()
        
        # 计算误差
        if command.control_type == ControlType.POSITION and command.target_position:
            current_pos = current_state.get('position', Position3D(0, 0, 0))
            error = command.target_position.z - current_pos.z  # 简化为高度控制
        else:
            error = 0.0
        
        # 计算时间差
        if self.last_time is None:
            dt = 0.01  # 默认时间步长
        else:
            dt = current_time - self.last_time
        
        if dt <= 0:
            dt = 0.01
        
        # 比例项
        proportional = self.kp * error
        
        # 积分项
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        integral = self.ki * self.integral
        
        # 微分项
        derivative = self.kd * (error - self.previous_error) / dt
        
        # 总输出
        output = proportional + integral + derivative
        output = np.clip(output, -self.output_limit, self.output_limit)
        
        # 更新状态
        self.previous_error = error
        self.last_time = current_time
        
        return np.array([output])
    
    def reset(self) -> None:
        """重置PID控制器状态"""
        self.integral = 0.0
        self.previous_error = 0.0
        self.last_time = None
        self.logger.log_event("info", "pid_controller_reset")
    
    def set_gains(self, kp: float, ki: float, kd: float) -> None:
        """设置PID增益
        
        Args:
            kp: 比例增益
            ki: 积分增益
            kd: 微分增益
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.logger.log_event("info", "pid_gains_updated", {
            "kp": kp, "ki": ki, "kd": kd
        })


class ControllerInterface:
    """控制器接口主类
    
    管理多个控制器和安全检查，提供统一的控制接口。
    """
    
    def __init__(self, safety_config: Optional[Dict[str, Any]] = None):
        """初始化控制器接口
        
        Args:
            safety_config: 安全配置参数
        """
        # 初始化安全检查器
        if safety_config is None:
            safety_config = {}
        
        self.safety_checker = SafetyChecker(**safety_config)
        
        # 控制器字典
        self.controllers: Dict[str, BaseController] = {}
        
        # 当前控制模式
        self.current_mode = ControlMode.MANUAL
        
        # 命令队列
        self.command_queue: List[ControlCommand] = []
        
        # 状态变量
        self.last_command_time = 0.0
        self.emergency_stop = False
        
        self.logger = get_logger("controller_interface")
        
        # 初始化默认控制器
        self._initialize_default_controllers()
        
        self.logger.log_event("info", "controller_interface_initialized")
    
    def _initialize_default_controllers(self) -> None:
        """初始化默认控制器"""
        # 高度控制器
        altitude_controller = PIDController(
            "altitude", kp=2.0, ki=0.1, kd=0.5, output_limit=5.0
        )
        self.add_controller("altitude", altitude_controller)
        
        # 位置控制器 (X轴)
        position_x_controller = PIDController(
            "position_x", kp=1.5, ki=0.05, kd=0.3, output_limit=3.0
        )
        self.add_controller("position_x", position_x_controller)
        
        # 位置控制器 (Y轴)
        position_y_controller = PIDController(
            "position_y", kp=1.5, ki=0.05, kd=0.3, output_limit=3.0
        )
        self.add_controller("position_y", position_y_controller)
    
    def add_controller(self, name: str, controller: BaseController) -> None:
        """添加控制器
        
        Args:
            name: 控制器名称
            controller: 控制器实例
        """
        self.controllers[name] = controller
        self.logger.log_event("info", "controller_added", {"name": name})
    
    def remove_controller(self, name: str) -> bool:
        """移除控制器
        
        Args:
            name: 控制器名称
            
        Returns:
            是否成功移除
        """
        if name in self.controllers:
            del self.controllers[name]
            self.logger.log_event("info", "controller_removed", {"name": name})
            return True
        return False
    
    @performance_monitor
    def execute_command(self, 
                       command: ControlCommand,
                       current_state: Dict[str, Any]) -> Tuple[bool, np.ndarray, str]:
        """执行控制命令
        
        Args:
            command: 控制命令
            current_state: 当前状态
            
        Returns:
            执行结果、控制输出和状态信息
        """
        # 紧急停止检查
        if self.emergency_stop:
            return False, np.zeros(4), "紧急停止激活"
        
        # 安全检查
        current_position = current_state.get('position', Position3D(0, 0, 0))
        safety_ok, safety_reason = self.safety_checker.check_command_safety(
            command, current_position
        )
        
        if not safety_ok:
            self.logger.log_event("warning", "safety_check_failed", {
                "reason": safety_reason
            })
            return False, np.zeros(4), f"安全检查失败: {safety_reason}"
        
        # 根据控制类型执行相应控制
        try:
            if command.control_type == ControlType.POSITION:
                control_output = self._execute_position_control(command, current_state)
            elif command.control_type == ControlType.VELOCITY:
                control_output = self._execute_velocity_control(command, current_state)
            elif command.control_type == ControlType.ATTITUDE:
                control_output = self._execute_attitude_control(command, current_state)
            else:
                control_output = np.zeros(4)
            
            self.last_command_time = time.time()
            
            return True, control_output, "命令执行成功"
            
        except Exception as e:
            self.logger.log_event("error", "command_execution_failed", {
                "error": str(e)
            })
            return False, np.zeros(4), f"命令执行失败: {str(e)}"
    
    def _execute_position_control(self, 
                                 command: ControlCommand,
                                 current_state: Dict[str, Any]) -> np.ndarray:
        """执行位置控制
        
        Args:
            command: 位置控制命令
            current_state: 当前状态
            
        Returns:
            控制输出 [thrust, roll_moment, pitch_moment, yaw_moment]
        """
        if not command.target_position:
            return np.zeros(4)
        
        # 高度控制
        altitude_output = self.controllers["altitude"].compute_control(command, current_state)
        
        # X轴位置控制
        x_output = self.controllers["position_x"].compute_control(command, current_state)
        
        # Y轴位置控制
        y_output = self.controllers["position_y"].compute_control(command, current_state)
        
        # 组合控制输出
        control_output = np.array([
            float(altitude_output[0]),  # 推力
            float(y_output[0]),         # 滚转力矩
            float(x_output[0]),         # 俯仰力矩
            0.0                         # 偏航力矩
        ])
        
        return control_output
    
    def _execute_velocity_control(self, 
                                 command: ControlCommand,
                                 current_state: Dict[str, Any]) -> np.ndarray:
        """执行速度控制
        
        Args:
            command: 速度控制命令
            current_state: 当前状态
            
        Returns:
            控制输出
        """
        # 简化实现，实际应用中需要更复杂的速度控制器
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    def _execute_attitude_control(self, 
                                 command: ControlCommand,
                                 current_state: Dict[str, Any]) -> np.ndarray:
        """执行姿态控制
        
        Args:
            command: 姿态控制命令
            current_state: 当前状态
            
        Returns:
            控制输出
        """
        # 简化实现，实际应用中需要姿态控制器
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    def set_emergency_stop(self, stop: bool) -> None:
        """设置紧急停止
        
        Args:
            stop: 是否紧急停止
        """
        self.emergency_stop = stop
        if stop:
            self.logger.log_event("warning", "emergency_stop_activated")
        else:
            self.logger.log_event("info", "emergency_stop_deactivated")
    
    def set_control_mode(self, mode: ControlMode) -> None:
        """设置控制模式
        
        Args:
            mode: 控制模式
        """
        self.current_mode = mode
        self.logger.log_event("info", "control_mode_changed", {"mode": mode.value})
    
    def enable_all_controllers(self) -> None:
        """启用所有控制器"""
        for controller in self.controllers.values():
            controller.enable()
        self.logger.log_event("info", "all_controllers_enabled")
    
    def disable_all_controllers(self) -> None:
        """禁用所有控制器"""
        for controller in self.controllers.values():
            controller.disable()
        self.logger.log_event("info", "all_controllers_disabled")
    
    def reset_all_controllers(self) -> None:
        """重置所有控制器"""
        for controller in self.controllers.values():
            controller.reset()
        self.logger.log_event("info", "all_controllers_reset")
    
    def get_status(self) -> Dict[str, Any]:
        """获取控制器状态
        
        Returns:
            状态信息字典
        """
        controller_status = {}
        for name, controller in self.controllers.items():
            controller_status[name] = {
                "enabled": controller.enabled,
                "type": type(controller).__name__
            }
        
        return {
            "current_mode": self.current_mode.value,
            "emergency_stop": self.emergency_stop,
            "last_command_time": self.last_command_time,
            "controllers": controller_status,
            "command_queue_length": len(self.command_queue)
        }