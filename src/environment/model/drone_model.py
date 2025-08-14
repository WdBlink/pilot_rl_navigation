#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强化学习无人机定位导航系统 - 无人机动力学模型

本模块实现无人机的物理动力学模型，用于仿真环境中的状态预测和控制响应。
包括六自由度运动模型、空气动力学效应和传感器噪声模拟。

Author: wdblink
Date: 2024
"""

import numpy as np
import time
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation

from ...utils.data_types import Position3D, FlightAttitude, VelocityVector
from ...utils.logger import get_logger, performance_monitor


@dataclass
class DroneState:
    """无人机状态类
    
    包含无人机的完整状态信息，包括位置、姿态、速度、角速度等。
    
    Attributes:
        position: 位置信息 [x, y, z] (米)
        attitude: 姿态信息 [roll, pitch, yaw] (弧度)
        velocity: 线速度 [vx, vy, vz] (米/秒)
        angular_velocity: 角速度 [wx, wy, wz] (弧度/秒)
        acceleration: 线加速度 [ax, ay, az] (米/秒²)
        angular_acceleration: 角加速度 [alpha_x, alpha_y, alpha_z] (弧度/秒²)
        motor_speeds: 电机转速 [rpm1, rpm2, rpm3, rpm4]
        battery_voltage: 电池电压 (伏特)
        timestamp: 时间戳
    """
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    attitude: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    motor_speeds: np.ndarray = field(default_factory=lambda: np.zeros(4))
    battery_voltage: float = 16.8
    timestamp: float = field(default_factory=time.time)
    
    def to_position3d(self) -> Position3D:
        """转换为Position3D对象
        
        Returns:
            Position3D对象
        """
        return Position3D(
            x=float(self.position[0]),
            y=float(self.position[1]),
            z=float(self.position[2]),
            timestamp=self.timestamp,
            confidence=1.0
        )
    
    def to_flight_attitude(self) -> FlightAttitude:
        """转换为FlightAttitude对象
        
        Returns:
            FlightAttitude对象
        """
        return FlightAttitude(
            roll=float(self.attitude[0]),
            pitch=float(self.attitude[1]),
            yaw=float(self.attitude[2]),
            timestamp=self.timestamp
        )
    
    def to_velocity_vector(self) -> VelocityVector:
        """转换为VelocityVector对象
        
        Returns:
            VelocityVector对象
        """
        return VelocityVector(
            vx=float(self.velocity[0]),
            vy=float(self.velocity[1]),
            vz=float(self.velocity[2]),
            timestamp=self.timestamp
        )
    
    def copy(self) -> 'DroneState':
        """创建状态副本
        
        Returns:
            状态副本
        """
        return DroneState(
            position=self.position.copy(),
            attitude=self.attitude.copy(),
            velocity=self.velocity.copy(),
            angular_velocity=self.angular_velocity.copy(),
            acceleration=self.acceleration.copy(),
            angular_acceleration=self.angular_acceleration.copy(),
            motor_speeds=self.motor_speeds.copy(),
            battery_voltage=self.battery_voltage,
            timestamp=self.timestamp
        )


class DroneModel:
    """无人机动力学模型类
    
    实现四旋翼无人机的完整动力学模型，包括：
    1. 六自由度运动方程
    2. 空气动力学效应
    3. 电机动力学
    4. 传感器噪声
    5. 环境扰动
    """
    
    def __init__(self, 
                 mass: float = 1.5,
                 inertia: Optional[np.ndarray] = None,
                 arm_length: float = 0.25,
                 motor_time_constant: float = 0.02,
                 drag_coefficient: float = 0.01,
                 noise_std: float = 0.01):
        """初始化无人机模型
        
        Args:
            mass: 无人机质量 (kg)
            inertia: 惯性矩阵 (kg·m²)
            arm_length: 机臂长度 (m)
            motor_time_constant: 电机时间常数 (s)
            drag_coefficient: 阻力系数
            noise_std: 噪声标准差
        """
        self.mass = mass
        self.arm_length = arm_length
        self.motor_time_constant = motor_time_constant
        self.drag_coefficient = drag_coefficient
        self.noise_std = noise_std
        
        # 惯性矩阵 (默认对称四旋翼)
        if inertia is None:
            self.inertia = np.diag([0.0347563, 0.0458929, 0.0977])
        else:
            self.inertia = inertia
        
        self.inertia_inv = np.linalg.inv(self.inertia)
        
        # 物理常数
        self.gravity = 9.81  # 重力加速度 (m/s²)
        self.air_density = 1.225  # 空气密度 (kg/m³)
        
        # 电机参数
        self.motor_thrust_coeff = 1.0e-5  # 推力系数
        self.motor_torque_coeff = 1.0e-7  # 扭矩系数
        self.max_motor_speed = 8000.0  # 最大电机转速 (rpm)
        
        # 状态变量
        self.state = DroneState()
        self.desired_motor_speeds = np.zeros(4)
        
        # 环境参数
        self.wind_velocity = np.zeros(3)
        self.temperature = 20.0  # 温度 (°C)
        self.pressure = 101325.0  # 气压 (Pa)
        
        self.logger = get_logger("drone_model")
        self.logger.log_event("info", "drone_model_initialized", {
            "mass": self.mass,
            "arm_length": self.arm_length,
            "inertia": self.inertia.tolist()
        })
    
    @performance_monitor
    def update(self, dt: float, control_input: np.ndarray) -> DroneState:
        """更新无人机状态
        
        Args:
            dt: 时间步长 (s)
            control_input: 控制输入 [thrust, roll_moment, pitch_moment, yaw_moment]
            
        Returns:
            更新后的无人机状态
        """
        # 更新电机转速
        self._update_motor_dynamics(dt, control_input)
        
        # 计算推力和力矩
        thrust, moments = self._calculate_forces_and_moments()
        
        # 更新线性动力学
        self._update_linear_dynamics(dt, thrust)
        
        # 更新角动力学
        self._update_angular_dynamics(dt, moments)
        
        # 添加传感器噪声
        self._add_sensor_noise()
        
        # 更新时间戳
        self.state.timestamp = time.time()
        
        return self.state.copy()
    
    def _update_motor_dynamics(self, dt: float, control_input: np.ndarray) -> None:
        """更新电机动力学
        
        Args:
            dt: 时间步长
            control_input: 控制输入
        """
        # 将控制输入转换为期望电机转速
        self.desired_motor_speeds = self._control_to_motor_speeds(control_input)
        
        # 一阶低通滤波器模拟电机动态响应
        alpha = dt / (self.motor_time_constant + dt)
        self.state.motor_speeds = (
            (1 - alpha) * self.state.motor_speeds + 
            alpha * self.desired_motor_speeds
        )
        
        # 限制电机转速
        self.state.motor_speeds = np.clip(
            self.state.motor_speeds, 0, self.max_motor_speed
        )
    
    def _control_to_motor_speeds(self, control_input: np.ndarray) -> np.ndarray:
        """将控制输入转换为电机转速
        
        Args:
            control_input: 控制输入 [thrust, roll_moment, pitch_moment, yaw_moment]
            
        Returns:
            电机转速数组
        """
        thrust, roll_moment, pitch_moment, yaw_moment = control_input
        
        # 控制分配矩阵 (四旋翼X型配置)
        # motor1: 前右, motor2: 后右, motor3: 后左, motor4: 前左
        allocation_matrix = np.array([
            [1, 1, 1, 1],  # 总推力
            [1, -1, -1, 1],  # 滚转力矩
            [1, 1, -1, -1],  # 俯仰力矩
            [-1, 1, -1, 1]  # 偏航力矩
        ])
        
        # 求解电机推力
        try:
            motor_thrusts = np.linalg.pinv(allocation_matrix) @ control_input
            motor_thrusts = np.maximum(motor_thrusts, 0)  # 确保推力非负
            
            # 转换为电机转速 (rpm)
            motor_speeds = np.sqrt(motor_thrusts / self.motor_thrust_coeff) * 60 / (2 * np.pi)
            
            return motor_speeds
            
        except np.linalg.LinAlgError:
            self.logger.log_event("warning", "control_allocation_failed")
            return np.zeros(4)
    
    def _calculate_forces_and_moments(self) -> Tuple[np.ndarray, np.ndarray]:
        """计算推力和力矩
        
        Returns:
            推力向量和力矩向量
        """
        # 计算各电机推力
        motor_thrusts = self.motor_thrust_coeff * (self.state.motor_speeds * 2 * np.pi / 60) ** 2
        
        # 总推力 (机体坐标系)
        total_thrust = np.sum(motor_thrusts)
        thrust_body = np.array([0, 0, total_thrust])
        
        # 转换到世界坐标系
        rotation_matrix = Rotation.from_euler('xyz', self.state.attitude).as_matrix()
        thrust_world = rotation_matrix @ thrust_body
        
        # 计算力矩
        # 滚转力矩
        roll_moment = self.arm_length * (motor_thrusts[0] + motor_thrusts[3] - 
                                        motor_thrusts[1] - motor_thrusts[2])
        
        # 俯仰力矩
        pitch_moment = self.arm_length * (motor_thrusts[0] + motor_thrusts[1] - 
                                         motor_thrusts[2] - motor_thrusts[3])
        
        # 偏航力矩
        motor_torques = self.motor_torque_coeff * (self.state.motor_speeds * 2 * np.pi / 60) ** 2
        yaw_moment = motor_torques[1] + motor_torques[3] - motor_torques[0] - motor_torques[2]
        
        moments = np.array([roll_moment, pitch_moment, yaw_moment])
        
        return thrust_world, moments
    
    def _update_linear_dynamics(self, dt: float, thrust: np.ndarray) -> None:
        """更新线性动力学
        
        Args:
            dt: 时间步长
            thrust: 推力向量
        """
        # 重力
        gravity_force = np.array([0, 0, -self.mass * self.gravity])
        
        # 空气阻力
        relative_velocity = self.state.velocity - self.wind_velocity
        drag_force = -self.drag_coefficient * np.linalg.norm(relative_velocity) * relative_velocity
        
        # 总力
        total_force = thrust + gravity_force + drag_force
        
        # 更新加速度
        self.state.acceleration = total_force / self.mass
        
        # 更新速度和位置 (欧拉积分)
        self.state.velocity += self.state.acceleration * dt
        self.state.position += self.state.velocity * dt
    
    def _update_angular_dynamics(self, dt: float, moments: np.ndarray) -> None:
        """更新角动力学
        
        Args:
            dt: 时间步长
            moments: 力矩向量
        """
        # 陀螺效应
        gyroscopic_moment = -np.cross(self.state.angular_velocity, 
                                     self.inertia @ self.state.angular_velocity)
        
        # 总力矩
        total_moment = moments + gyroscopic_moment
        
        # 更新角加速度
        self.state.angular_acceleration = self.inertia_inv @ total_moment
        
        # 更新角速度
        self.state.angular_velocity += self.state.angular_acceleration * dt
        
        # 更新姿态 (四元数积分，这里简化为欧拉角)
        self.state.attitude += self.state.angular_velocity * dt
        
        # 限制姿态角范围
        self.state.attitude[0] = np.clip(self.state.attitude[0], -np.pi/2, np.pi/2)  # roll
        self.state.attitude[1] = np.clip(self.state.attitude[1], -np.pi/2, np.pi/2)  # pitch
        self.state.attitude[2] = (self.state.attitude[2] + np.pi) % (2 * np.pi) - np.pi  # yaw
    
    def _add_sensor_noise(self) -> None:
        """添加传感器噪声"""
        if self.noise_std > 0:
            # 位置噪声
            position_noise = np.random.normal(0, self.noise_std, 3)
            self.state.position += position_noise
            
            # 姿态噪声
            attitude_noise = np.random.normal(0, self.noise_std * 0.1, 3)
            self.state.attitude += attitude_noise
            
            # 速度噪声
            velocity_noise = np.random.normal(0, self.noise_std * 0.5, 3)
            self.state.velocity += velocity_noise
    
    def set_wind(self, wind_velocity: np.ndarray) -> None:
        """设置风速
        
        Args:
            wind_velocity: 风速向量 [vx, vy, vz] (m/s)
        """
        self.wind_velocity = wind_velocity.copy()
        self.logger.log_event("info", "wind_updated", {
            "wind_velocity": wind_velocity.tolist()
        })
    
    def set_environment(self, temperature: float, pressure: float) -> None:
        """设置环境参数
        
        Args:
            temperature: 温度 (°C)
            pressure: 气压 (Pa)
        """
        self.temperature = temperature
        self.pressure = pressure
        
        # 更新空气密度
        self.air_density = pressure / (287.05 * (temperature + 273.15))
        
        self.logger.log_event("info", "environment_updated", {
            "temperature": temperature,
            "pressure": pressure,
            "air_density": self.air_density
        })
    
    def reset(self, initial_state: Optional[DroneState] = None) -> DroneState:
        """重置无人机状态
        
        Args:
            initial_state: 初始状态，如果为None则重置为默认状态
            
        Returns:
            重置后的状态
        """
        if initial_state is not None:
            self.state = initial_state.copy()
        else:
            self.state = DroneState()
        
        self.desired_motor_speeds = np.zeros(4)
        
        self.logger.log_event("info", "drone_model_reset")
        
        return self.state.copy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            "mass": self.mass,
            "inertia": self.inertia.tolist(),
            "arm_length": self.arm_length,
            "motor_time_constant": self.motor_time_constant,
            "drag_coefficient": self.drag_coefficient,
            "noise_std": self.noise_std,
            "max_motor_speed": self.max_motor_speed,
            "current_state": {
                "position": self.state.position.tolist(),
                "attitude": self.state.attitude.tolist(),
                "velocity": self.state.velocity.tolist(),
                "motor_speeds": self.state.motor_speeds.tolist()
            }
        }