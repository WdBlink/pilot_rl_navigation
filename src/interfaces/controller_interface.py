#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强化学习无人机定位导航系统 - 控制器接口模块

本模块定义了无人机控制器的统一接口规范，包括：
1. 飞行控制器基类和接口定义
2. ArduPilot控制器接口实现
3. PX4控制器接口实现
4. 控制指令标准化处理
5. 飞行状态监控和安全检查
6. 紧急处理和故障恢复

Author: wdblink
Date: 2024
"""

import time
import threading
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable, Tuple
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
import numpy as np

# 导入项目模块
from ..utils.data_types import (
    Position3D, FlightCommand, FlightMode, SystemState,
    SafetyStatus, ControllerStatus
)
from ..utils.logger import logger_manager, performance_monitor


class ControllerType(Enum):
    """控制器类型枚举"""
    ARDUPILOT = "ardupilot"
    PX4 = "px4"
    BETAFLIGHT = "betaflight"
    CLEANFLIGHT = "cleanflight"
    INAV = "inav"
    CUSTOM = "custom"


class ConnectionType(Enum):
    """连接类型枚举"""
    SERIAL = "serial"
    UDP = "udp"
    TCP = "tcp"
    USB = "usb"
    BLUETOOTH = "bluetooth"
    WIFI = "wifi"


class ControllerState(Enum):
    """控制器状态枚举"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ARMED = "armed"
    DISARMED = "disarmed"
    ERROR = "error"
    EMERGENCY = "emergency"


class SafetyLevel(Enum):
    """安全等级枚举"""
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class ControllerInfo:
    """控制器信息"""
    controller_id: str
    controller_type: ControllerType
    firmware_version: str
    hardware_version: str
    connection_type: ConnectionType
    connection_string: str
    capabilities: List[str]
    max_velocity: float  # m/s
    max_acceleration: float  # m/s²
    max_altitude: float  # m
    battery_cells: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FlightLimits:
    """飞行限制参数"""
    max_velocity_xy: float = 10.0  # m/s
    max_velocity_z: float = 5.0   # m/s
    max_acceleration: float = 5.0  # m/s²
    max_altitude: float = 120.0   # m
    min_altitude: float = 1.0     # m
    max_distance: float = 500.0   # m
    max_tilt_angle: float = 30.0  # degrees
    min_battery_voltage: float = 14.0  # V
    max_wind_speed: float = 15.0  # m/s
    geofence_enabled: bool = True
    geofence_radius: float = 100.0  # m
    home_position: Optional[Position3D] = None


@dataclass
class SafetyCheck:
    """安全检查结果"""
    check_name: str
    passed: bool
    level: SafetyLevel
    message: str
    timestamp: float
    value: Optional[float] = None
    threshold: Optional[float] = None


class ControllerInterface(ABC):
    """控制器接口基类
    
    定义了所有飞行控制器必须实现的基本接口，包括：
    1. 连接和通信管理
    2. 飞行控制指令发送
    3. 状态监控和数据获取
    4. 安全检查和紧急处理
    """
    
    def __init__(self, controller_info: ControllerInfo, config: Dict[str, Any]):
        """
        初始化控制器接口
        
        Args:
            controller_info: 控制器信息
            config: 配置参数
        """
        self.controller_info = controller_info
        self.config = config
        self.state = ControllerState.DISCONNECTED
        
        # 飞行限制
        self.flight_limits = FlightLimits(**config.get('flight_limits', {}))
        
        # 状态缓存
        self.current_position = Position3D(0, 0, 0, time.time())
        self.current_velocity = np.zeros(3)
        self.current_attitude = np.zeros(3)  # roll, pitch, yaw
        self.battery_voltage = 0.0
        self.battery_percentage = 0.0
        
        # 命令历史
        self.command_history = deque(maxlen=config.get('command_history_size', 100))
        self.safety_checks = deque(maxlen=config.get('safety_history_size', 50))
        
        # 回调函数
        self.status_callbacks: List[Callable[[ControllerStatus], None]] = []
        self.safety_callbacks: List[Callable[[SafetyCheck], None]] = []
        self.emergency_callbacks: List[Callable[[str, Exception], None]] = []
        
        # 统计信息
        self.total_commands = 0
        self.successful_commands = 0
        self.connection_attempts = 0
        self.last_heartbeat = 0.0
        
        # 线程安全
        self.lock = threading.Lock()
        
        # 性能监控
        self.performance_stats = {
            'command_times': deque(maxlen=100),
            'response_times': deque(maxlen=100),
            'connection_errors': 0,
            'timeout_errors': 0
        }
        
        logger_manager.info(f"控制器接口初始化: {controller_info.controller_id} ({controller_info.controller_type.value})")
    
    @abstractmethod
    def connect(self) -> bool:
        """
        连接到控制器
        
        Returns:
            是否连接成功
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        断开控制器连接
        
        Returns:
            是否断开成功
        """
        pass
    
    @abstractmethod
    def arm(self) -> bool:
        """
        解锁无人机
        
        Returns:
            是否解锁成功
        """
        pass
    
    @abstractmethod
    def disarm(self) -> bool:
        """
        锁定无人机
        
        Returns:
            是否锁定成功
        """
        pass
    
    @abstractmethod
    def takeoff(self, altitude: float) -> bool:
        """
        起飞到指定高度
        
        Args:
            altitude: 目标高度 (m)
            
        Returns:
            是否起飞成功
        """
        pass
    
    @abstractmethod
    def land(self) -> bool:
        """
        降落
        
        Returns:
            是否降落成功
        """
        pass
    
    @abstractmethod
    def goto_position(self, position: Position3D, velocity: float = 5.0) -> bool:
        """
        飞行到指定位置
        
        Args:
            position: 目标位置
            velocity: 飞行速度 (m/s)
            
        Returns:
            是否执行成功
        """
        pass
    
    @abstractmethod
    def set_velocity(self, velocity: np.ndarray) -> bool:
        """
        设置速度向量
        
        Args:
            velocity: 速度向量 [vx, vy, vz] (m/s)
            
        Returns:
            是否设置成功
        """
        pass
    
    @abstractmethod
    def set_attitude(self, roll: float, pitch: float, yaw: float, thrust: float) -> bool:
        """
        设置姿态和推力
        
        Args:
            roll: 横滚角 (degrees)
            pitch: 俯仰角 (degrees)
            yaw: 偏航角 (degrees)
            thrust: 推力 (0-1)
            
        Returns:
            是否设置成功
        """
        pass
    
    @abstractmethod
    def get_position(self) -> Optional[Position3D]:
        """
        获取当前位置
        
        Returns:
            当前位置或None
        """
        pass
    
    @abstractmethod
    def get_velocity(self) -> Optional[np.ndarray]:
        """
        获取当前速度
        
        Returns:
            当前速度向量或None
        """
        pass
    
    @abstractmethod
    def get_attitude(self) -> Optional[np.ndarray]:
        """
        获取当前姿态
        
        Returns:
            当前姿态 [roll, pitch, yaw] 或None
        """
        pass
    
    @abstractmethod
    def get_battery_status(self) -> Optional[Tuple[float, float]]:
        """
        获取电池状态
        
        Returns:
            (电压, 电量百分比) 或None
        """
        pass
    
    def execute_command(self, command: FlightCommand) -> bool:
        """
        执行飞行命令
        
        Args:
            command: 飞行命令
            
        Returns:
            是否执行成功
        """
        start_time = time.time()
        
        try:
            # 安全检查
            if not self._validate_command(command):
                logger_manager.warning(f"命令验证失败: {command}")
                return False
            
            # 执行命令
            success = self._execute_command_internal(command)
            
            # 记录命令历史
            execution_time = time.time() - start_time
            command_record = {
                'command': command,
                'success': success,
                'timestamp': time.time(),
                'execution_time': execution_time
            }
            
            with self.lock:
                self.command_history.append(command_record)
                self.total_commands += 1
                if success:
                    self.successful_commands += 1
                self.performance_stats['command_times'].append(execution_time)
            
            if success:
                logger_manager.debug(f"命令执行成功: {command.command_type}, 耗时: {execution_time:.3f}s")
            else:
                logger_manager.warning(f"命令执行失败: {command.command_type}")
            
            return success
        
        except Exception as e:
            self._handle_error(f"命令执行异常: {command.command_type}", e)
            return False
    
    def _execute_command_internal(self, command: FlightCommand) -> bool:
        """
        内部命令执行逻辑
        
        Args:
            command: 飞行命令
            
        Returns:
            是否执行成功
        """
        if command.command_type == "takeoff":
            return self.takeoff(command.parameters.get('altitude', 10.0))
        elif command.command_type == "land":
            return self.land()
        elif command.command_type == "goto":
            position = command.parameters.get('position')
            velocity = command.parameters.get('velocity', 5.0)
            if position:
                return self.goto_position(position, velocity)
        elif command.command_type == "set_velocity":
            velocity = command.parameters.get('velocity')
            if velocity is not None:
                return self.set_velocity(np.array(velocity))
        elif command.command_type == "set_attitude":
            params = command.parameters
            return self.set_attitude(
                params.get('roll', 0),
                params.get('pitch', 0),
                params.get('yaw', 0),
                params.get('thrust', 0.5)
            )
        elif command.command_type == "arm":
            return self.arm()
        elif command.command_type == "disarm":
            return self.disarm()
        else:
            logger_manager.warning(f"未知命令类型: {command.command_type}")
            return False
    
    def _validate_command(self, command: FlightCommand) -> bool:
        """
        验证飞行命令
        
        Args:
            command: 飞行命令
            
        Returns:
            是否有效
        """
        # 检查连接状态
        if self.state == ControllerState.DISCONNECTED:
            logger_manager.error("控制器未连接")
            return False
        
        # 检查紧急状态
        if self.state == ControllerState.EMERGENCY:
            if command.command_type not in ['land', 'disarm']:
                logger_manager.error("紧急状态下只允许降落和锁定命令")
                return False
        
        # 检查飞行限制
        if command.command_type == "goto":
            position = command.parameters.get('position')
            if position and not self._check_position_limits(position):
                return False
        
        elif command.command_type == "set_velocity":
            velocity = command.parameters.get('velocity')
            if velocity and not self._check_velocity_limits(np.array(velocity)):
                return False
        
        elif command.command_type == "takeoff":
            altitude = command.parameters.get('altitude', 10.0)
            if altitude > self.flight_limits.max_altitude:
                logger_manager.error(f"起飞高度超限: {altitude} > {self.flight_limits.max_altitude}")
                return False
        
        return True
    
    def _check_position_limits(self, position: Position3D) -> bool:
        """
        检查位置限制
        
        Args:
            position: 目标位置
            
        Returns:
            是否在限制范围内
        """
        # 检查高度限制
        if position.z > self.flight_limits.max_altitude:
            logger_manager.error(f"目标高度超限: {position.z} > {self.flight_limits.max_altitude}")
            return False
        
        if position.z < self.flight_limits.min_altitude:
            logger_manager.error(f"目标高度过低: {position.z} < {self.flight_limits.min_altitude}")
            return False
        
        # 检查地理围栏
        if self.flight_limits.geofence_enabled and self.flight_limits.home_position:
            distance = np.sqrt(
                (position.x - self.flight_limits.home_position.x) ** 2 +
                (position.y - self.flight_limits.home_position.y) ** 2
            )
            
            if distance > self.flight_limits.geofence_radius:
                logger_manager.error(f"目标位置超出地理围栏: {distance} > {self.flight_limits.geofence_radius}")
                return False
        
        return True
    
    def _check_velocity_limits(self, velocity: np.ndarray) -> bool:
        """
        检查速度限制
        
        Args:
            velocity: 速度向量
            
        Returns:
            是否在限制范围内
        """
        # 检查水平速度
        horizontal_speed = np.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
        if horizontal_speed > self.flight_limits.max_velocity_xy:
            logger_manager.error(f"水平速度超限: {horizontal_speed} > {self.flight_limits.max_velocity_xy}")
            return False
        
        # 检查垂直速度
        if abs(velocity[2]) > self.flight_limits.max_velocity_z:
            logger_manager.error(f"垂直速度超限: {abs(velocity[2])} > {self.flight_limits.max_velocity_z}")
            return False
        
        return True
    
    def perform_safety_checks(self) -> List[SafetyCheck]:
        """
        执行安全检查
        
        Returns:
            安全检查结果列表
        """
        checks = []
        current_time = time.time()
        
        # 电池电压检查
        if self.battery_voltage > 0:
            battery_check = SafetyCheck(
                check_name="battery_voltage",
                passed=self.battery_voltage >= self.flight_limits.min_battery_voltage,
                level=SafetyLevel.CRITICAL if self.battery_voltage < self.flight_limits.min_battery_voltage else SafetyLevel.SAFE,
                message=f"电池电压: {self.battery_voltage:.1f}V",
                timestamp=current_time,
                value=self.battery_voltage,
                threshold=self.flight_limits.min_battery_voltage
            )
            checks.append(battery_check)
        
        # 连接状态检查
        connection_timeout = current_time - self.last_heartbeat
        connection_check = SafetyCheck(
            check_name="connection",
            passed=connection_timeout < 5.0,
            level=SafetyLevel.CRITICAL if connection_timeout > 10.0 else SafetyLevel.WARNING if connection_timeout > 5.0 else SafetyLevel.SAFE,
            message=f"连接超时: {connection_timeout:.1f}s",
            timestamp=current_time,
            value=connection_timeout,
            threshold=5.0
        )
        checks.append(connection_check)
        
        # 高度检查
        if self.current_position.z > 0:
            altitude_check = SafetyCheck(
                check_name="altitude",
                passed=self.current_position.z <= self.flight_limits.max_altitude,
                level=SafetyLevel.CRITICAL if self.current_position.z > self.flight_limits.max_altitude else SafetyLevel.SAFE,
                message=f"当前高度: {self.current_position.z:.1f}m",
                timestamp=current_time,
                value=self.current_position.z,
                threshold=self.flight_limits.max_altitude
            )
            checks.append(altitude_check)
        
        # 地理围栏检查
        if self.flight_limits.geofence_enabled and self.flight_limits.home_position:
            distance = np.sqrt(
                (self.current_position.x - self.flight_limits.home_position.x) ** 2 +
                (self.current_position.y - self.flight_limits.home_position.y) ** 2
            )
            
            geofence_check = SafetyCheck(
                check_name="geofence",
                passed=distance <= self.flight_limits.geofence_radius,
                level=SafetyLevel.CRITICAL if distance > self.flight_limits.geofence_radius else SafetyLevel.SAFE,
                message=f"距离起飞点: {distance:.1f}m",
                timestamp=current_time,
                value=distance,
                threshold=self.flight_limits.geofence_radius
            )
            checks.append(geofence_check)
        
        # 保存检查结果
        with self.lock:
            self.safety_checks.extend(checks)
        
        # 调用安全回调
        for check in checks:
            if not check.passed:
                for callback in self.safety_callbacks:
                    try:
                        callback(check)
                    except Exception as e:
                        logger_manager.error(f"安全回调执行失败: {e}")
        
        return checks
    
    def emergency_stop(self) -> bool:
        """
        紧急停止
        
        Returns:
            是否执行成功
        """
        try:
            logger_manager.warning("执行紧急停止")
            self.state = ControllerState.EMERGENCY
            
            # 立即停止所有运动
            self.set_velocity(np.zeros(3))
            
            # 启动紧急降落
            success = self.land()
            
            if success:
                logger_manager.info("紧急停止执行成功")
            else:
                logger_manager.error("紧急停止执行失败")
            
            return success
        
        except Exception as e:
            self._handle_error("紧急停止异常", e)
            return False
    
    def get_status(self) -> ControllerStatus:
        """
        获取控制器状态
        
        Returns:
            控制器状态信息
        """
        with self.lock:
            return ControllerStatus(
                controller_id=self.controller_info.controller_id,
                state=self.state.value,
                connection_type=self.controller_info.connection_type.value,
                last_heartbeat=self.last_heartbeat,
                battery_voltage=self.battery_voltage,
                battery_percentage=self.battery_percentage,
                current_position=self.current_position,
                current_velocity=self.current_velocity.tolist(),
                current_attitude=self.current_attitude.tolist(),
                flight_mode=self._get_current_flight_mode(),
                armed=self.state == ControllerState.ARMED,
                metadata={
                    'total_commands': self.total_commands,
                    'successful_commands': self.successful_commands,
                    'success_rate': self.successful_commands / max(1, self.total_commands),
                    'connection_attempts': self.connection_attempts,
                    'average_command_time': np.mean(self.performance_stats['command_times']) if self.performance_stats['command_times'] else 0.0
                }
            )
    
    def add_status_callback(self, callback: Callable[[ControllerStatus], None]) -> None:
        """
        添加状态回调函数
        
        Args:
            callback: 回调函数
        """
        self.status_callbacks.append(callback)
    
    def add_safety_callback(self, callback: Callable[[SafetyCheck], None]) -> None:
        """
        添加安全回调函数
        
        Args:
            callback: 回调函数
        """
        self.safety_callbacks.append(callback)
    
    def add_emergency_callback(self, callback: Callable[[str, Exception], None]) -> None:
        """
        添加紧急回调函数
        
        Args:
            callback: 回调函数
        """
        self.emergency_callbacks.append(callback)
    
    def get_command_history(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        获取命令历史
        
        Args:
            count: 历史记录数量
            
        Returns:
            命令历史列表
        """
        with self.lock:
            return list(self.command_history)[-count:]
    
    def get_safety_history(self, count: int = 10) -> List[SafetyCheck]:
        """
        获取安全检查历史
        
        Args:
            count: 历史记录数量
            
        Returns:
            安全检查历史列表
        """
        with self.lock:
            return list(self.safety_checks)[-count:]
    
    def reset_statistics(self) -> None:
        """
        重置统计信息
        """
        with self.lock:
            self.total_commands = 0
            self.successful_commands = 0
            self.connection_attempts = 0
            self.command_history.clear()
            self.safety_checks.clear()
            self.performance_stats['command_times'].clear()
            self.performance_stats['response_times'].clear()
            self.performance_stats['connection_errors'] = 0
            self.performance_stats['timeout_errors'] = 0
        
        logger_manager.info(f"控制器统计信息已重置: {self.controller_info.controller_id}")
    
    def _handle_error(self, message: str, exception: Exception) -> None:
        """
        处理控制器错误
        
        Args:
            message: 错误消息
            exception: 异常对象
        """
        logger_manager.error(f"控制器错误 [{self.controller_info.controller_id}]: {message} - {exception}")
        
        # 调用紧急回调
        for callback in self.emergency_callbacks:
            try:
                callback(message, exception)
            except Exception as e:
                logger_manager.error(f"紧急回调执行失败: {e}")
    
    def _get_current_flight_mode(self) -> str:
        """
        获取当前飞行模式
        
        Returns:
            飞行模式字符串
        """
        # 子类应该重写此方法
        return "UNKNOWN"
    
    def _update_heartbeat(self) -> None:
        """
        更新心跳时间
        """
        self.last_heartbeat = time.time()


class ArduPilotInterface(ControllerInterface):
    """ArduPilot控制器接口实现
    
    实现ArduPilot飞控的具体功能，包括：
    1. MAVLink协议通信
    2. ArduPilot特定命令
    3. 飞行模式管理
    4. 参数配置
    """
    
    def __init__(self, controller_info: ControllerInfo, config: Dict[str, Any]):
        """
        初始化ArduPilot接口
        
        Args:
            controller_info: 控制器信息
            config: 配置参数
        """
        super().__init__(controller_info, config)
        
        # ArduPilot特定参数
        self.system_id = config.get('system_id', 1)
        self.component_id = config.get('component_id', 1)
        self.target_system = config.get('target_system', 1)
        self.target_component = config.get('target_component', 1)
        
        # 飞行模式映射
        self.flight_modes = {
            'STABILIZE': 0,
            'ACRO': 1,
            'ALT_HOLD': 2,
            'AUTO': 3,
            'GUIDED': 4,
            'LOITER': 5,
            'RTL': 6,
            'CIRCLE': 7,
            'LAND': 9,
            'BRAKE': 17,
            'THROW': 18,
            'GUIDED_NOGPS': 20
        }
        
        self.current_flight_mode = "UNKNOWN"
        
        logger_manager.info(f"ArduPilot接口初始化完成: {controller_info.controller_id}")
    
    def connect(self) -> bool:
        """
        连接到ArduPilot
        
        Returns:
            是否连接成功
        """
        try:
            logger_manager.info(f"正在连接ArduPilot: {self.controller_info.connection_string}")
            self.state = ControllerState.CONNECTING
            self.connection_attempts += 1
            
            # 模拟连接过程
            time.sleep(2.0)
            
            # 模拟连接成功
            if np.random.random() > 0.1:  # 90%成功率
                self.state = ControllerState.CONNECTED
                self._update_heartbeat()
                logger_manager.info("ArduPilot连接成功")
                return True
            else:
                self.state = ControllerState.DISCONNECTED
                self.performance_stats['connection_errors'] += 1
                logger_manager.error("ArduPilot连接失败")
                return False
        
        except Exception as e:
            self.state = ControllerState.ERROR
            self._handle_error("ArduPilot连接异常", e)
            return False
    
    def disconnect(self) -> bool:
        """
        断开ArduPilot连接
        
        Returns:
            是否断开成功
        """
        try:
            logger_manager.info("断开ArduPilot连接")
            self.state = ControllerState.DISCONNECTED
            return True
        
        except Exception as e:
            self._handle_error("ArduPilot断开连接异常", e)
            return False
    
    @performance_monitor
    def arm(self) -> bool:
        """
        解锁ArduPilot
        
        Returns:
            是否解锁成功
        """
        try:
            if self.state != ControllerState.CONNECTED:
                logger_manager.error("ArduPilot未连接")
                return False
            
            logger_manager.info("正在解锁ArduPilot")
            
            # 执行解锁前检查
            safety_checks = self.perform_safety_checks()
            critical_failures = [check for check in safety_checks if check.level == SafetyLevel.CRITICAL and not check.passed]
            
            if critical_failures:
                logger_manager.error(f"解锁失败，存在严重安全问题: {[check.check_name for check in critical_failures]}")
                return False
            
            # 模拟解锁过程
            time.sleep(1.0)
            
            if np.random.random() > 0.05:  # 95%成功率
                self.state = ControllerState.ARMED
                logger_manager.info("ArduPilot解锁成功")
                return True
            else:
                logger_manager.error("ArduPilot解锁失败")
                return False
        
        except Exception as e:
            self._handle_error("ArduPilot解锁异常", e)
            return False
    
    def disarm(self) -> bool:
        """
        锁定ArduPilot
        
        Returns:
            是否锁定成功
        """
        try:
            logger_manager.info("正在锁定ArduPilot")
            
            # 模拟锁定过程
            time.sleep(0.5)
            
            self.state = ControllerState.DISARMED
            logger_manager.info("ArduPilot锁定成功")
            return True
        
        except Exception as e:
            self._handle_error("ArduPilot锁定异常", e)
            return False
    
    def takeoff(self, altitude: float) -> bool:
        """
        ArduPilot起飞
        
        Args:
            altitude: 目标高度 (m)
            
        Returns:
            是否起飞成功
        """
        try:
            if self.state != ControllerState.ARMED:
                logger_manager.error("ArduPilot未解锁")
                return False
            
            logger_manager.info(f"ArduPilot起飞到 {altitude}m")
            
            # 切换到GUIDED模式
            if not self._set_flight_mode("GUIDED"):
                logger_manager.error("切换到GUIDED模式失败")
                return False
            
            # 模拟起飞过程
            time.sleep(3.0)
            
            # 更新位置
            self.current_position.z = altitude
            self.current_position.timestamp = time.time()
            
            logger_manager.info(f"ArduPilot起飞成功，当前高度: {altitude}m")
            return True
        
        except Exception as e:
            self._handle_error("ArduPilot起飞异常", e)
            return False
    
    def land(self) -> bool:
        """
        ArduPilot降落
        
        Returns:
            是否降落成功
        """
        try:
            logger_manager.info("ArduPilot开始降落")
            
            # 切换到LAND模式
            if not self._set_flight_mode("LAND"):
                logger_manager.error("切换到LAND模式失败")
                return False
            
            # 模拟降落过程
            time.sleep(5.0)
            
            # 更新位置
            self.current_position.z = 0.0
            self.current_position.timestamp = time.time()
            
            # 自动锁定
            self.state = ControllerState.DISARMED
            
            logger_manager.info("ArduPilot降落成功")
            return True
        
        except Exception as e:
            self._handle_error("ArduPilot降落异常", e)
            return False
    
    def goto_position(self, position: Position3D, velocity: float = 5.0) -> bool:
        """
        ArduPilot飞行到指定位置
        
        Args:
            position: 目标位置
            velocity: 飞行速度 (m/s)
            
        Returns:
            是否执行成功
        """
        try:
            if self.state != ControllerState.ARMED:
                logger_manager.error("ArduPilot未解锁")
                return False
            
            logger_manager.info(f"ArduPilot飞行到位置: ({position.x:.1f}, {position.y:.1f}, {position.z:.1f})")
            
            # 确保在GUIDED模式
            if self.current_flight_mode != "GUIDED":
                if not self._set_flight_mode("GUIDED"):
                    logger_manager.error("切换到GUIDED模式失败")
                    return False
            
            # 模拟飞行过程
            time.sleep(2.0)
            
            # 更新位置
            self.current_position = position
            self.current_position.timestamp = time.time()
            
            logger_manager.info("ArduPilot到达目标位置")
            return True
        
        except Exception as e:
            self._handle_error("ArduPilot位置飞行异常", e)
            return False
    
    def set_velocity(self, velocity: np.ndarray) -> bool:
        """
        设置ArduPilot速度
        
        Args:
            velocity: 速度向量 [vx, vy, vz] (m/s)
            
        Returns:
            是否设置成功
        """
        try:
            if self.state != ControllerState.ARMED:
                logger_manager.error("ArduPilot未解锁")
                return False
            
            logger_manager.debug(f"设置ArduPilot速度: [{velocity[0]:.1f}, {velocity[1]:.1f}, {velocity[2]:.1f}]")
            
            # 更新当前速度
            self.current_velocity = velocity.copy()
            
            return True
        
        except Exception as e:
            self._handle_error("ArduPilot速度设置异常", e)
            return False
    
    def set_attitude(self, roll: float, pitch: float, yaw: float, thrust: float) -> bool:
        """
        设置ArduPilot姿态
        
        Args:
            roll: 横滚角 (degrees)
            pitch: 俯仰角 (degrees)
            yaw: 偏航角 (degrees)
            thrust: 推力 (0-1)
            
        Returns:
            是否设置成功
        """
        try:
            if self.state != ControllerState.ARMED:
                logger_manager.error("ArduPilot未解锁")
                return False
            
            logger_manager.debug(f"设置ArduPilot姿态: roll={roll:.1f}, pitch={pitch:.1f}, yaw={yaw:.1f}, thrust={thrust:.2f}")
            
            # 更新当前姿态
            self.current_attitude = np.array([roll, pitch, yaw])
            
            return True
        
        except Exception as e:
            self._handle_error("ArduPilot姿态设置异常", e)
            return False
    
    def get_position(self) -> Optional[Position3D]:
        """
        获取ArduPilot当前位置
        
        Returns:
            当前位置或None
        """
        try:
            # 模拟位置更新
            self.current_position.timestamp = time.time()
            self._update_heartbeat()
            return self.current_position
        
        except Exception as e:
            self._handle_error("ArduPilot位置获取异常", e)
            return None
    
    def get_velocity(self) -> Optional[np.ndarray]:
        """
        获取ArduPilot当前速度
        
        Returns:
            当前速度向量或None
        """
        try:
            self._update_heartbeat()
            return self.current_velocity.copy()
        
        except Exception as e:
            self._handle_error("ArduPilot速度获取异常", e)
            return None
    
    def get_attitude(self) -> Optional[np.ndarray]:
        """
        获取ArduPilot当前姿态
        
        Returns:
            当前姿态 [roll, pitch, yaw] 或None
        """
        try:
            self._update_heartbeat()
            return self.current_attitude.copy()
        
        except Exception as e:
            self._handle_error("ArduPilot姿态获取异常", e)
            return None
    
    def get_battery_status(self) -> Optional[Tuple[float, float]]:
        """
        获取ArduPilot电池状态
        
        Returns:
            (电压, 电量百分比) 或None
        """
        try:
            # 模拟电池状态
            self.battery_voltage = 16.8 - np.random.uniform(0, 2.0)  # 模拟电压下降
            self.battery_percentage = max(0, (self.battery_voltage - 14.0) / 2.8 * 100)  # 简单的百分比计算
            
            self._update_heartbeat()
            return self.battery_voltage, self.battery_percentage
        
        except Exception as e:
            self._handle_error("ArduPilot电池状态获取异常", e)
            return None
    
    def _set_flight_mode(self, mode: str) -> bool:
        """
        设置飞行模式
        
        Args:
            mode: 飞行模式名称
            
        Returns:
            是否设置成功
        """
        try:
            if mode not in self.flight_modes:
                logger_manager.error(f"未知飞行模式: {mode}")
                return False
            
            logger_manager.info(f"切换ArduPilot飞行模式: {self.current_flight_mode} -> {mode}")
            
            # 模拟模式切换
            time.sleep(0.5)
            
            self.current_flight_mode = mode
            logger_manager.info(f"ArduPilot飞行模式切换成功: {mode}")
            return True
        
        except Exception as e:
            self._handle_error(f"ArduPilot飞行模式切换异常: {mode}", e)
            return False
    
    def _get_current_flight_mode(self) -> str:
        """
        获取当前飞行模式
        
        Returns:
            飞行模式字符串
        """
        return self.current_flight_mode
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        获取ArduPilot参数
        
        Returns:
            参数字典
        """
        # 模拟参数获取
        return {
            'WPNAV_SPEED': 500,  # cm/s
            'WPNAV_SPEED_UP': 250,  # cm/s
            'WPNAV_SPEED_DN': 150,  # cm/s
            'WPNAV_ACCEL': 100,  # cm/s²
            'FENCE_ENABLE': 1,
            'FENCE_RADIUS': 100,  # m
            'FENCE_ALT_MAX': 120,  # m
            'BATT_LOW_VOLT': 14.0,  # V
            'BATT_CRT_VOLT': 13.0,  # V
        }
    
    def set_parameter(self, param_name: str, value: Any) -> bool:
        """
        设置ArduPilot参数
        
        Args:
            param_name: 参数名称
            value: 参数值
            
        Returns:
            是否设置成功
        """
        try:
            logger_manager.info(f"设置ArduPilot参数: {param_name} = {value}")
            
            # 模拟参数设置
            time.sleep(0.1)
            
            logger_manager.info(f"ArduPilot参数设置成功: {param_name}")
            return True
        
        except Exception as e:
            self._handle_error(f"ArduPilot参数设置异常: {param_name}", e)
            return False