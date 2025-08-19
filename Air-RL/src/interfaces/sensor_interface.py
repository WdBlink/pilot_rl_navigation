#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强化学习无人机定位导航系统 - 传感器接口模块

本模块定义了统一的传感器接口规范，包括：
1. 传感器基类和接口定义
2. IMU传感器接口实现
3. GPS传感器接口实现
4. 光学传感器接口实现
5. 传感器数据标准化处理
6. 传感器状态监控和故障检测

Author: wdblink
Date: 2024
"""

import time
import threading
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
import numpy as np

# 导入项目模块
from ..utils.data_types import (
    Position3D, IMUData, GPSData, OpticalMatchResult,
    SensorStatus, SystemState
)
from ..utils.logger import logger_manager, performance_monitor


class SensorType(Enum):
    """传感器类型枚举"""
    IMU = "imu"
    GPS = "gps"
    OPTICAL = "optical"
    BAROMETER = "barometer"
    MAGNETOMETER = "magnetometer"
    LIDAR = "lidar"
    CAMERA = "camera"


class SensorState(Enum):
    """传感器状态枚举"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    CALIBRATING = "calibrating"
    MAINTENANCE = "maintenance"


class DataQuality(Enum):
    """数据质量等级"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    INVALID = "invalid"


@dataclass
class SensorInfo:
    """传感器信息"""
    sensor_id: str
    sensor_type: SensorType
    manufacturer: str
    model: str
    firmware_version: str
    sampling_rate: float  # Hz
    accuracy: float
    range_min: float
    range_max: float
    power_consumption: float  # mW
    operating_temperature: tuple  # (min, max) in Celsius
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SensorReading:
    """传感器读数基类"""
    sensor_id: str
    timestamp: float
    data: Any
    quality: DataQuality
    confidence: float
    error_code: Optional[int] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SensorInterface(ABC):
    """传感器接口基类
    
    定义了所有传感器必须实现的基本接口，包括：
    1. 传感器初始化和配置
    2. 数据读取和处理
    3. 状态监控和错误处理
    4. 校准和维护功能
    """
    
    def __init__(self, sensor_info: SensorInfo, config: Dict[str, Any]):
        """
        初始化传感器接口
        
        Args:
            sensor_info: 传感器信息
            config: 配置参数
        """
        self.sensor_info = sensor_info
        self.config = config
        self.state = SensorState.INITIALIZING
        
        # 数据缓存
        self.data_buffer = deque(maxlen=config.get('buffer_size', 1000))
        self.error_history = deque(maxlen=config.get('error_history_size', 100))
        
        # 统计信息
        self.total_readings = 0
        self.valid_readings = 0
        self.last_reading_time = 0.0
        self.average_reading_interval = 0.0
        
        # 回调函数
        self.data_callbacks: List[Callable[[SensorReading], None]] = []
        self.error_callbacks: List[Callable[[str, Exception], None]] = []
        
        # 线程安全
        self.lock = threading.Lock()
        
        # 性能监控
        self.performance_stats = {
            'read_times': deque(maxlen=100),
            'process_times': deque(maxlen=100),
            'error_count': 0,
            'timeout_count': 0
        }
        
        logger_manager.info(f"传感器接口初始化: {sensor_info.sensor_id} ({sensor_info.sensor_type.value})")
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        初始化传感器
        
        Returns:
            是否初始化成功
        """
        pass
    
    @abstractmethod
    def read_data(self) -> Optional[SensorReading]:
        """
        读取传感器数据
        
        Returns:
            传感器读数或None
        """
        pass
    
    @abstractmethod
    def calibrate(self) -> bool:
        """
        校准传感器
        
        Returns:
            是否校准成功
        """
        pass
    
    @abstractmethod
    def self_test(self) -> bool:
        """
        传感器自检
        
        Returns:
            是否自检通过
        """
        pass
    
    def start(self) -> bool:
        """
        启动传感器
        
        Returns:
            是否启动成功
        """
        try:
            if self.state == SensorState.INITIALIZING:
                if not self.initialize():
                    logger_manager.error(f"传感器初始化失败: {self.sensor_info.sensor_id}")
                    return False
            
            self.state = SensorState.ACTIVE
            logger_manager.info(f"传感器启动成功: {self.sensor_info.sensor_id}")
            return True
        
        except Exception as e:
            self.state = SensorState.ERROR
            self._handle_error("传感器启动失败", e)
            return False
    
    def stop(self) -> bool:
        """
        停止传感器
        
        Returns:
            是否停止成功
        """
        try:
            self.state = SensorState.INACTIVE
            logger_manager.info(f"传感器停止: {self.sensor_info.sensor_id}")
            return True
        
        except Exception as e:
            self._handle_error("传感器停止失败", e)
            return False
    
    def get_status(self) -> SensorStatus:
        """
        获取传感器状态
        
        Returns:
            传感器状态信息
        """
        with self.lock:
            return SensorStatus(
                sensor_id=self.sensor_info.sensor_id,
                sensor_type=self.sensor_info.sensor_type.value,
                state=self.state.value,
                last_update=self.last_reading_time,
                data_rate=1.0 / max(0.001, self.average_reading_interval),
                error_count=self.performance_stats['error_count'],
                quality=self._assess_data_quality(),
                metadata={
                    'total_readings': self.total_readings,
                    'valid_readings': self.valid_readings,
                    'success_rate': self.valid_readings / max(1, self.total_readings),
                    'buffer_size': len(self.data_buffer),
                    'average_read_time': np.mean(self.performance_stats['read_times']) if self.performance_stats['read_times'] else 0.0
                }
            )
    
    def add_data_callback(self, callback: Callable[[SensorReading], None]) -> None:
        """
        添加数据回调函数
        
        Args:
            callback: 回调函数
        """
        self.data_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[str, Exception], None]) -> None:
        """
        添加错误回调函数
        
        Args:
            callback: 回调函数
        """
        self.error_callbacks.append(callback)
    
    def get_recent_data(self, count: int = 10) -> List[SensorReading]:
        """
        获取最近的传感器数据
        
        Args:
            count: 数据数量
            
        Returns:
            最近的传感器读数列表
        """
        with self.lock:
            return list(self.data_buffer)[-count:]
    
    def clear_buffer(self) -> None:
        """
        清空数据缓存
        """
        with self.lock:
            self.data_buffer.clear()
            logger_manager.debug(f"传感器数据缓存已清空: {self.sensor_info.sensor_id}")
    
    def reset_statistics(self) -> None:
        """
        重置统计信息
        """
        with self.lock:
            self.total_readings = 0
            self.valid_readings = 0
            self.performance_stats['error_count'] = 0
            self.performance_stats['timeout_count'] = 0
            self.performance_stats['read_times'].clear()
            self.performance_stats['process_times'].clear()
            self.error_history.clear()
            
            logger_manager.info(f"传感器统计信息已重置: {self.sensor_info.sensor_id}")
    
    def _handle_error(self, message: str, exception: Exception) -> None:
        """
        处理传感器错误
        
        Args:
            message: 错误消息
            exception: 异常对象
        """
        error_info = {
            'timestamp': time.time(),
            'message': message,
            'exception': str(exception),
            'sensor_id': self.sensor_info.sensor_id
        }
        
        with self.lock:
            self.error_history.append(error_info)
            self.performance_stats['error_count'] += 1
        
        logger_manager.error(f"传感器错误 [{self.sensor_info.sensor_id}]: {message} - {exception}")
        
        # 调用错误回调
        for callback in self.error_callbacks:
            try:
                callback(message, exception)
            except Exception as e:
                logger_manager.error(f"错误回调执行失败: {e}")
    
    def _assess_data_quality(self) -> DataQuality:
        """
        评估数据质量
        
        Returns:
            数据质量等级
        """
        if not self.data_buffer:
            return DataQuality.INVALID
        
        # 计算成功率
        success_rate = self.valid_readings / max(1, self.total_readings)
        
        # 检查数据时效性
        current_time = time.time()
        time_since_last = current_time - self.last_reading_time
        expected_interval = 1.0 / self.sensor_info.sampling_rate
        
        if time_since_last > 5 * expected_interval:
            return DataQuality.POOR
        
        # 根据成功率评估质量
        if success_rate >= 0.95:
            return DataQuality.EXCELLENT
        elif success_rate >= 0.85:
            return DataQuality.GOOD
        elif success_rate >= 0.70:
            return DataQuality.FAIR
        else:
            return DataQuality.POOR
    
    def _update_statistics(self, reading: SensorReading, read_time: float) -> None:
        """
        更新统计信息
        
        Args:
            reading: 传感器读数
            read_time: 读取时间
        """
        with self.lock:
            self.total_readings += 1
            
            if reading.quality != DataQuality.INVALID:
                self.valid_readings += 1
            
            # 更新时间统计
            current_time = time.time()
            if self.last_reading_time > 0:
                interval = current_time - self.last_reading_time
                if self.average_reading_interval == 0:
                    self.average_reading_interval = interval
                else:
                    # 指数移动平均
                    alpha = 0.1
                    self.average_reading_interval = alpha * interval + (1 - alpha) * self.average_reading_interval
            
            self.last_reading_time = current_time
            self.performance_stats['read_times'].append(read_time)
            
            # 添加到缓存
            self.data_buffer.append(reading)
            
            # 调用数据回调
            for callback in self.data_callbacks:
                try:
                    callback(reading)
                except Exception as e:
                    logger_manager.error(f"数据回调执行失败: {e}")


class IMUSensorInterface(SensorInterface):
    """IMU传感器接口实现
    
    实现IMU传感器的具体功能，包括：
    1. 加速度计和陀螺仪数据读取
    2. 传感器融合和滤波
    3. 校准和偏差补偿
    4. 温度补偿
    """
    
    def __init__(self, sensor_info: SensorInfo, config: Dict[str, Any]):
        """
        初始化IMU传感器接口
        
        Args:
            sensor_info: 传感器信息
            config: 配置参数
        """
        super().__init__(sensor_info, config)
        
        # IMU特定参数
        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self.accel_scale = np.ones(3)
        self.gyro_scale = np.ones(3)
        
        # 滤波器参数
        self.use_filter = config.get('use_filter', True)
        self.filter_alpha = config.get('filter_alpha', 0.8)
        self.filtered_accel = np.zeros(3)
        self.filtered_gyro = np.zeros(3)
        
        # 校准参数
        self.calibration_samples = config.get('calibration_samples', 1000)
        self.calibration_data = {'accel': [], 'gyro': []}
        
        logger_manager.info(f"IMU传感器接口初始化完成: {sensor_info.sensor_id}")
    
    def initialize(self) -> bool:
        """
        初始化IMU传感器
        
        Returns:
            是否初始化成功
        """
        try:
            # 模拟IMU初始化过程
            logger_manager.info(f"正在初始化IMU传感器: {self.sensor_info.sensor_id}")
            
            # 执行自检
            if not self.self_test():
                logger_manager.error("IMU自检失败")
                return False
            
            # 加载校准参数
            self._load_calibration_parameters()
            
            logger_manager.info("IMU传感器初始化成功")
            return True
        
        except Exception as e:
            self._handle_error("IMU初始化失败", e)
            return False
    
    @performance_monitor
    def read_data(self) -> Optional[SensorReading]:
        """
        读取IMU数据
        
        Returns:
            IMU传感器读数
        """
        if self.state != SensorState.ACTIVE:
            return None
        
        start_time = time.time()
        
        try:
            # 模拟读取原始IMU数据
            raw_accel, raw_gyro = self._read_raw_imu_data()
            
            # 应用校准
            calibrated_accel = self._apply_calibration(raw_accel, self.accel_bias, self.accel_scale)
            calibrated_gyro = self._apply_calibration(raw_gyro, self.gyro_bias, self.gyro_scale)
            
            # 应用滤波
            if self.use_filter:
                filtered_accel = self._apply_filter(calibrated_accel, self.filtered_accel)
                filtered_gyro = self._apply_filter(calibrated_gyro, self.filtered_gyro)
                self.filtered_accel = filtered_accel
                self.filtered_gyro = filtered_gyro
            else:
                filtered_accel = calibrated_accel
                filtered_gyro = calibrated_gyro
            
            # 创建IMU数据对象
            imu_data = IMUData(
                acceleration=filtered_accel,
                angular_velocity=filtered_gyro,
                timestamp=time.time(),
                temperature=self._read_temperature(),
                confidence=self._calculate_confidence(filtered_accel, filtered_gyro)
            )
            
            # 评估数据质量
            quality = self._assess_imu_quality(imu_data)
            
            # 创建传感器读数
            reading = SensorReading(
                sensor_id=self.sensor_info.sensor_id,
                timestamp=imu_data.timestamp,
                data=imu_data,
                quality=quality,
                confidence=imu_data.confidence
            )
            
            # 更新统计信息
            read_time = time.time() - start_time
            self._update_statistics(reading, read_time)
            
            return reading
        
        except Exception as e:
            self._handle_error("IMU数据读取失败", e)
            return None
    
    def calibrate(self) -> bool:
        """
        校准IMU传感器
        
        Returns:
            是否校准成功
        """
        try:
            logger_manager.info(f"开始IMU校准: {self.sensor_info.sensor_id}")
            self.state = SensorState.CALIBRATING
            
            # 收集校准数据
            self.calibration_data = {'accel': [], 'gyro': []}
            
            for i in range(self.calibration_samples):
                raw_accel, raw_gyro = self._read_raw_imu_data()
                self.calibration_data['accel'].append(raw_accel)
                self.calibration_data['gyro'].append(raw_gyro)
                
                if i % 100 == 0:
                    logger_manager.debug(f"校准进度: {i}/{self.calibration_samples}")
                
                time.sleep(0.01)  # 10ms间隔
            
            # 计算偏差
            accel_data = np.array(self.calibration_data['accel'])
            gyro_data = np.array(self.calibration_data['gyro'])
            
            # 陀螺仪偏差（静止时应为0）
            self.gyro_bias = np.mean(gyro_data, axis=0)
            
            # 加速度计偏差（静止时应为重力加速度）
            accel_mean = np.mean(accel_data, axis=0)
            gravity_magnitude = 9.81
            self.accel_bias = accel_mean - np.array([0, 0, gravity_magnitude])
            
            # 保存校准参数
            self._save_calibration_parameters()
            
            self.state = SensorState.ACTIVE
            logger_manager.info("IMU校准完成")
            return True
        
        except Exception as e:
            self.state = SensorState.ERROR
            self._handle_error("IMU校准失败", e)
            return False
    
    def self_test(self) -> bool:
        """
        IMU自检
        
        Returns:
            是否自检通过
        """
        try:
            # 模拟自检过程
            logger_manager.debug("执行IMU自检")
            
            # 检查传感器响应
            for _ in range(10):
                raw_accel, raw_gyro = self._read_raw_imu_data()
                
                # 检查数据范围
                if np.any(np.abs(raw_accel) > 50) or np.any(np.abs(raw_gyro) > 10):
                    logger_manager.error("IMU数据超出正常范围")
                    return False
                
                time.sleep(0.01)
            
            logger_manager.debug("IMU自检通过")
            return True
        
        except Exception as e:
            self._handle_error("IMU自检失败", e)
            return False
    
    def _read_raw_imu_data(self) -> tuple:
        """
        读取原始IMU数据（模拟）
        
        Returns:
            (加速度, 角速度)
        """
        # 模拟IMU数据（实际实现中应该从硬件读取）
        accel = np.random.normal(0, 0.1, 3) + np.array([0, 0, 9.81])  # 重力加速度
        gyro = np.random.normal(0, 0.01, 3)  # 角速度
        
        return accel, gyro
    
    def _read_temperature(self) -> float:
        """
        读取温度（模拟）
        
        Returns:
            温度值
        """
        return 25.0 + np.random.normal(0, 1.0)  # 模拟温度
    
    def _apply_calibration(self, data: np.ndarray, bias: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """
        应用校准参数
        
        Args:
            data: 原始数据
            bias: 偏差
            scale: 缩放因子
            
        Returns:
            校准后的数据
        """
        return (data - bias) * scale
    
    def _apply_filter(self, new_data: np.ndarray, old_data: np.ndarray) -> np.ndarray:
        """
        应用低通滤波器
        
        Args:
            new_data: 新数据
            old_data: 旧数据
            
        Returns:
            滤波后的数据
        """
        return self.filter_alpha * old_data + (1 - self.filter_alpha) * new_data
    
    def _calculate_confidence(self, accel: np.ndarray, gyro: np.ndarray) -> float:
        """
        计算数据置信度
        
        Args:
            accel: 加速度数据
            gyro: 角速度数据
            
        Returns:
            置信度值
        """
        # 基于数据稳定性计算置信度
        accel_magnitude = np.linalg.norm(accel)
        gyro_magnitude = np.linalg.norm(gyro)
        
        # 加速度接近重力加速度时置信度较高
        accel_confidence = 1.0 - abs(accel_magnitude - 9.81) / 9.81
        accel_confidence = max(0.0, min(1.0, accel_confidence))
        
        # 角速度较小时置信度较高
        gyro_confidence = 1.0 - min(1.0, gyro_magnitude / 5.0)
        
        return (accel_confidence + gyro_confidence) / 2.0
    
    def _assess_imu_quality(self, imu_data: IMUData) -> DataQuality:
        """
        评估IMU数据质量
        
        Args:
            imu_data: IMU数据
            
        Returns:
            数据质量等级
        """
        if imu_data.confidence >= 0.9:
            return DataQuality.EXCELLENT
        elif imu_data.confidence >= 0.7:
            return DataQuality.GOOD
        elif imu_data.confidence >= 0.5:
            return DataQuality.FAIR
        elif imu_data.confidence >= 0.3:
            return DataQuality.POOR
        else:
            return DataQuality.INVALID
    
    def _load_calibration_parameters(self) -> None:
        """
        加载校准参数
        """
        # 实际实现中应该从文件或配置中加载
        logger_manager.debug("加载IMU校准参数")
    
    def _save_calibration_parameters(self) -> None:
        """
        保存校准参数
        """
        # 实际实现中应该保存到文件
        logger_manager.debug("保存IMU校准参数")


class GPSSensorInterface(SensorInterface):
    """GPS传感器接口实现
    
    实现GPS传感器的具体功能，包括：
    1. 位置和速度数据读取
    2. 卫星信号质量监控
    3. 精度评估和滤波
    4. 多星座支持
    """
    
    def __init__(self, sensor_info: SensorInfo, config: Dict[str, Any]):
        """
        初始化GPS传感器接口
        
        Args:
            sensor_info: 传感器信息
            config: 配置参数
        """
        super().__init__(sensor_info, config)
        
        # GPS特定参数
        self.min_satellites = config.get('min_satellites', 4)
        self.max_hdop = config.get('max_hdop', 5.0)
        self.position_filter_alpha = config.get('position_filter_alpha', 0.7)
        
        # 滤波状态
        self.filtered_position = None
        self.last_valid_position = None
        
        logger_manager.info(f"GPS传感器接口初始化完成: {sensor_info.sensor_id}")
    
    def initialize(self) -> bool:
        """
        初始化GPS传感器
        
        Returns:
            是否初始化成功
        """
        try:
            logger_manager.info(f"正在初始化GPS传感器: {self.sensor_info.sensor_id}")
            
            # 执行自检
            if not self.self_test():
                logger_manager.error("GPS自检失败")
                return False
            
            # 等待GPS信号
            logger_manager.info("等待GPS信号锁定...")
            
            logger_manager.info("GPS传感器初始化成功")
            return True
        
        except Exception as e:
            self._handle_error("GPS初始化失败", e)
            return False
    
    @performance_monitor
    def read_data(self) -> Optional[SensorReading]:
        """
        读取GPS数据
        
        Returns:
            GPS传感器读数
        """
        if self.state != SensorState.ACTIVE:
            return None
        
        start_time = time.time()
        
        try:
            # 模拟读取GPS数据
            raw_gps_data = self._read_raw_gps_data()
            
            if raw_gps_data is None:
                return None
            
            # 验证GPS数据质量
            if not self._validate_gps_data(raw_gps_data):
                logger_manager.warning("GPS数据质量不符合要求")
                return None
            
            # 应用滤波
            filtered_position = self._apply_position_filter(raw_gps_data)
            
            # 创建GPS数据对象
            gps_data = GPSData(
                latitude=filtered_position[0],
                longitude=filtered_position[1],
                altitude=filtered_position[2],
                velocity=raw_gps_data['velocity'],
                heading=raw_gps_data['heading'],
                satellites=raw_gps_data['satellites'],
                hdop=raw_gps_data['hdop'],
                timestamp=time.time(),
                confidence=self._calculate_gps_confidence(raw_gps_data)
            )
            
            # 评估数据质量
            quality = self._assess_gps_quality(gps_data)
            
            # 创建传感器读数
            reading = SensorReading(
                sensor_id=self.sensor_info.sensor_id,
                timestamp=gps_data.timestamp,
                data=gps_data,
                quality=quality,
                confidence=gps_data.confidence
            )
            
            # 更新统计信息
            read_time = time.time() - start_time
            self._update_statistics(reading, read_time)
            
            # 更新最后有效位置
            if quality != DataQuality.INVALID:
                self.last_valid_position = [gps_data.latitude, gps_data.longitude, gps_data.altitude]
            
            return reading
        
        except Exception as e:
            self._handle_error("GPS数据读取失败", e)
            return None
    
    def calibrate(self) -> bool:
        """
        校准GPS传感器（GPS通常不需要用户校准）
        
        Returns:
            是否校准成功
        """
        logger_manager.info("GPS传感器不需要手动校准")
        return True
    
    def self_test(self) -> bool:
        """
        GPS自检
        
        Returns:
            是否自检通过
        """
        try:
            logger_manager.debug("执行GPS自检")
            
            # 检查GPS模块响应
            for _ in range(5):
                raw_data = self._read_raw_gps_data()
                if raw_data is not None:
                    logger_manager.debug("GPS自检通过")
                    return True
                time.sleep(1.0)
            
            logger_manager.error("GPS无响应")
            return False
        
        except Exception as e:
            self._handle_error("GPS自检失败", e)
            return False
    
    def _read_raw_gps_data(self) -> Optional[Dict[str, Any]]:
        """
        读取原始GPS数据（模拟）
        
        Returns:
            GPS数据字典或None
        """
        # 模拟GPS数据（实际实现中应该从GPS模块读取）
        if np.random.random() < 0.1:  # 10%概率无信号
            return None
        
        # 模拟GPS坐标（北京附近）
        base_lat = 39.9042
        base_lon = 116.4074
        base_alt = 50.0
        
        return {
            'latitude': base_lat + np.random.normal(0, 0.0001),
            'longitude': base_lon + np.random.normal(0, 0.0001),
            'altitude': base_alt + np.random.normal(0, 5.0),
            'velocity': np.random.uniform(0, 10),  # m/s
            'heading': np.random.uniform(0, 360),  # degrees
            'satellites': np.random.randint(4, 12),
            'hdop': np.random.uniform(0.5, 3.0)
        }
    
    def _validate_gps_data(self, gps_data: Dict[str, Any]) -> bool:
        """
        验证GPS数据质量
        
        Args:
            gps_data: GPS数据
            
        Returns:
            是否有效
        """
        # 检查卫星数量
        if gps_data['satellites'] < self.min_satellites:
            return False
        
        # 检查HDOP
        if gps_data['hdop'] > self.max_hdop:
            return False
        
        # 检查坐标范围
        if not (-90 <= gps_data['latitude'] <= 90):
            return False
        
        if not (-180 <= gps_data['longitude'] <= 180):
            return False
        
        return True
    
    def _apply_position_filter(self, gps_data: Dict[str, Any]) -> List[float]:
        """
        应用位置滤波
        
        Args:
            gps_data: GPS数据
            
        Returns:
            滤波后的位置
        """
        current_position = [gps_data['latitude'], gps_data['longitude'], gps_data['altitude']]
        
        if self.filtered_position is None:
            self.filtered_position = current_position
        else:
            # 指数移动平均滤波
            alpha = self.position_filter_alpha
            self.filtered_position = [
                alpha * self.filtered_position[i] + (1 - alpha) * current_position[i]
                for i in range(3)
            ]
        
        return self.filtered_position
    
    def _calculate_gps_confidence(self, gps_data: Dict[str, Any]) -> float:
        """
        计算GPS置信度
        
        Args:
            gps_data: GPS数据
            
        Returns:
            置信度值
        """
        # 基于卫星数量的置信度
        sat_confidence = min(1.0, gps_data['satellites'] / 8.0)
        
        # 基于HDOP的置信度
        hdop_confidence = max(0.0, 1.0 - gps_data['hdop'] / 5.0)
        
        return (sat_confidence + hdop_confidence) / 2.0
    
    def _assess_gps_quality(self, gps_data: GPSData) -> DataQuality:
        """
        评估GPS数据质量
        
        Args:
            gps_data: GPS数据
            
        Returns:
            数据质量等级
        """
        if gps_data.satellites >= 8 and gps_data.hdop <= 1.0:
            return DataQuality.EXCELLENT
        elif gps_data.satellites >= 6 and gps_data.hdop <= 2.0:
            return DataQuality.GOOD
        elif gps_data.satellites >= 4 and gps_data.hdop <= 3.0:
            return DataQuality.FAIR
        elif gps_data.satellites >= 4 and gps_data.hdop <= 5.0:
            return DataQuality.POOR
        else:
            return DataQuality.INVALID


class OpticalSensorInterface(SensorInterface):
    """光学传感器接口实现
    
    实现光学定位传感器的具体功能，包括：
    1. 图像采集和处理
    2. 特征匹配和位置估计
    3. 匹配质量评估
    4. 相机参数管理
    """
    
    def __init__(self, sensor_info: SensorInfo, config: Dict[str, Any]):
        """
        初始化光学传感器接口
        
        Args:
            sensor_info: 传感器信息
            config: 配置参数
        """
        super().__init__(sensor_info, config)
        
        # 光学传感器特定参数
        self.min_features = config.get('min_features', 50)
        self.min_match_score = config.get('min_match_score', 0.3)
        self.image_resolution = config.get('image_resolution', (640, 480))
        
        # 性能统计
        self.match_success_count = 0
        self.total_match_attempts = 0
        
        logger_manager.info(f"光学传感器接口初始化完成: {sensor_info.sensor_id}")
    
    def initialize(self) -> bool:
        """
        初始化光学传感器
        
        Returns:
            是否初始化成功
        """
        try:
            logger_manager.info(f"正在初始化光学传感器: {self.sensor_info.sensor_id}")
            
            # 执行自检
            if not self.self_test():
                logger_manager.error("光学传感器自检失败")
                return False
            
            # 初始化相机参数
            self._initialize_camera_parameters()
            
            logger_manager.info("光学传感器初始化成功")
            return True
        
        except Exception as e:
            self._handle_error("光学传感器初始化失败", e)
            return False
    
    @performance_monitor
    def read_data(self) -> Optional[SensorReading]:
        """
        读取光学传感器数据
        
        Returns:
            光学传感器读数
        """
        if self.state != SensorState.ACTIVE:
            return None
        
        start_time = time.time()
        
        try:
            # 模拟光学匹配结果
            match_result = self._perform_optical_matching()
            
            if match_result is None:
                self.total_match_attempts += 1
                return None
            
            # 评估数据质量
            quality = self._assess_optical_quality(match_result)
            
            # 创建传感器读数
            reading = SensorReading(
                sensor_id=self.sensor_info.sensor_id,
                timestamp=match_result.timestamp,
                data=match_result,
                quality=quality,
                confidence=match_result.match_score
            )
            
            # 更新统计信息
            read_time = time.time() - start_time
            self._update_statistics(reading, read_time)
            
            # 更新匹配统计
            self.total_match_attempts += 1
            if quality != DataQuality.INVALID:
                self.match_success_count += 1
            
            return reading
        
        except Exception as e:
            self._handle_error("光学传感器数据读取失败", e)
            return None
    
    def calibrate(self) -> bool:
        """
        校准光学传感器
        
        Returns:
            是否校准成功
        """
        try:
            logger_manager.info(f"开始光学传感器校准: {self.sensor_info.sensor_id}")
            self.state = SensorState.CALIBRATING
            
            # 模拟相机校准过程
            logger_manager.info("执行相机内参校准...")
            time.sleep(2.0)  # 模拟校准时间
            
            # 模拟畸变校正校准
            logger_manager.info("执行畸变校正校准...")
            time.sleep(1.0)
            
            self.state = SensorState.ACTIVE
            logger_manager.info("光学传感器校准完成")
            return True
        
        except Exception as e:
            self.state = SensorState.ERROR
            self._handle_error("光学传感器校准失败", e)
            return False
    
    def self_test(self) -> bool:
        """
        光学传感器自检
        
        Returns:
            是否自检通过
        """
        try:
            logger_manager.debug("执行光学传感器自检")
            
            # 检查相机连接
            if not self._check_camera_connection():
                logger_manager.error("相机连接失败")
                return False
            
            # 检查图像质量
            if not self._check_image_quality():
                logger_manager.error("图像质量检查失败")
                return False
            
            logger_manager.debug("光学传感器自检通过")
            return True
        
        except Exception as e:
            self._handle_error("光学传感器自检失败", e)
            return False
    
    def _initialize_camera_parameters(self) -> None:
        """
        初始化相机参数
        """
        logger_manager.debug("初始化相机参数")
        # 实际实现中应该加载相机内参和畸变参数
    
    def _perform_optical_matching(self) -> Optional[OpticalMatchResult]:
        """
        执行光学匹配（模拟）
        
        Returns:
            光学匹配结果或None
        """
        # 模拟光学匹配过程
        if np.random.random() < 0.2:  # 20%概率匹配失败
            return None
        
        # 模拟匹配结果
        position = Position3D(
            x=np.random.normal(0, 10),
            y=np.random.normal(0, 10),
            z=np.random.uniform(10, 100),
            timestamp=time.time(),
            confidence=np.random.uniform(0.3, 1.0)
        )
        
        return OpticalMatchResult(
            matched_position=position,
            match_score=np.random.uniform(0.3, 1.0),
            feature_count=np.random.randint(50, 200),
            processing_time=np.random.uniform(0.1, 0.5),
            timestamp=time.time()
        )
    
    def _check_camera_connection(self) -> bool:
        """
        检查相机连接
        
        Returns:
            是否连接正常
        """
        # 模拟相机连接检查
        return np.random.random() > 0.05  # 95%概率连接正常
    
    def _check_image_quality(self) -> bool:
        """
        检查图像质量
        
        Returns:
            是否质量合格
        """
        # 模拟图像质量检查
        return np.random.random() > 0.1  # 90%概率质量合格
    
    def _assess_optical_quality(self, match_result: OpticalMatchResult) -> DataQuality:
        """
        评估光学数据质量
        
        Args:
            match_result: 匹配结果
            
        Returns:
            数据质量等级
        """
        if match_result.match_score >= 0.8 and match_result.feature_count >= 100:
            return DataQuality.EXCELLENT
        elif match_result.match_score >= 0.6 and match_result.feature_count >= 75:
            return DataQuality.GOOD
        elif match_result.match_score >= 0.4 and match_result.feature_count >= 50:
            return DataQuality.FAIR
        elif match_result.match_score >= 0.3 and match_result.feature_count >= 30:
            return DataQuality.POOR
        else:
            return DataQuality.INVALID
    
    def get_match_statistics(self) -> Dict[str, Any]:
        """
        获取匹配统计信息
        
        Returns:
            统计信息字典
        """
        with self.lock:
            success_rate = self.match_success_count / max(1, self.total_match_attempts)
            
            return {
                'total_attempts': self.total_match_attempts,
                'successful_matches': self.match_success_count,
                'success_rate': success_rate,
                'image_resolution': self.image_resolution,
                'min_features': self.min_features,
                'min_match_score': self.min_match_score
            }