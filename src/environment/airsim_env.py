#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AirSim训练环境模块

该模块实现了基于AirSim的无人机仿真训练环境，包括：
- AirSim环境初始化和连接管理
- 基于地理坐标的飞行轨迹规划
- 自动飞行控制和状态监控
- 卫星底图和相机图像获取
- 传感器数据采集和处理

Author: wdblink
Date: 2024
"""

import asyncio
import logging
import math
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2
from dataclasses import dataclass
from enum import Enum

try:
    import airsim
except ImportError:
    print("Warning: AirSim package not found. Please install airsim package.")
    airsim = None

from ..utils.data_types import Position3D, FlightAttitude, VelocityVector
from ..utils.logger import setup_logger
from ..utils.config import AirSimConfig
from ..core.reward_function import (
    MultiDimensionalRewardFunction, FlightStatus, TaskPhase, 
    DecisionType, RewardComponents
)


class FlightMode(Enum):
    """飞行模式枚举"""
    MANUAL = "Manual"
    GUIDED = "Guided"
    AUTO = "Auto"
    RTL = "RTL"  # Return to Launch
    LAND = "Land"


class MissionStatus(Enum):
    """任务状态枚举"""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Waypoint:
    """航点数据类"""
    latitude: float  # 纬度
    longitude: float  # 经度
    altitude: float  # 海拔高度(米)
    speed: float = 5.0  # 飞行速度(m/s)
    hold_time: float = 0.0  # 悬停时间(秒)
    heading: Optional[float] = None  # 航向角(度)


@dataclass
class FlightPlan:
    """飞行计划数据类"""
    waypoints: List[Waypoint]
    name: str = "default_mission"
    description: str = ""
    max_speed: float = 10.0
    return_to_launch: bool = True


@dataclass
class SensorData:
    """传感器数据类"""
    timestamp: float
    position: Position3D
    attitude: FlightAttitude
    velocity: VelocityVector
    camera_image: Optional[np.ndarray] = None
    satellite_image: Optional[np.ndarray] = None
    imu_data: Optional[Dict[str, Any]] = None
    gps_data: Optional[Dict[str, Any]] = None


class AirSimEnvironment:
    """
    AirSim仿真环境类
    
    该类负责管理AirSim仿真环境，包括无人机连接、飞行控制、
    传感器数据采集和图像获取等功能。
    """
    
    def __init__(self, config: AirSimConfig):
        """
        初始化AirSim环境
        
        Args:
            config: AirSim配置对象
        """
        self.config = config
        self.logger = setup_logger("AirSimEnvironment")
        
        # AirSim客户端
        self.client: Optional[airsim.MultirotorClient] = None
        self.vehicle_name = config.vehicle_name
        
        # 飞行状态
        self.is_connected = False
        self.is_armed = False
        self.current_mode = FlightMode.MANUAL
        self.mission_status = MissionStatus.IDLE
        
        # 飞行计划
        self.current_plan: Optional[FlightPlan] = None
        self.current_waypoint_index = 0
        
        # 传感器数据缓存
        self.latest_sensor_data: Optional[SensorData] = None
        
        # 地理坐标原点(用于坐标转换)
        self.origin_lat = config.origin_latitude
        self.origin_lon = config.origin_longitude
        self.origin_alt = config.origin_altitude
        
    async def initialize(self) -> bool:
        """
        初始化AirSim环境连接
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            if airsim is None:
                self.logger.error("AirSim package not available")
                return False
                
            # 连接到AirSim
            self.client = airsim.MultirotorClient(
                ip=self.config.host,
                port=self.config.port
            )
            
            self.logger.info(f"Connecting to AirSim at {self.config.host}:{self.config.port}")
            self.client.confirmConnection()
            
            # 启用API控制
            self.client.enableApiControl(True, self.vehicle_name)
            
            # 解锁无人机
            await self._arm_drone()
            
            # 设置相机参数
            await self._setup_cameras()
            
            self.is_connected = True
            self.logger.info("AirSim environment initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize AirSim environment: {e}")
            return False
    
    async def _arm_drone(self) -> bool:
        """
        解锁无人机
        
        Returns:
            bool: 解锁是否成功
        """
        try:
            self.client.armDisarm(True, self.vehicle_name)
            self.is_armed = True
            self.logger.info("Drone armed successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to arm drone: {e}")
            return False
    
    async def _setup_cameras(self):
        """
        设置相机参数
        """
        try:
            # 设置下视相机(机腹垂直向下)
            camera_pose = airsim.Pose(
                airsim.Vector3r(0, 0, 0),  # 位置偏移
                airsim.to_quaternion(np.pi/2, 0, 0)  # 向下90度
            )
            
            # 应用相机姿态
            success = self.client.simSetCameraPose(
                "bottom_center", camera_pose, self.vehicle_name
            )
            
            if success:
                self.logger.info("Camera setup completed")
            else:
                self.logger.warning("Camera setup may have failed")
                
        except Exception as e:
            self.logger.error(f"Failed to setup cameras: {e}")
    
    def create_flight_plan(self, waypoints: List[Tuple[float, float, float]], 
                          name: str = "auto_mission") -> FlightPlan:
        """
        创建基于地理坐标的飞行计划
        
        Args:
            waypoints: 航点列表，格式为[(lat, lon, alt), ...]
            name: 任务名称
            
        Returns:
            FlightPlan: 飞行计划对象
        """
        waypoint_objects = []
        
        for i, (lat, lon, alt) in enumerate(waypoints):
            waypoint = Waypoint(
                latitude=lat,
                longitude=lon,
                altitude=alt,
                speed=self.config.default_speed,
                hold_time=self.config.waypoint_hold_time
            )
            waypoint_objects.append(waypoint)
        
        plan = FlightPlan(
            waypoints=waypoint_objects,
            name=name,
            max_speed=self.config.max_speed,
            return_to_launch=True
        )
        
        self.logger.info(f"Created flight plan '{name}' with {len(waypoints)} waypoints")
        return plan
    
    def _geo_to_ned(self, lat: float, lon: float, alt: float) -> Tuple[float, float, float]:
        """
        将地理坐标转换为NED坐标系
        
        Args:
            lat: 纬度
            lon: 经度
            alt: 海拔高度
            
        Returns:
            Tuple[float, float, float]: NED坐标(North, East, Down)
        """
        # 简化的坐标转换(实际应用中需要更精确的转换)
        R_earth = 6371000  # 地球半径(米)
        
        # 计算相对于原点的偏移
        d_lat = np.radians(lat - self.origin_lat)
        d_lon = np.radians(lon - self.origin_lon)
        
        # 转换为NED坐标
        north = d_lat * R_earth
        east = d_lon * R_earth * np.cos(np.radians(self.origin_lat))
        down = -(alt - self.origin_alt)  # NED系统中向下为正
        
        return north, east, down
    
    async def execute_flight_plan(self, plan: FlightPlan) -> bool:
        """
        执行飞行计划
        
        Args:
            plan: 飞行计划对象
            
        Returns:
            bool: 执行是否成功
        """
        if not self.is_connected or not self.is_armed:
            self.logger.error("Drone not ready for flight")
            return False
        
        self.current_plan = plan
        self.mission_status = MissionStatus.EXECUTING
        self.current_waypoint_index = 0
        
        try:
            # 起飞到安全高度
            await self._takeoff()
            
            # 依次飞向各个航点
            for i, waypoint in enumerate(plan.waypoints):
                self.current_waypoint_index = i
                self.logger.info(f"Flying to waypoint {i+1}/{len(plan.waypoints)}")
                
                # 转换坐标
                north, east, down = self._geo_to_ned(
                    waypoint.latitude, waypoint.longitude, waypoint.altitude
                )
                
                # 飞向航点
                await self._fly_to_position(north, east, down, waypoint.speed)
                
                # 悬停指定时间
                if waypoint.hold_time > 0:
                    self.logger.info(f"Holding at waypoint for {waypoint.hold_time}s")
                    await asyncio.sleep(waypoint.hold_time)
                
                # 采集传感器数据
                await self._collect_sensor_data()
            
            # 返航(如果设置)
            if plan.return_to_launch:
                await self._return_to_launch()
            
            self.mission_status = MissionStatus.COMPLETED
            self.logger.info("Flight plan completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Flight plan execution failed: {e}")
            self.mission_status = MissionStatus.FAILED
            return False
    
    async def _takeoff(self, altitude: float = 10.0):
        """
        起飞到指定高度
        
        Args:
            altitude: 起飞高度(米)
        """
        self.logger.info(f"Taking off to {altitude}m")
        self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()
        
        # 等待起飞完成
        await asyncio.sleep(3)
    
    async def _fly_to_position(self, north: float, east: float, down: float, speed: float):
        """
        飞向指定NED坐标位置
        
        Args:
            north: 北向距离(米)
            east: 东向距离(米)
            down: 向下距离(米)
            speed: 飞行速度(m/s)
        """
        # 移动到目标位置
        self.client.moveToPositionAsync(
            north, east, down, speed, vehicle_name=self.vehicle_name
        ).join()
    
    async def _return_to_launch(self):
        """
        返回起飞点并降落
        """
        self.logger.info("Returning to launch point")
        
        # 返回起飞点
        self.client.goHomeAsync(vehicle_name=self.vehicle_name).join()
        
        # 降落
        await asyncio.sleep(2)
        self.client.landAsync(vehicle_name=self.vehicle_name).join()
    
    async def _collect_sensor_data(self) -> SensorData:
        """
        采集当前位置的传感器数据
        
        Returns:
            SensorData: 传感器数据对象
        """
        try:
            timestamp = time.time()
            
            # 获取无人机状态
            state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
            
            # 提取位置信息
            pos = state.kinematics_estimated.position
            position = Position3D(x=pos.x_val, y=pos.y_val, z=pos.z_val)
            
            # 提取姿态信息
            orientation = state.kinematics_estimated.orientation
            attitude = FlightAttitude(
                roll=orientation.x_val,
                pitch=orientation.y_val,
                yaw=orientation.z_val
            )
            
            # 提取速度信息
            vel = state.kinematics_estimated.linear_velocity
            velocity = VelocityVector(vx=vel.x_val, vy=vel.y_val, vz=vel.z_val)
            
            # 获取相机图像
            camera_image = await self._get_camera_image()
            
            # 获取卫星底图(模拟)
            satellite_image = await self._get_satellite_image(position)
            
            # 创建传感器数据对象
            sensor_data = SensorData(
                timestamp=timestamp,
                position=position,
                attitude=attitude,
                velocity=velocity,
                camera_image=camera_image,
                satellite_image=satellite_image
            )
            
            self.latest_sensor_data = sensor_data
            return sensor_data
            
        except Exception as e:
            self.logger.error(f"Failed to collect sensor data: {e}")
            return None
    
    async def _get_camera_image(self) -> Optional[np.ndarray]:
        """
        获取机腹向下相机图像
        
        Returns:
            Optional[np.ndarray]: 相机图像数组
        """
        try:
            # 获取相机图像
            responses = self.client.simGetImages([
                airsim.ImageRequest(
                    "bottom_center", 
                    airsim.ImageType.Scene, 
                    False, 
                    False
                )
            ], vehicle_name=self.vehicle_name)
            
            if responses:
                response = responses[0]
                
                # 转换为numpy数组
                img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                img_rgb = img1d.reshape(response.height, response.width, 3)
                
                return img_rgb
            
        except Exception as e:
            self.logger.error(f"Failed to get camera image: {e}")
        
        return None
    
    async def _get_satellite_image(self, position: Position3D) -> Optional[np.ndarray]:
        """
        获取当前位置的卫星底图(模拟实现)
        
        Args:
            position: 当前位置
            
        Returns:
            Optional[np.ndarray]: 卫星图像数组
        """
        try:
            # 这里是模拟实现，实际应用中需要调用地图服务API
            # 例如Google Maps API, Bing Maps API等
            
            # 创建模拟的卫星图像
            satellite_img = np.zeros((512, 512, 3), dtype=np.uint8)
            
            # 添加一些模拟的地形特征
            cv2.circle(satellite_img, (256, 256), 100, (0, 255, 0), -1)  # 绿色区域
            cv2.rectangle(satellite_img, (200, 200), (312, 312), (139, 69, 19), 2)  # 棕色边框
            
            # 添加位置标记
            cv2.circle(satellite_img, (256, 256), 5, (0, 0, 255), -1)  # 红色位置点
            
            return satellite_img
            
        except Exception as e:
            self.logger.error(f"Failed to get satellite image: {e}")
        
        return None
    
    def get_current_position(self) -> Optional[Position3D]:
        """
        获取当前位置
        
        Returns:
            Optional[Position3D]: 当前位置
        """
        if self.latest_sensor_data:
            return self.latest_sensor_data.position
        return None
    
    def get_mission_status(self) -> MissionStatus:
        """
        获取任务状态
        
        Returns:
            MissionStatus: 当前任务状态
        """
        return self.mission_status
    
    async def emergency_stop(self):
        """
        紧急停止
        """
        try:
            self.logger.warning("Emergency stop initiated")
            self.client.hoverAsync(vehicle_name=self.vehicle_name)
            self.mission_status = MissionStatus.PAUSED
        except Exception as e:
            self.logger.error(f"Emergency stop failed: {e}")
    
    async def shutdown(self):
        """
        关闭环境连接
        """
        try:
            if self.client and self.is_connected:
                # 降落
                self.client.landAsync(vehicle_name=self.vehicle_name).join()
                
                # 上锁
                self.client.armDisarm(False, self.vehicle_name)
                
                # 禁用API控制
                self.client.enableApiControl(False, self.vehicle_name)
                
                self.is_connected = False
                self.is_armed = False
                
                self.logger.info("AirSim environment shutdown completed")
                
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


class AirSimTrainingEnvironment:
    """
    AirSim训练环境包装类
    
    该类将AirSim环境包装为适合强化学习训练的接口
    """
    
    def __init__(self, config: AirSimConfig):
        """
        初始化训练环境
        
        Args:
            config: AirSim配置对象
        """
        self.airsim_env = AirSimEnvironment(config)
        self.config = config
        self.logger = setup_logger("AirSimTrainingEnvironment")
        
        # 训练相关参数
        self.episode_count = 0
        self.step_count = 0
        self.max_episode_steps = config.max_episode_steps
        
        # 初始化多元化奖励函数
        self.reward_function = MultiDimensionalRewardFunction()
        
        # 飞行状态管理
        self.current_task_phase = TaskPhase.NORMAL
        self.flight_status = FlightStatus(
            planned_trajectory=[],
            target_position=Position3D(x=0, y=0, z=-10),
            trajectory_tolerance=5.0,
            battery_level=1.0
        )
        
        # 状态历史记录
        self.previous_state: Optional[Dict[str, Any]] = None
        self.deviation_detected = False
        self.recovery_start_time: Optional[float] = None
        
    async def reset(self) -> Dict[str, Any]:
        """
        重置环境
        
        Returns:
            Dict[str, Any]: 初始观测状态
        """
        # 重置AirSim环境
        await self.airsim_env.shutdown()
        await self.airsim_env.initialize()
        
        # 重置计数器
        self.step_count = 0
        self.episode_count += 1
        
        # 重置奖励函数状态
        self.reward_function.reset()
        
        # 重置飞行状态
        self.current_task_phase = TaskPhase.NORMAL
        self.flight_status.battery_level = 1.0
        self.flight_status.is_collision = False
        self.flight_status.deviation_distance = 0.0
        self.flight_status.recovery_time = 0.0
        self.flight_status.recovery_success = True
        
        # 重置状态历史
        self.previous_state = None
        self.deviation_detected = False
        self.recovery_start_time = None
        
        # 生成随机飞行计划（示例）
        self._generate_random_flight_plan()
        
        # 获取初始状态
        sensor_data = await self.airsim_env._collect_sensor_data()
        initial_observation = self._create_observation(sensor_data)
        
        # 保存初始状态
        self.previous_state = initial_observation
        
        return initial_observation
    
    def _create_observation(self, sensor_data: SensorData) -> Dict[str, Any]:
        """
        创建观测状态
        
        Args:
            sensor_data: 传感器数据
            
        Returns:
            Dict[str, Any]: 观测状态字典
        """
        if sensor_data is None:
            return {}
        
        observation = {
            'position': [sensor_data.position.x, sensor_data.position.y, sensor_data.position.z],
            'attitude': [sensor_data.attitude.roll, sensor_data.attitude.pitch, sensor_data.attitude.yaw],
            'velocity': [sensor_data.velocity.vx, sensor_data.velocity.vy, sensor_data.velocity.vz],
            'timestamp': sensor_data.timestamp
        }
        
        # 添加图像数据(如果可用)
        if sensor_data.camera_image is not None:
            observation['camera_image'] = sensor_data.camera_image
        
        if sensor_data.satellite_image is not None:
            observation['satellite_image'] = sensor_data.satellite_image
        
        return observation
    
    async def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        执行一步动作
        
        Args:
            action: 动作字典
            
        Returns:
            Tuple: (observation, reward, done, info)
        """
        self.step_count += 1
        
        # 执行动作(这里需要根据具体的动作空间实现)
        await self._execute_action(action)
        
        # 获取新的观测
        sensor_data = await self.airsim_env._collect_sensor_data()
        observation = self._create_observation(sensor_data)
        
        # 更新飞行状态
        self._update_flight_status(observation, action)
        
        # 使用多元化奖励函数计算奖励
        reward_components = self.reward_function.calculate_reward(
            current_state=self.previous_state or observation,
            action=action,
            next_state=observation,
            flight_status=self.flight_status,
            task_phase=self.current_task_phase
        )
        
        reward = reward_components.total
        
        # 检查是否结束
        done = self._check_episode_termination(observation)
        
        # 附加信息
        info = {
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'mission_status': self.airsim_env.get_mission_status().value,
            'task_phase': self.current_task_phase.value,
            'reward_components': {
                'tracking': reward_components.tracking,
                'recovery': reward_components.recovery,
                'emergency': reward_components.emergency,
                'safety': reward_components.safety,
                'efficiency': reward_components.efficiency
            },
            'flight_status': {
                'battery_level': self.flight_status.battery_level,
                'deviation_distance': self.flight_status.deviation_distance,
                'is_collision': self.flight_status.is_collision
            }
        }
        
        # 更新前一状态
        self.previous_state = observation
        
        return observation, reward, done, info
    
    def _generate_random_flight_plan(self):
        """
        生成随机飞行计划
        """
        # 生成简单的矩形飞行路径
        waypoints = [
            Position3D(x=0, y=0, z=-10),
            Position3D(x=50, y=0, z=-10),
            Position3D(x=50, y=50, z=-10),
            Position3D(x=0, y=50, z=-10),
            Position3D(x=0, y=0, z=-10)
        ]
        
        self.flight_status.planned_trajectory = waypoints
        self.flight_status.target_position = waypoints[-1]
    
    async def _execute_action(self, action: Dict[str, Any]):
        """
        执行动作
        
        Args:
            action: 动作字典
        """
        # 这里需要根据具体的动作空间实现
        # 示例：假设动作包含目标位置
        if 'target_position' in action:
            target_pos = action['target_position']
            await self.airsim_env._fly_to_position(
                north=target_pos[0],
                east=target_pos[1], 
                down=target_pos[2],
                speed=5.0
            )
    
    def _update_flight_status(self, observation: Dict[str, Any], action: Dict[str, Any]):
        """
        更新飞行状态
        
        Args:
            observation: 当前观测
            action: 执行的动作
        """
        current_pos = Position3D(
            x=observation['position'][0],
            y=observation['position'][1],
            z=observation['position'][2]
        )
        
        # 更新电池电量（简化模型）
        energy_consumption = 0.001  # 每步消耗0.1%电量
        self.flight_status.battery_level = max(0.0, self.flight_status.battery_level - energy_consumption)
        
        # 检测碰撞（简化实现）
        altitude = -current_pos.z
        if altitude < 1.0:  # 接地检测
            self.flight_status.is_collision = True
        
        # 计算偏离距离
        if self.flight_status.planned_trajectory:
            min_distance = float('inf')
            for waypoint in self.flight_status.planned_trajectory:
                distance = self._calculate_distance_3d(current_pos, waypoint)
                min_distance = min(min_distance, distance)
            
            self.flight_status.deviation_distance = min_distance
            
            # 检测是否偏离轨迹
            if min_distance > self.flight_status.trajectory_tolerance and not self.deviation_detected:
                self.deviation_detected = True
                self.recovery_start_time = time.time()
                self.current_task_phase = TaskPhase.RECOVERY
            elif min_distance <= self.flight_status.trajectory_tolerance and self.deviation_detected:
                # 成功寻回
                self.deviation_detected = False
                if self.recovery_start_time:
                    self.flight_status.recovery_time = time.time() - self.recovery_start_time
                    self.flight_status.recovery_success = True
                self.current_task_phase = TaskPhase.NORMAL
        
        # 检测紧急状态
        if self.flight_status.battery_level < 0.2:
            self.current_task_phase = TaskPhase.EMERGENCY
    
    def _check_episode_termination(self, observation: Dict[str, Any]) -> bool:
        """
        检查episode是否应该结束
        
        Args:
            observation: 当前观测
            
        Returns:
            bool: 是否结束
        """
        # 基本终止条件
        if self.step_count >= self.max_episode_steps:
            return True
        
        # 碰撞终止
        if self.flight_status.is_collision:
            return True
        
        # 电池耗尽终止
        if self.flight_status.battery_level <= 0.0:
            return True
        
        # 高度异常终止
        altitude = -observation['position'][2]
        if altitude < 1.0 or altitude > 100.0:
            return True
        
        return False
    
    def _calculate_distance_3d(self, pos1: Position3D, pos2: Position3D) -> float:
        """
        计算3D距离
        
        Args:
            pos1: 位置1
            pos2: 位置2
            
        Returns:
            float: 距离
        """
        return math.sqrt(
            (pos1.x - pos2.x) ** 2 +
            (pos1.y - pos2.y) ** 2 +
            (pos1.z - pos2.z) ** 2
        )