#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AirSim环境测试模块

该模块包含AirSim训练环境的单元测试和集成测试，用于验证环境配置和功能的正确性。

Author: wdblink
Date: 2024
"""

import pytest
import asyncio
import sys
import os
from pathlib import Path
import numpy as np
from unittest.mock import Mock, patch, AsyncMock

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.environment.airsim_env import (
    AirSimEnvironment,
    AirSimTrainingEnvironment,
    FlightPlan,
    Waypoint,
    MissionStatus,
    SensorData,
    Position3D,
    Attitude,
    Velocity3D
)
from src.utils.config import AirSimConfig


class TestAirSimConfig:
    """
    AirSim配置测试类
    
    测试AirSim配置类的创建和验证功能。
    """
    
    def test_default_config_creation(self):
        """
        测试默认配置创建
        """
        config = AirSimConfig()
        
        # 验证默认值
        assert config.connection.ip == "127.0.0.1"
        assert config.connection.port == 41451
        assert config.connection.vehicle_name == "Drone1"
        assert config.origin.latitude == 47.641468
        assert config.origin.longitude == -122.140165
        assert config.flight.max_altitude == 120.0
    
    def test_custom_config_creation(self):
        """
        测试自定义配置创建
        """
        config = AirSimConfig(
            connection={
                "ip": "192.168.1.100",
                "port": 41452,
                "vehicle_name": "TestDrone"
            },
            origin={
                "latitude": 40.7128,
                "longitude": -74.0060,
                "altitude": 10.0
            }
        )
        
        assert config.connection.ip == "192.168.1.100"
        assert config.connection.port == 41452
        assert config.connection.vehicle_name == "TestDrone"
        assert config.origin.latitude == 40.7128
        assert config.origin.longitude == -74.0060
    
    def test_config_validation(self):
        """
        测试配置验证
        """
        # 测试无效纬度
        with pytest.raises(ValueError):
            AirSimConfig(origin={"latitude": 91.0, "longitude": 0.0, "altitude": 0.0})
        
        # 测试无效经度
        with pytest.raises(ValueError):
            AirSimConfig(origin={"latitude": 0.0, "longitude": 181.0, "altitude": 0.0})
        
        # 测试无效端口
        with pytest.raises(ValueError):
            AirSimConfig(connection={"ip": "127.0.0.1", "port": 70000, "vehicle_name": "Drone1"})


class TestAirSimEnvironment:
    """
    AirSim环境测试类
    
    测试AirSim环境的核心功能。
    """
    
    @pytest.fixture
    def mock_config(self):
        """
        创建模拟配置
        """
        return AirSimConfig()
    
    @pytest.fixture
    def mock_airsim_client(self):
        """
        创建模拟AirSim客户端
        """
        with patch('airsim.MultirotorClient') as mock_client:
            # 配置模拟客户端的返回值
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            
            # 模拟连接确认
            mock_instance.confirmConnection.return_value = True
            
            # 模拟状态获取
            mock_state = Mock()
            mock_state.kinematics_estimated.position.x_val = 0.0
            mock_state.kinematics_estimated.position.y_val = 0.0
            mock_state.kinematics_estimated.position.z_val = -10.0
            mock_instance.getMultirotorState.return_value = mock_state
            
            # 模拟图像获取
            mock_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            mock_instance.simGetImage.return_value = mock_image.tobytes()
            
            yield mock_instance
    
    def test_environment_creation(self, mock_config):
        """
        测试环境创建
        """
        env = AirSimEnvironment(mock_config)
        
        assert env.config == mock_config
        assert env.client is None
        assert env.mission_status == MissionStatus.IDLE
        assert env.current_flight_plan is None
    
    @pytest.mark.asyncio
    async def test_environment_initialization(self, mock_config, mock_airsim_client):
        """
        测试环境初始化
        """
        with patch('airsim.MultirotorClient', return_value=mock_airsim_client):
            env = AirSimEnvironment(mock_config)
            
            # 测试成功初始化
            success = await env.initialize()
            assert success is True
            assert env.client is not None
            
            # 验证客户端方法被调用
            mock_airsim_client.confirmConnection.assert_called_once()
            mock_airsim_client.enableApiControl.assert_called_once_with(True)
            mock_airsim_client.armDisarm.assert_called_once_with(True)
    
    @pytest.mark.asyncio
    async def test_environment_initialization_failure(self, mock_config):
        """
        测试环境初始化失败
        """
        with patch('airsim.MultirotorClient') as mock_client:
            # 模拟连接失败
            mock_client.side_effect = Exception("Connection failed")
            
            env = AirSimEnvironment(mock_config)
            success = await env.initialize()
            
            assert success is False
            assert env.client is None
    
    def test_flight_plan_creation(self, mock_config):
        """
        测试飞行计划创建
        """
        env = AirSimEnvironment(mock_config)
        
        waypoints = [
            (47.642, -122.140, 30.0),
            (47.643, -122.139, 30.0),
            (47.642, -122.138, 30.0)
        ]
        
        flight_plan = env.create_flight_plan(waypoints, "test_mission")
        
        assert isinstance(flight_plan, FlightPlan)
        assert flight_plan.name == "test_mission"
        assert len(flight_plan.waypoints) == 3
        assert flight_plan.estimated_duration > 0
    
    def test_coordinate_conversion(self, mock_config):
        """
        测试坐标转换
        """
        env = AirSimEnvironment(mock_config)
        
        # 测试地理坐标到NED坐标转换
        lat, lon, alt = 47.642, -122.140, 30.0
        ned_x, ned_y, ned_z = env._geo_to_ned(lat, lon, alt)
        
        assert isinstance(ned_x, float)
        assert isinstance(ned_y, float)
        assert isinstance(ned_z, float)
        assert ned_z < 0  # NED坐标系中，向下为正
    
    @pytest.mark.asyncio
    async def test_sensor_data_collection(self, mock_config, mock_airsim_client):
        """
        测试传感器数据采集
        """
        with patch('airsim.MultirotorClient', return_value=mock_airsim_client):
            env = AirSimEnvironment(mock_config)
            await env.initialize()
            
            sensor_data = await env._collect_sensor_data()
            
            assert isinstance(sensor_data, SensorData)
            assert sensor_data.timestamp > 0
            assert isinstance(sensor_data.position, Position3D)
            assert isinstance(sensor_data.attitude, Attitude)
            assert isinstance(sensor_data.velocity, Velocity3D)
    
    @pytest.mark.asyncio
    async def test_emergency_stop(self, mock_config, mock_airsim_client):
        """
        测试紧急停止
        """
        with patch('airsim.MultirotorClient', return_value=mock_airsim_client):
            env = AirSimEnvironment(mock_config)
            await env.initialize()
            
            # 设置任务状态为执行中
            env.mission_status = MissionStatus.EXECUTING
            
            await env.emergency_stop()
            
            assert env.mission_status == MissionStatus.EMERGENCY_STOPPED
            mock_airsim_client.moveByVelocityAsync.assert_called_with(0, 0, 0, 1)


class TestAirSimTrainingEnvironment:
    """
    AirSim训练环境测试类
    
    测试强化学习训练环境的功能。
    """
    
    @pytest.fixture
    def mock_config(self):
        """
        创建模拟配置
        """
        return AirSimConfig()
    
    @pytest.fixture
    def mock_airsim_env(self):
        """
        创建模拟AirSim环境
        """
        mock_env = Mock(spec=AirSimEnvironment)
        mock_env.initialize = AsyncMock(return_value=True)
        mock_env.shutdown = AsyncMock()
        mock_env._collect_sensor_data = AsyncMock()
        mock_env.get_mission_status.return_value = MissionStatus.IDLE
        
        return mock_env
    
    def test_training_environment_creation(self, mock_config):
        """
        测试训练环境创建
        """
        with patch('src.environment.airsim_env.AirSimEnvironment') as mock_env_class:
            training_env = AirSimTrainingEnvironment(mock_config)
            
            assert training_env.config == mock_config
            assert training_env.step_count == 0
            assert training_env.episode_count == 0
            mock_env_class.assert_called_once_with(mock_config)
    
    @pytest.mark.asyncio
    async def test_environment_reset(self, mock_config, mock_airsim_env):
        """
        测试环境重置
        """
        with patch('src.environment.airsim_env.AirSimEnvironment', return_value=mock_airsim_env):
            training_env = AirSimTrainingEnvironment(mock_config)
            
            # 模拟传感器数据
            mock_sensor_data = SensorData(
                timestamp=1234567890.0,
                position=Position3D(0.0, 0.0, -10.0),
                attitude=Attitude(0.0, 0.0, 0.0),
                velocity=Velocity3D(0.0, 0.0, 0.0),
                camera_image=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                satellite_image=None,
                imu_data=None,
                gps_data=None
            )
            mock_airsim_env._collect_sensor_data.return_value = mock_sensor_data
            
            observation = await training_env.reset()
            
            assert isinstance(observation, dict)
            assert 'camera_image' in observation
            assert 'position' in observation
            assert 'attitude' in observation
            assert 'velocity' in observation
            assert training_env.step_count == 0
            assert training_env.episode_count == 1
    
    @pytest.mark.asyncio
    async def test_environment_step(self, mock_config, mock_airsim_env):
        """
        测试环境步进
        """
        with patch('src.environment.airsim_env.AirSimEnvironment', return_value=mock_airsim_env):
            training_env = AirSimTrainingEnvironment(mock_config)
            
            # 模拟传感器数据
            mock_sensor_data = SensorData(
                timestamp=1234567890.0,
                position=Position3D(1.0, 1.0, -10.0),
                attitude=Attitude(0.1, 0.1, 0.1),
                velocity=Velocity3D(1.0, 0.0, 0.0),
                camera_image=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                satellite_image=None,
                imu_data=None,
                gps_data=None
            )
            mock_airsim_env._collect_sensor_data.return_value = mock_sensor_data
            
            action = {
                'thrust': 0.5,
                'roll': 0.1,
                'pitch': 0.0,
                'yaw_rate': 0.0
            }
            
            observation, reward, done, info = await training_env.step(action)
            
            assert isinstance(observation, dict)
            assert isinstance(reward, float)
            assert isinstance(done, bool)
            assert isinstance(info, dict)
            assert training_env.step_count == 1
    
    def test_reward_calculation(self, mock_config):
        """
        测试奖励计算
        """
        with patch('src.environment.airsim_env.AirSimEnvironment'):
            training_env = AirSimTrainingEnvironment(mock_config)
            
            # 测试正常飞行奖励
            sensor_data = SensorData(
                timestamp=1234567890.0,
                position=Position3D(0.0, 0.0, -10.0),
                attitude=Attitude(0.0, 0.0, 0.0),
                velocity=Velocity3D(2.0, 0.0, 0.0),
                camera_image=None,
                satellite_image=None,
                imu_data=None,
                gps_data=None
            )
            
            reward = training_env._calculate_reward(sensor_data, collision=False)
            assert isinstance(reward, float)
            
            # 测试碰撞惩罚
            collision_reward = training_env._calculate_reward(sensor_data, collision=True)
            assert collision_reward < reward
    
    def test_done_condition_check(self, mock_config):
        """
        测试结束条件检查
        """
        with patch('src.environment.airsim_env.AirSimEnvironment'):
            training_env = AirSimTrainingEnvironment(mock_config)
            
            # 测试正常情况
            sensor_data = SensorData(
                timestamp=1234567890.0,
                position=Position3D(0.0, 0.0, -10.0),
                attitude=Attitude(0.0, 0.0, 0.0),
                velocity=Velocity3D(2.0, 0.0, 0.0),
                camera_image=None,
                satellite_image=None,
                imu_data=None,
                gps_data=None
            )
            
            done = training_env._check_done_condition(sensor_data, collision=False)
            assert isinstance(done, bool)
            
            # 测试碰撞情况
            collision_done = training_env._check_done_condition(sensor_data, collision=True)
            assert collision_done is True
            
            # 测试超时情况
            training_env.step_count = training_env.config.training.max_steps_per_episode + 1
            timeout_done = training_env._check_done_condition(sensor_data, collision=False)
            assert timeout_done is True


class TestIntegration:
    """
    集成测试类
    
    测试AirSim环境的集成功能。
    """
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_training_episode(self):
        """
        测试完整的训练轮次
        
        注意：这是一个集成测试，需要实际的AirSim环境运行
        """
        # 跳过集成测试，除非明确指定
        pytest.skip("集成测试需要实际的AirSim环境")
        
        config = AirSimConfig()
        training_env = AirSimTrainingEnvironment(config)
        
        try:
            # 初始化环境
            await training_env.airsim_env.initialize()
            
            # 重置环境
            observation = await training_env.reset()
            assert observation is not None
            
            # 执行几个步骤
            for step in range(5):
                action = {
                    'thrust': 0.5,
                    'roll': 0.0,
                    'pitch': 0.0,
                    'yaw_rate': 0.0
                }
                
                observation, reward, done, info = await training_env.step(action)
                
                assert observation is not None
                assert isinstance(reward, float)
                assert isinstance(done, bool)
                assert isinstance(info, dict)
                
                if done:
                    break
        
        finally:
            # 清理资源
            await training_env.airsim_env.shutdown()


if __name__ == "__main__":
    # 运行测试
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "not integration"  # 默认跳过集成测试
    ])