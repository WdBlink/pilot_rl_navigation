#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强化学习无人机定位导航系统 - 配置管理模块

本模块提供统一的配置管理功能，支持YAML配置文件的加载、验证和管理。
使用Pydantic进行配置验证，确保配置参数的正确性和类型安全。

Author: wdblink
Date: 2024
"""

import os
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path
from pydantic import BaseModel, Field, validator
from loguru import logger


class RLAgentConfig(BaseModel):
    """强化学习智能体配置类
    
    定义强化学习算法的相关配置参数。
    """
    algorithm: str = Field(default="PPO", description="强化学习算法类型")
    learning_rate: float = Field(default=3e-4, gt=0, description="学习率")
    batch_size: int = Field(default=64, gt=0, description="批次大小")
    buffer_size: int = Field(default=100000, gt=0, description="经验回放缓冲区大小")
    gamma: float = Field(default=0.99, ge=0, le=1, description="折扣因子")
    tau: float = Field(default=0.005, ge=0, le=1, description="软更新系数")
    exploration_noise: float = Field(default=0.1, ge=0, description="探索噪声")
    policy_frequency: int = Field(default=2, gt=0, description="策略更新频率")
    
    # 网络架构配置
    hidden_layers: List[int] = Field(default=[256, 256], description="隐藏层神经元数量")
    activation: str = Field(default="relu", description="激活函数")
    
    # 训练配置
    total_timesteps: int = Field(default=1000000, gt=0, description="总训练步数")
    eval_frequency: int = Field(default=10000, gt=0, description="评估频率")
    save_frequency: int = Field(default=50000, gt=0, description="模型保存频率")
    
    @validator('algorithm')
    def validate_algorithm(cls, v):
        """验证算法类型"""
        allowed_algorithms = ['PPO', 'SAC', 'TD3', 'A2C']
        if v not in allowed_algorithms:
            raise ValueError(f"算法必须是以下之一: {allowed_algorithms}")
        return v


class EnvironmentConfig(BaseModel):
    """仿真环境配置类
    
    定义AirSim仿真环境的相关配置参数。
    """
    # AirSim连接配置
    airsim_ip: str = Field(default="127.0.0.1", description="AirSim服务器IP")
    airsim_port: int = Field(default=41451, gt=0, description="AirSim服务器端口")
    
    # 环境参数
    max_episode_steps: int = Field(default=1000, gt=0, description="最大轮次步数")
    step_time: float = Field(default=0.1, gt=0, description="仿真步长时间")
    
    # 传感器配置
    camera_name: str = Field(default="0", description="相机名称")
    image_width: int = Field(default=640, gt=0, description="图像宽度")
    image_height: int = Field(default=480, gt=0, description="图像高度")
    
    # 噪声配置
    position_noise_std: float = Field(default=0.1, ge=0, description="位置噪声标准差")
    attitude_noise_std: float = Field(default=0.01, ge=0, description="姿态噪声标准差")
    optical_failure_rate: float = Field(default=0.1, ge=0, le=1, description="光学定位失效率")
    
    # 风场配置
    wind_speed_range: List[float] = Field(default=[0.0, 10.0], description="风速范围")
    wind_direction_range: List[float] = Field(default=[0.0, 360.0], description="风向范围")
    
    # 管道配置
    pipeline_length: float = Field(default=1000.0, gt=0, description="管道长度")
    pipeline_diameter: float = Field(default=0.5, gt=0, description="管道直径")
    pipeline_height: float = Field(default=50.0, gt=0, description="管道高度")
    
    @validator('wind_speed_range')
    def validate_wind_speed_range(cls, v):
        """验证风速范围"""
        if len(v) != 2 or v[0] >= v[1]:
            raise ValueError("风速范围必须是[最小值, 最大值]格式")
        return v


class FusionConfig(BaseModel):
    """位置融合配置类
    
    定义多传感器位置融合算法的配置参数。
    """
    # 卡尔曼滤波器配置
    process_noise_std: float = Field(default=0.1, gt=0, description="过程噪声标准差")
    measurement_noise_std: float = Field(default=0.5, gt=0, description="测量噪声标准差")
    initial_covariance: float = Field(default=1.0, gt=0, description="初始协方差")
    
    # 融合权重配置
    default_ins_weight: float = Field(default=0.7, ge=0, le=1, description="默认惯导权重")
    default_optical_weight: float = Field(default=0.3, ge=0, le=1, description="默认光学权重")
    
    # 历史数据配置
    history_length: int = Field(default=100, gt=0, description="历史数据长度")
    fusion_history_length: int = Field(default=50, gt=0, description="融合历史长度")
    
    # 置信度配置
    min_confidence_threshold: float = Field(default=0.3, ge=0, le=1, description="最小置信度阈值")
    max_confidence_threshold: float = Field(default=0.9, ge=0, le=1, description="最大置信度阈值")
    
    @validator('default_optical_weight')
    def validate_weights_sum(cls, v, values):
        """验证权重和"""
        if 'default_ins_weight' in values:
            total = v + values['default_ins_weight']
            if abs(total - 1.0) > 1e-6:
                raise ValueError("惯导权重和光学权重之和必须等于1.0")
        return v


class RecoveryConfig(BaseModel):
    """自主恢复配置类
    
    定义位置丢失后的自主恢复策略配置参数。
    """
    # 位置丢失检测配置
    position_loss_threshold: float = Field(default=5.0, gt=0, description="位置丢失阈值")
    optical_timeout: float = Field(default=2.0, gt=0, description="光学定位超时时间")
    
    # 搜索策略配置
    spiral_turns: int = Field(default=3, gt=0, description="螺旋搜索圈数")
    points_per_turn: int = Field(default=8, gt=0, description="每圈搜索点数")
    search_radius: float = Field(default=20.0, gt=0, description="搜索半径")
    search_velocity: float = Field(default=3.0, gt=0, description="搜索速度")
    
    # 恢复策略配置
    max_recovery_time: float = Field(default=60.0, gt=0, description="最大恢复时间")
    recovery_success_threshold: float = Field(default=0.8, ge=0, le=1, description="恢复成功阈值")
    emergency_landing_height: float = Field(default=5.0, gt=0, description="紧急降落高度")
    
    # 安全配置
    min_flight_height: float = Field(default=10.0, gt=0, description="最小飞行高度")
    max_flight_height: float = Field(default=100.0, gt=0, description="最大飞行高度")
    safe_distance_margin: float = Field(default=5.0, gt=0, description="安全距离边界")


class LoggingConfig(BaseModel):
    """日志配置类
    
    定义系统日志的配置参数。
    """
    log_level: str = Field(default="INFO", description="日志级别")
    log_file: str = Field(default="logs/pilot_rl_navigation.log", description="日志文件路径")
    max_file_size: str = Field(default="10 MB", description="最大文件大小")
    backup_count: int = Field(default=5, gt=0, description="备份文件数量")
    log_format: str = Field(
        default="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        description="日志格式"
    )
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """验证日志级别"""
        allowed_levels = ['TRACE', 'DEBUG', 'INFO', 'SUCCESS', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in allowed_levels:
            raise ValueError(f"日志级别必须是以下之一: {allowed_levels}")
        return v.upper()


class SystemConfig(BaseModel):
    """系统总配置类
    
    整合所有子系统的配置参数。
    """
    # 项目信息
    project_name: str = Field(default="pilot_rl_navigation", description="项目名称")
    version: str = Field(default="1.0.0", description="版本号")
    author: str = Field(default="wdblink", description="作者")
    
    # 子系统配置
    rl_agent: RLAgentConfig = Field(default_factory=RLAgentConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    recovery: RecoveryConfig = Field(default_factory=RecoveryConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # 路径配置
    model_save_path: str = Field(default="models", description="模型保存路径")
    log_save_path: str = Field(default="logs", description="日志保存路径")
    data_save_path: str = Field(default="data", description="数据保存路径")
    
    # 设备配置
    device: str = Field(default="auto", description="计算设备")
    num_workers: int = Field(default=4, gt=0, description="工作进程数")
    
    @validator('device')
    def validate_device(cls, v):
        """验证设备配置"""
        allowed_devices = ['auto', 'cpu', 'cuda', 'mps']
        if v not in allowed_devices:
            raise ValueError(f"设备必须是以下之一: {allowed_devices}")
        return v


class ConfigManager:
    """配置管理器类
    
    提供配置文件的加载、保存、验证和管理功能。
    """
    
    def __init__(self, config_dir: str = "config"):
        """初始化配置管理器
        
        Args:
            config_dir: 配置文件目录路径
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self._config: Optional[SystemConfig] = None
    
    def load_config(self, config_file: str = "system_config.yaml") -> SystemConfig:
        """加载配置文件
        
        Args:
            config_file: 配置文件名
            
        Returns:
            系统配置对象
            
        Raises:
            FileNotFoundError: 配置文件不存在
            ValueError: 配置文件格式错误
        """
        config_path = self.config_dir / config_file
        
        if not config_path.exists():
            logger.warning(f"配置文件 {config_path} 不存在，使用默认配置")
            self._config = SystemConfig()
            self.save_config(config_file)
            return self._config
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            self._config = SystemConfig(**config_data)
            logger.info(f"成功加载配置文件: {config_path}")
            return self._config
            
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise ValueError(f"配置文件格式错误: {e}")
    
    def save_config(self, config_file: str = "system_config.yaml") -> None:
        """保存配置文件
        
        Args:
            config_file: 配置文件名
        """
        if self._config is None:
            self._config = SystemConfig()
        
        config_path = self.config_dir / config_file
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(
                    self._config.dict(),
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    indent=2
                )
            logger.info(f"配置文件已保存: {config_path}")
            
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
            raise
    
    def get_config(self) -> SystemConfig:
        """获取当前配置
        
        Returns:
            系统配置对象
        """
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def update_config(self, **kwargs) -> None:
        """更新配置参数
        
        Args:
            **kwargs: 要更新的配置参数
        """
        if self._config is None:
            self._config = SystemConfig()
        
        # 递归更新配置
        self._update_nested_dict(self._config.dict(), kwargs)
        
        # 重新验证配置
        self._config = SystemConfig(**self._config.dict())
        logger.info("配置已更新")
    
    def _update_nested_dict(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """递归更新嵌套字典
        
        Args:
            base_dict: 基础字典
            update_dict: 更新字典
        """
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._update_nested_dict(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def create_default_configs(self) -> None:
        """创建默认配置文件"""
        # 创建训练配置
        training_config = {
            'rl_agent': {
                'algorithm': 'PPO',
                'learning_rate': 3e-4,
                'total_timesteps': 1000000,
                'eval_frequency': 10000
            },
            'environment': {
                'max_episode_steps': 1000,
                'position_noise_std': 0.1
            }
        }
        
        training_path = self.config_dir / "training_config.yaml"
        with open(training_path, 'w', encoding='utf-8') as f:
            yaml.dump(training_config, f, default_flow_style=False, allow_unicode=True, indent=2)
        
        # 创建部署配置
        deployment_config = {
            'environment': {
                'airsim_ip': '127.0.0.1',
                'airsim_port': 41451
            },
            'fusion': {
                'default_ins_weight': 0.7,
                'default_optical_weight': 0.3
            },
            'recovery': {
                'position_loss_threshold': 5.0,
                'max_recovery_time': 60.0
            }
        }
        
        deployment_path = self.config_dir / "deployment_config.yaml"
        with open(deployment_path, 'w', encoding='utf-8') as f:
            yaml.dump(deployment_config, f, default_flow_style=False, allow_unicode=True, indent=2)
        
        # 创建环境配置
        environment_config = {
            'airsim': {
                'SettingsVersion': 1.2,
                'SimMode': 'Multirotor',
                'ClockSpeed': 1.0,
                'Vehicles': {
                    'Drone1': {
                        'VehicleType': 'SimpleFlight',
                        'X': 0, 'Y': 0, 'Z': -10
                    }
                }
            }
        }
        
        environment_path = self.config_dir / "environment_config.yaml"
        with open(environment_path, 'w', encoding='utf-8') as f:
            yaml.dump(environment_config, f, default_flow_style=False, allow_unicode=True, indent=2)
        
        logger.info("默认配置文件已创建")


# 全局配置管理器实例
config_manager = ConfigManager()


def get_config() -> SystemConfig:
    """获取全局配置
    
    Returns:
        系统配置对象
    """
    return config_manager.get_config()


def load_config(config_file: str = "system_config.yaml") -> SystemConfig:
    """加载配置文件
    
    Args:
        config_file: 配置文件名
        
    Returns:
        系统配置对象
    """
    return config_manager.load_config(config_file)