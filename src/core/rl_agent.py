#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强化学习无人机定位导航系统 - 强化学习智能体模块

本模块实现了分层强化学习智能体，包括高层任务决策、中层传感器融合权重分配
和低层控制参数优化。支持PPO、SAC、TD3等多种强化学习算法。

Author: wdblink
Date: 2024
"""

import os
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
from stable_baselines3 import PPO, SAC, TD3, A2C
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium import spaces
import gymnasium as gym

from ..utils.data_types import SystemState, RLAction, ControlMode, TrainingMetrics
from ..utils.logger import get_logger, performance_monitor, log_function_call
from ..utils.config import RLAgentConfig


class CustomFeaturesExtractor(BaseFeaturesExtractor):
    """自定义特征提取器类
    
    为无人机导航任务设计的专用特征提取网络，能够有效处理多模态传感器数据。
    """
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        """初始化特征提取器
        
        Args:
            observation_space: 观测空间
            features_dim: 特征维度
        """
        super().__init__(observation_space, features_dim)
        
        # 计算输入维度
        n_input_channels = observation_space.shape[0]
        
        # 位置和姿态特征提取网络
        self.position_net = nn.Sequential(
            nn.Linear(6, 64),  # 位置(3) + 姿态(3)
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # 传感器质量特征提取网络
        self.sensor_net = nn.Sequential(
            nn.Linear(4, 32),  # 光学位置(3) + 匹配得分(1)
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # 环境状态特征提取网络
        self.environment_net = nn.Sequential(
            nn.Linear(8, 32),  # 速度(3) + 管道偏离(1) + 电池(1) + 风况(2) + 历史误差(1)
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # 特征融合网络
        self.fusion_net = nn.Sequential(
            nn.Linear(32 + 16 + 16, features_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            observations: 观测数据
            
        Returns:
            提取的特征
        """
        # 分割观测数据
        position_attitude = observations[:, :6]  # 位置 + 姿态
        sensor_quality = observations[:, 6:10]   # 光学位置 + 匹配得分
        environment = observations[:, 10:]       # 其他环境状态
        
        # 特征提取
        position_features = self.position_net(position_attitude)
        sensor_features = self.sensor_net(sensor_quality)
        environment_features = self.environment_net(environment)
        
        # 特征融合
        combined_features = torch.cat([
            position_features, sensor_features, environment_features
        ], dim=1)
        
        return self.fusion_net(combined_features)


class BaseRLAgent(ABC):
    """强化学习智能体基类
    
    定义了强化学习智能体的基本接口和通用功能。
    """
    
    def __init__(self, config: RLAgentConfig):
        """初始化基础智能体
        
        Args:
            config: 智能体配置
        """
        self.config = config
        self.logger = get_logger("rl_agent")
        self.training_metrics = TrainingMetrics()
        
    @abstractmethod
    def predict(self, observation: np.ndarray) -> Tuple[np.ndarray, Optional[Dict]]:
        """预测动作
        
        Args:
            observation: 观测数据
            
        Returns:
            动作和额外信息
        """
        pass
    
    @abstractmethod
    def train(self, total_timesteps: int) -> None:
        """训练智能体
        
        Args:
            total_timesteps: 总训练步数
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """保存模型
        
        Args:
            path: 保存路径
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """加载模型
        
        Args:
            path: 模型路径
        """
        pass


class RLAgent(BaseRLAgent):
    """强化学习智能体主类
    
    实现了完整的强化学习智能体，支持多种算法和训练策略。
    """
    
    def __init__(self, 
                 algorithm: str = "ppo",
                 env: Optional[gym.Env] = None,
                 config: Optional[RLAgentConfig] = None):
        """初始化强化学习智能体
        
        Args:
            algorithm: 算法类型 ("ppo", "sac", "td3", "a2c")
            env: 训练环境
            config: 智能体配置
        """
        if config is None:
            config = RLAgentConfig()
        
        super().__init__(config)
        
        self.algorithm = algorithm.lower()
        self.env = env
        self.model = None
        self.is_trained = False
        
        # 支持的算法
        self.supported_algorithms = {
            "ppo": PPO,
            "sac": SAC,
            "td3": TD3,
            "a2c": A2C
        }
        
        if self.algorithm not in self.supported_algorithms:
            raise ValueError(f"不支持的算法: {algorithm}. 支持的算法: {list(self.supported_algorithms.keys())}")
        
        # 初始化模型
        if env is not None:
            self._initialize_model()
    
    def _initialize_model(self) -> None:
        """初始化强化学习模型"""
        try:
            algorithm_class = self.supported_algorithms[self.algorithm]
            
            # 策略配置
            policy_kwargs = {
                "features_extractor_class": CustomFeaturesExtractor,
                "features_extractor_kwargs": {"features_dim": self.config.features_dim},
                "net_arch": self.config.network_architecture
            }
            
            # 模型参数
            model_kwargs = {
                "policy": "MlpPolicy",
                "env": self.env,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "gamma": self.config.gamma,
                "policy_kwargs": policy_kwargs,
                "verbose": 1,
                "tensorboard_log": self.config.tensorboard_log_dir
            }
            
            # 算法特定参数
            if self.algorithm == "ppo":
                model_kwargs.update({
                    "n_steps": self.config.n_steps,
                    "n_epochs": self.config.n_epochs,
                    "clip_range": self.config.clip_range
                })
            elif self.algorithm in ["sac", "td3"]:
                model_kwargs.update({
                    "buffer_size": self.config.buffer_size,
                    "learning_starts": self.config.learning_starts,
                    "train_freq": self.config.train_freq
                })
            
            self.model = algorithm_class(**model_kwargs)
            self.logger.info(f"成功初始化{self.algorithm.upper()}智能体")
            
        except Exception as e:
            self.logger.error(f"初始化模型失败: {e}")
            raise
    
    @performance_monitor
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[Dict]]:
        """预测动作
        
        Args:
            observation: 观测数据
            deterministic: 是否使用确定性策略
            
        Returns:
            动作和额外信息
        """
        if self.model is None:
            raise RuntimeError("模型未初始化，请先调用_initialize_model()")
        
        try:
            action, state = self.model.predict(observation, deterministic=deterministic)
            return action, {"state": state}
        except Exception as e:
            self.logger.error(f"预测动作失败: {e}")
            raise
    
    @log_function_call
    def train(self, total_timesteps: int, callback=None) -> None:
        """训练智能体
        
        Args:
            total_timesteps: 总训练步数
            callback: 训练回调函数
        """
        if self.model is None:
            raise RuntimeError("模型未初始化")
        
        try:
            self.logger.info(f"开始训练{self.algorithm.upper()}智能体，总步数: {total_timesteps}")
            
            # 训练模型
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                progress_bar=True
            )
            
            self.is_trained = True
            self.logger.info("训练完成")
            
        except Exception as e:
            self.logger.error(f"训练失败: {e}")
            raise
    
    def save(self, path: str) -> None:
        """保存模型
        
        Args:
            path: 保存路径
        """
        if self.model is None:
            raise RuntimeError("模型未初始化")
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # 保存模型
            self.model.save(path)
            
            # 保存配置
            config_path = path + "_config.json"
            self.config.save(config_path)
            
            self.logger.info(f"模型已保存到: {path}")
            
        except Exception as e:
            self.logger.error(f"保存模型失败: {e}")
            raise
    
    @classmethod
    def load(cls, path: str, env: Optional[gym.Env] = None) -> 'RLAgent':
        """加载模型
        
        Args:
            path: 模型路径
            env: 环境对象
            
        Returns:
            加载的智能体实例
        """
        try:
            # 加载配置
            config_path = path + "_config.json"
            if os.path.exists(config_path):
                config = RLAgentConfig.load(config_path)
            else:
                config = RLAgentConfig()
            
            # 创建智能体实例
            agent = cls(config=config)
            
            # 检测算法类型
            for alg_name, alg_class in agent.supported_algorithms.items():
                try:
                    model = alg_class.load(path, env=env)
                    agent.model = model
                    agent.algorithm = alg_name
                    agent.env = env
                    agent.is_trained = True
                    break
                except:
                    continue
            
            if agent.model is None:
                raise RuntimeError("无法加载模型，请检查文件路径和格式")
            
            agent.logger.info(f"成功加载{agent.algorithm.upper()}模型: {path}")
            return agent
            
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {e}")
    
    def evaluate(self, env: gym.Env, n_episodes: int = 10) -> Dict[str, float]:
        """评估智能体性能
        
        Args:
            env: 评估环境
            n_episodes: 评估回合数
            
        Returns:
            评估结果
        """
        if self.model is None:
            raise RuntimeError("模型未初始化")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action, _ = self.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        results = {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "std_length": np.std(episode_lengths),
            "min_reward": np.min(episode_rewards),
            "max_reward": np.max(episode_rewards)
        }
        
        self.logger.info(f"评估结果: {results}")
        return results
    
    def get_action_distribution(self, observation: np.ndarray) -> Dict[str, Any]:
        """获取动作分布信息
        
        Args:
            observation: 观测数据
            
        Returns:
            动作分布信息
        """
        if self.model is None:
            raise RuntimeError("模型未初始化")
        
        try:
            # 获取策略网络的输出
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            
            with torch.no_grad():
                if hasattr(self.model.policy, 'get_distribution'):
                    distribution = self.model.policy.get_distribution(obs_tensor)
                    return {
                        "mean": distribution.distribution.mean.cpu().numpy(),
                        "std": distribution.distribution.stddev.cpu().numpy(),
                        "entropy": distribution.entropy().cpu().numpy()
                    }
                else:
                    # 对于确定性策略
                    action = self.model.policy(obs_tensor)
                    return {
                        "action": action.cpu().numpy(),
                        "deterministic": True
                    }
        except Exception as e:
            self.logger.error(f"获取动作分布失败: {e}")
            return {}
    
    def update_config(self, new_config: RLAgentConfig) -> None:
        """更新智能体配置
        
        Args:
            new_config: 新的配置
        """
        self.config = new_config
        self.logger.info("智能体配置已更新")
        
        # 如果模型已初始化，需要重新初始化
        if self.model is not None and self.env is not None:
            self.logger.warning("配置更新后需要重新初始化模型")
            self._initialize_model()
    
    @property
    def info(self) -> Dict[str, Any]:
        """获取智能体信息
        
        Returns:
            智能体信息字典
        """
        return {
            "algorithm": self.algorithm,
            "is_trained": self.is_trained,
            "config": self.config.__dict__,
            "model_initialized": self.model is not None,
            "env_attached": self.env is not None
        }