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
    
    定义了所有RL智能体必须实现的接口方法，提供统一的训练和推理接口。
    """
    
    def __init__(self, config: RLAgentConfig):
        """初始化基础智能体
        
        Args:
            config: 智能体配置
        """
        self.config = config
        self.logger = get_logger("rl_agent")
        self.model = None
        self.is_trained = False
    
    @abstractmethod
    def predict(self, state: SystemState, deterministic: bool = True) -> RLAction:
        """根据当前状态预测动作
        
        Args:
            state: 当前系统状态
            deterministic: 是否使用确定性策略
            
        Returns:
            预测的动作
        """
        pass
    
    @abstractmethod
    def train(self, env: gym.Env, total_timesteps: int, **kwargs) -> None:
        """训练智能体
        
        Args:
            env: 训练环境
            total_timesteps: 总训练步数
            **kwargs: 其他训练参数
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
    
    def _state_to_observation(self, state: SystemState) -> np.ndarray:
        """将系统状态转换为观测向量
        
        Args:
            state: 系统状态
            
        Returns:
            观测向量
        """
        return state.to_observation_vector()
    
    def _action_vector_to_rl_action(self, action_vector: np.ndarray) -> RLAction:
        """将动作向量转换为RLAction对象
        
        Args:
            action_vector: 动作向量
            
        Returns:
            RLAction对象
        """
        # 解析动作向量
        fusion_weights = tuple(action_vector[:3])
        update_decision = bool(action_vector[3] > 0.5)
        
        # 控制模式（one-hot解码）
        mode_probs = action_vector[4:7]
        mode_idx = np.argmax(mode_probs)
        control_modes = [ControlMode.NORMAL, ControlMode.RECOVERY, ControlMode.EMERGENCY]
        control_mode = control_modes[mode_idx]
        
        pipeline_adjustment = tuple(action_vector[7:9])
        confidence_threshold = action_vector[9]
        
        return RLAction(
            fusion_weights=fusion_weights,
            update_decision=update_decision,
            control_mode=control_mode,
            pipeline_adjustment=pipeline_adjustment,
            confidence_threshold=confidence_threshold
        )


class HierarchicalRLAgent(BaseRLAgent):
    """分层强化学习智能体类
    
    实现三层决策架构：
    - 高层：任务级决策（模式切换）
    - 中层：传感器融合权重分配
    - 低层：具体控制动作执行
    """
    
    def __init__(self, config: RLAgentConfig, observation_space: gym.Space, action_space: gym.Space):
        """初始化分层智能体
        
        Args:
            config: 智能体配置
            observation_space: 观测空间
            action_space: 动作空间
        """
        super().__init__(config)
        
        self.observation_space = observation_space
        self.action_space = action_space
        
        # 分层智能体
        self.high_level_agent = None    # 高层策略网络
        self.mid_level_agent = None     # 中层策略网络
        self.low_level_agent = None     # 低层策略网络
        
        # 训练历史
        self.training_metrics: List[TrainingMetrics] = []
        
        self._initialize_agents()
    
    def _initialize_agents(self) -> None:
        """初始化各层智能体"""
        # 创建自定义策略
        policy_kwargs = {
            'features_extractor_class': CustomFeaturesExtractor,
            'features_extractor_kwargs': {'features_dim': 256},
            'net_arch': self.config.hidden_layers
        }
        
        # 根据配置选择算法
        if self.config.algorithm == "PPO":
            self.model = PPO(
                "MlpPolicy",
                env=None,  # 将在训练时设置
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                gamma=self.config.gamma,
                policy_kwargs=policy_kwargs,
                verbose=1,
                device="auto"
            )
        elif self.config.algorithm == "SAC":
            self.model = SAC(
                "MlpPolicy",
                env=None,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                gamma=self.config.gamma,
                tau=self.config.tau,
                policy_kwargs=policy_kwargs,
                verbose=1,
                device="auto"
            )
        elif self.config.algorithm == "TD3":
            self.model = TD3(
                "MlpPolicy",
                env=None,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                gamma=self.config.gamma,
                tau=self.config.tau,
                policy_kwargs=policy_kwargs,
                verbose=1,
                device="auto"
            )
        else:
            raise ValueError(f"不支持的算法: {self.config.algorithm}")
        
        self.logger.log_event("info", "agent_initialized", algorithm=self.config.algorithm)
    
    @performance_monitor
    @log_function_call("rl_agent")
    def predict(self, state: SystemState, deterministic: bool = True) -> RLAction:
        """分层预测动作
        
        Args:
            state: 当前系统状态
            deterministic: 是否使用确定性策略
            
        Returns:
            预测的动作
        """
        if self.model is None:
            raise RuntimeError("模型未初始化")
        
        # 转换状态为观测向量
        observation = self._state_to_observation(state)
        
        # 模型预测
        action_vector, _ = self.model.predict(observation, deterministic=deterministic)
        
        # 转换为RLAction
        rl_action = self._action_vector_to_rl_action(action_vector)
        
        self.logger.log_event(
            "debug",
            "action_predicted",
            control_mode=rl_action.control_mode.value,
            fusion_weights=rl_action.fusion_weights,
            confidence_threshold=rl_action.confidence_threshold
        )
        
        return rl_action
    
    def _high_level_decision(self, state: SystemState) -> ControlMode:
        """高层决策：确定控制模式
        
        Args:
            state: 系统状态
            
        Returns:
            控制模式
        """
        # 基于规则的高层决策逻辑
        if state.battery_level < 0.2:
            return ControlMode.EMERGENCY
        
        if (state.optical_position is None or 
            state.optical_position.match_score < 0.3 or
            state.pipeline_deviation > 10.0):
            return ControlMode.RECOVERY
        
        return ControlMode.NORMAL
    
    def _mid_level_decision(self, state: SystemState, control_mode: ControlMode) -> Tuple[float, float, float]:
        """中层决策：传感器融合权重
        
        Args:
            state: 系统状态
            control_mode: 控制模式
            
        Returns:
            融合权重 (lambda_ins, alpha_opt, bias)
        """
        if control_mode == ControlMode.EMERGENCY:
            # 紧急模式：主要依赖惯导
            return (0.9, 0.1, 0.0)
        elif control_mode == ControlMode.RECOVERY:
            # 恢复模式：平衡权重
            return (0.6, 0.4, 0.0)
        else:
            # 正常模式：根据光学质量动态调整
            if state.optical_position and state.optical_position.match_score > 0.7:
                return (0.3, 0.7, 0.0)
            else:
                return (0.7, 0.3, 0.0)
    
    def _low_level_decision(self, state: SystemState, control_mode: ControlMode) -> Dict[str, Any]:
        """低层决策：具体控制参数
        
        Args:
            state: 系统状态
            control_mode: 控制模式
            
        Returns:
            控制参数字典
        """
        # 基于当前状态和模式确定控制参数
        if control_mode == ControlMode.EMERGENCY:
            return {
                'update_decision': False,
                'pipeline_adjustment': (0.0, 0.0),
                'confidence_threshold': 0.9
            }
        elif control_mode == ControlMode.RECOVERY:
            return {
                'update_decision': True,
                'pipeline_adjustment': (-state.pipeline_deviation * 0.1, 0.0),
                'confidence_threshold': 0.5
            }
        else:
            return {
                'update_decision': True,
                'pipeline_adjustment': (-state.pipeline_deviation * 0.05, 0.0),
                'confidence_threshold': 0.7
            }
    
    @log_function_call("rl_agent")
    def train(self, env: gym.Env, total_timesteps: int, **kwargs) -> None:
        """训练智能体
        
        Args:
            env: 训练环境
            total_timesteps: 总训练步数
            **kwargs: 其他训练参数
        """
        if self.model is None:
            raise RuntimeError("模型未初始化")
        
        # 设置环境
        self.model.set_env(env)
        
        # 创建训练回调
        callback = TrainingCallback(self)
        
        self.logger.log_event(
            "info",
            "training_started",
            total_timesteps=total_timesteps,
            algorithm=self.config.algorithm
        )
        
        try:
            # 开始训练
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                **kwargs
            )
            
            self.is_trained = True
            self.logger.log_event("info", "training_completed")
            
        except Exception as e:
            self.logger.log_error(e, {"phase": "training"})
            raise
    
    def save(self, path: str) -> None:
        """保存模型
        
        Args:
            path: 保存路径
        """
        if self.model is None:
            raise RuntimeError("模型未初始化")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存模型
        self.model.save(path)
        
        # 保存训练指标
        metrics_path = path + "_metrics.npy"
        if self.training_metrics:
            metrics_data = [{
                'episode': m.episode,
                'total_reward': m.total_reward,
                'episode_length': m.episode_length,
                'position_error': m.position_error,
                'fusion_accuracy': m.fusion_accuracy,
                'recovery_success_rate': m.recovery_success_rate,
                'timestamp': m.timestamp
            } for m in self.training_metrics]
            
            np.save(metrics_path, metrics_data)
        
        self.logger.log_event("info", "model_saved", path=path)
    
    def load(self, path: str) -> None:
        """加载模型
        
        Args:
            path: 模型路径
        """
        if not os.path.exists(path + ".zip"):
            raise FileNotFoundError(f"模型文件不存在: {path}")
        
        # 根据算法类型加载模型
        if self.config.algorithm == "PPO":
            self.model = PPO.load(path)
        elif self.config.algorithm == "SAC":
            self.model = SAC.load(path)
        elif self.config.algorithm == "TD3":
            self.model = TD3.load(path)
        else:
            raise ValueError(f"不支持的算法: {self.config.algorithm}")
        
        # 加载训练指标
        metrics_path = path + "_metrics.npy"
        if os.path.exists(metrics_path):
            metrics_data = np.load(metrics_path, allow_pickle=True)
            self.training_metrics = [
                TrainingMetrics(**m) for m in metrics_data
            ]
        
        self.is_trained = True
        self.logger.log_event("info", "model_loaded", path=path)
    
    def get_training_metrics(self) -> List[TrainingMetrics]:
        """获取训练指标
        
        Returns:
            训练指标列表
        """
        return self.training_metrics.copy()
    
    def evaluate(self, env: gym.Env, n_episodes: int = 10) -> Dict[str, float]:
        """评估智能体性能
        
        Args:
            env: 评估环境
            n_episodes: 评估轮次数
            
        Returns:
            评估结果字典
        """
        if self.model is None:
            raise RuntimeError("模型未初始化")
        
        total_rewards = []
        episode_lengths = []
        position_errors = []
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_position_errors = []
            
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                
                if 'position_error' in info:
                    episode_position_errors.append(info['position_error'])
            
            total_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            if episode_position_errors:
                position_errors.append(np.mean(episode_position_errors))
        
        evaluation_results = {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'mean_position_error': np.mean(position_errors) if position_errors else 0.0
        }
        
        self.logger.log_event(
            "info",
            "evaluation_completed",
            **evaluation_results
        )
        
        return evaluation_results


class TrainingCallback(BaseCallback):
    """训练回调类
    
    用于监控训练过程并记录训练指标。
    """
    
    def __init__(self, agent: HierarchicalRLAgent, verbose: int = 0):
        """初始化训练回调
        
        Args:
            agent: 智能体实例
            verbose: 详细程度
        """
        super().__init__(verbose)
        self.agent = agent
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        """每步回调
        
        Returns:
            是否继续训练
        """
        # 检查是否完成一个episode
        if self.locals.get('dones', [False])[0]:
            self.episode_count += 1
            
            # 记录episode指标
            if 'episode' in self.locals.get('infos', [{}])[0]:
                episode_info = self.locals['infos'][0]['episode']
                
                metrics = TrainingMetrics(
                    episode=self.episode_count,
                    total_reward=episode_info['r'],
                    episode_length=episode_info['l'],
                    position_error=episode_info.get('position_error', 0.0),
                    fusion_accuracy=episode_info.get('fusion_accuracy', 0.0),
                    recovery_success_rate=episode_info.get('recovery_success_rate', 0.0)
                )
                
                self.agent.training_metrics.append(metrics)
                
                # 定期记录训练进度
                if self.episode_count % 100 == 0:
                    self.agent.logger.log_event(
                        "info",
                        "training_progress",
                        episode=self.episode_count,
                        total_timesteps=self.num_timesteps,
                        mean_reward=np.mean([m.total_reward for m in self.agent.training_metrics[-100:]]),
                        mean_episode_length=np.mean([m.episode_length for m in self.agent.training_metrics[-100:]])
                    )
        
        return True