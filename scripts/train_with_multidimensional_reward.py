#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用多元化奖励函数的强化学习训练脚本

该脚本展示了如何使用新的多元化奖励函数进行无人机导航的强化学习训练。
包含了循迹能力、寻回定位能力、紧急决策能力等多个维度的奖励评估。

Author: AI Assistant
Date: 2024
"""

import os
import sys
import asyncio
import numpy as np
import yaml
from datetime import datetime
from typing import Dict, Any, Optional

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.airsim_env import AirSimTrainingEnvironment
from src.utils.config import AirSimConfig
from src.utils.logger import setup_logger
from src.core.reward_function import TaskPhase, DecisionType

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
except ImportError:
    print("Warning: stable_baselines3 not found. Please install it for training.")
    PPO = None


class MultiDimensionalRewardTrainer:
    """
    多元化奖励函数训练器
    """
    
    def __init__(self, config_path: str):
        """
        初始化训练器
        
        Args:
            config_path: 配置文件路径
        """
        self.logger = setup_logger("MultiDimensionalRewardTrainer")
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 创建AirSim配置
        self.airsim_config = AirSimConfig(
            host=self.config.get('airsim', {}).get('host', '127.0.0.1'),
            port=self.config.get('airsim', {}).get('port', 41451),
            max_episode_steps=self.config.get('training', {}).get('max_episode_steps', 1000)
        )
        
        # 训练参数
        self.training_config = self.config.get('training', {})
        self.model_save_path = self.config.get('model_save_path', './models')
        
        # 创建保存目录
        os.makedirs(self.model_save_path, exist_ok=True)
        
        self.logger.info(f"训练器初始化完成，配置文件: {config_path}")
    
    async def create_environment(self) -> AirSimTrainingEnvironment:
        """
        创建训练环境
        
        Returns:
            AirSimTrainingEnvironment: 训练环境实例
        """
        env = AirSimTrainingEnvironment(self.airsim_config)
        await env.airsim_env.initialize()
        return env
    
    def create_model(self, env) -> Optional[PPO]:
        """
        创建PPO模型
        
        Args:
            env: 训练环境
            
        Returns:
            PPO: PPO模型实例
        """
        if PPO is None:
            self.logger.error("stable_baselines3未安装，无法创建模型")
            return None
        
        model_config = self.training_config.get('model', {})
        
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=model_config.get('learning_rate', 3e-4),
            n_steps=model_config.get('n_steps', 2048),
            batch_size=model_config.get('batch_size', 64),
            n_epochs=model_config.get('n_epochs', 10),
            gamma=model_config.get('gamma', 0.99),
            gae_lambda=model_config.get('gae_lambda', 0.95),
            clip_range=model_config.get('clip_range', 0.2),
            ent_coef=model_config.get('ent_coef', 0.0),
            vf_coef=model_config.get('vf_coef', 0.5),
            max_grad_norm=model_config.get('max_grad_norm', 0.5),
            verbose=1,
            tensorboard_log=f"{self.model_save_path}/tensorboard/"
        )
        
        self.logger.info("PPO模型创建完成")
        return model
    
    def create_callbacks(self):
        """
        创建训练回调函数
        
        Returns:
            List: 回调函数列表
        """
        callbacks = []
        
        # 检查点保存回调
        checkpoint_callback = CheckpointCallback(
            save_freq=self.training_config.get('save_freq', 10000),
            save_path=f"{self.model_save_path}/checkpoints/",
            name_prefix="multidimensional_reward_model"
        )
        callbacks.append(checkpoint_callback)
        
        self.logger.info("训练回调函数创建完成")
        return callbacks
    
    async def train(self):
        """
        开始训练
        """
        self.logger.info("开始多元化奖励函数训练")
        
        try:
            # 创建环境
            env = await self.create_environment()
            
            # 包装环境用于监控
            env = Monitor(env)
            
            # 创建模型
            model = self.create_model(env)
            if model is None:
                return
            
            # 创建回调函数
            callbacks = self.create_callbacks()
            
            # 开始训练
            total_timesteps = self.training_config.get('total_timesteps', 100000)
            
            self.logger.info(f"开始训练，总时间步数: {total_timesteps}")
            
            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                progress_bar=True
            )
            
            # 保存最终模型
            final_model_path = f"{self.model_save_path}/final_multidimensional_reward_model"
            model.save(final_model_path)
            
            self.logger.info(f"训练完成，模型已保存到: {final_model_path}")
            
            # 评估模型
            await self.evaluate_model(model, env)
            
        except Exception as e:
            self.logger.error(f"训练过程中发生错误: {e}")
            raise
        finally:
            # 清理资源
            if 'env' in locals():
                await env.airsim_env.shutdown()
    
    async def evaluate_model(self, model, env, num_episodes: int = 10):
        """
        评估模型性能
        
        Args:
            model: 训练好的模型
            env: 评估环境
            num_episodes: 评估episode数量
        """
        self.logger.info(f"开始模型评估，评估{num_episodes}个episodes")
        
        total_rewards = []
        reward_components_stats = {
            'tracking': [],
            'recovery': [],
            'emergency': [],
            'safety': [],
            'efficiency': []
        }
        
        task_phase_stats = {
            TaskPhase.NORMAL: 0,
            TaskPhase.RECOVERY: 0,
            TaskPhase.EMERGENCY: 0
        }
        
        for episode in range(num_episodes):
            obs = await env.reset()
            episode_reward = 0
            episode_reward_components = {
                'tracking': 0,
                'recovery': 0,
                'emergency': 0,
                'safety': 0,
                'efficiency': 0
            }
            
            done = False
            step_count = 0
            
            while not done and step_count < 1000:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = await env.step(action)
                
                episode_reward += reward
                step_count += 1
                
                # 统计奖励组件
                if 'reward_components' in info:
                    for component, value in info['reward_components'].items():
                        episode_reward_components[component] += value
                
                # 统计任务阶段
                if 'task_phase' in info:
                    phase = TaskPhase(info['task_phase'])
                    task_phase_stats[phase] += 1
            
            total_rewards.append(episode_reward)
            
            for component, value in episode_reward_components.items():
                reward_components_stats[component].append(value)
            
            self.logger.info(
                f"Episode {episode + 1}: 总奖励={episode_reward:.2f}, "
                f"步数={step_count}, 任务阶段统计={dict(task_phase_stats)}"
            )
        
        # 计算统计信息
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        
        self.logger.info(f"评估完成:")
        self.logger.info(f"  平均总奖励: {avg_reward:.2f} ± {std_reward:.2f}")
        
        for component, values in reward_components_stats.items():
            if values:
                avg_component = np.mean(values)
                self.logger.info(f"  平均{component}奖励: {avg_component:.2f}")
        
        # 保存评估结果
        eval_results = {
            'timestamp': datetime.now().isoformat(),
            'num_episodes': num_episodes,
            'avg_reward': float(avg_reward),
            'std_reward': float(std_reward),
            'reward_components': {k: float(np.mean(v)) if v else 0.0 
                               for k, v in reward_components_stats.items()},
            'task_phase_stats': {k.value: v for k, v in task_phase_stats.items()}
        }
        
        eval_path = f"{self.model_save_path}/evaluation_results.yaml"
        with open(eval_path, 'w', encoding='utf-8') as f:
            yaml.dump(eval_results, f, default_flow_style=False, allow_unicode=True)
        
        self.logger.info(f"评估结果已保存到: {eval_path}")


def create_default_config():
    """
    创建默认训练配置
    
    Returns:
        Dict: 默认配置字典
    """
    return {
        'airsim': {
            'host': '127.0.0.1',
            'port': 41451
        },
        'training': {
            'total_timesteps': 100000,
            'max_episode_steps': 1000,
            'save_freq': 10000,
            'model': {
                'learning_rate': 3e-4,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.0,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5
            }
        },
        'model_save_path': './models/multidimensional_reward'
    }


async def main():
    """
    主函数
    """
    # 检查配置文件
    config_path = "configs/multidimensional_reward_training.yaml"
    
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        print("创建默认配置文件...")
        
        # 创建配置目录
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # 保存默认配置
        default_config = create_default_config()
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"默认配置文件已创建: {config_path}")
        print("请根据需要修改配置文件，然后重新运行训练脚本。")
        return
    
    # 创建训练器并开始训练
    trainer = MultiDimensionalRewardTrainer(config_path)
    await trainer.train()


if __name__ == "__main__":
    asyncio.run(main())