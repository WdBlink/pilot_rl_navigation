#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强化学习无人机定位导航系统 - 强化学习智能体训练脚本

本脚本用于训练强化学习智能体，包括：
1. 环境初始化和配置
2. 智能体创建和参数设置
3. 训练过程管理和监控
4. 模型保存和评估
5. 训练结果可视化
6. 超参数调优支持

Author: wdblink
Date: 2024
"""

import os
import sys
import time
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import json

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# 导入项目模块
from core.rl_agent import RLAgent
from environment.airsim_env import AirSimEnvironment
from utils.logger import logger_manager
from utils.config import ConfigManager
from utils.visualization import PerformanceAnalyzer


class TrainingManager:
    """训练管理器
    
    负责管理整个训练过程，包括：
    1. 训练配置和环境设置
    2. 训练过程监控和日志记录
    3. 模型保存和检查点管理
    4. 性能评估和可视化
    """
    
    def __init__(self, config_path: str, output_dir: str):
        """
        初始化训练管理器
        
        Args:
            config_path: 配置文件路径
            output_dir: 输出目录
        """
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载配置
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        
        # 训练参数
        self.training_config = self.config.get('training', {})
        self.algorithm = self.training_config.get('algorithm', 'ppo')
        self.total_timesteps = self.training_config.get('total_timesteps', 100000)
        self.eval_freq = self.training_config.get('eval_freq', 10000)
        self.save_freq = self.training_config.get('save_freq', 20000)
        
        # 创建输出子目录
        self.models_dir = self.output_dir / "models"
        self.logs_dir = self.output_dir / "logs"
        self.plots_dir = self.output_dir / "plots"
        
        for directory in [self.models_dir, self.logs_dir, self.plots_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # 训练统计
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_rates': [],
            'collision_rates': [],
            'training_times': [],
            'evaluation_scores': [],
            'best_score': float('-inf'),
            'total_episodes': 0,
            'total_steps': 0
        }
        
        # 性能分析器
        self.performance_analyzer = PerformanceAnalyzer()
        
        logger_manager.info(f"训练管理器初始化完成: {self.algorithm}")
    
    def create_environment(self) -> AirSimEnvironment:
        """
        创建训练环境
        
        Returns:
            AirSim环境实例
        """
        try:
            env_config = self.config.get('environment', {})
            env = AirSimEnvironment(config=env_config)
            
            logger_manager.info(f"训练环境创建成功: {env.get_info()}")
            return env
        
        except Exception as e:
            logger_manager.error(f"环境创建失败: {e}")
            raise
    
    def create_agent(self, env: AirSimEnvironment) -> RLAgent:
        """
        创建强化学习智能体
        
        Args:
            env: 训练环境
            
        Returns:
            强化学习智能体实例
        """
        try:
            agent_config = self.config.get('agent', {})
            agent_config['algorithm'] = self.algorithm
            
            agent = RLAgent(
                env=env,
                algorithm=self.algorithm,
                config=agent_config
            )
            
            logger_manager.info(f"智能体创建成功: {self.algorithm}")
            return agent
        
        except Exception as e:
            logger_manager.error(f"智能体创建失败: {e}")
            raise
    
    def train(self, resume_from: Optional[str] = None) -> Dict[str, Any]:
        """
        执行训练过程
        
        Args:
            resume_from: 恢复训练的模型路径
            
        Returns:
            训练结果统计
        """
        start_time = time.time()
        
        try:
            # 创建环境和智能体
            env = self.create_environment()
            agent = self.create_agent(env)
            
            # 恢复训练
            if resume_from:
                logger_manager.info(f"从检查点恢复训练: {resume_from}")
                agent.load(resume_from)
            
            # 设置回调函数
            self._setup_callbacks(agent)
            
            logger_manager.info(f"开始训练 - 算法: {self.algorithm}, 总步数: {self.total_timesteps}")
            
            # 执行训练
            agent.train(
                total_timesteps=self.total_timesteps,
                callback=self._training_callback,
                eval_env=env,
                eval_freq=self.eval_freq,
                n_eval_episodes=10,
                eval_log_path=str(self.logs_dir)
            )
            
            # 保存最终模型
            final_model_path = self.models_dir / f"{self.algorithm}_final_model"
            agent.save(str(final_model_path))
            
            # 训练完成统计
            training_time = time.time() - start_time
            self.training_stats['total_training_time'] = training_time
            
            logger_manager.info(f"训练完成 - 总时间: {training_time:.2f}s")
            
            # 生成训练报告
            self._generate_training_report()
            
            # 清理环境
            env.close()
            
            return self.training_stats
        
        except Exception as e:
            logger_manager.error(f"训练过程异常: {e}")
            raise
    
    def _setup_callbacks(self, agent: RLAgent) -> None:
        """
        设置训练回调函数
        
        Args:
            agent: 强化学习智能体
        """
        # 添加性能监控回调
        def performance_callback(locals_, globals_):
            if 'infos' in locals_ and locals_['infos']:
                for info in locals_['infos']:
                    if isinstance(info, dict):
                        # 记录性能指标
                        if 'episode_reward' in info:
                            self.training_stats['episode_rewards'].append(info['episode_reward'])
                        if 'episode_length' in info:
                            self.training_stats['episode_lengths'].append(info['episode_length'])
                        if 'success_rate' in info:
                            self.training_stats['success_rates'].append(info['success_rate'])
                        if 'collision_rate' in info:
                            self.training_stats['collision_rates'].append(info['collision_rate'])
            
            return True
        
        agent.add_callback(performance_callback)
    
    def _training_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> bool:
        """
        训练过程回调函数
        
        Args:
            locals_: 局部变量
            globals_: 全局变量
            
        Returns:
            是否继续训练
        """
        try:
            # 获取当前步数
            current_step = locals_.get('self').num_timesteps
            
            # 定期保存检查点
            if current_step % self.save_freq == 0:
                checkpoint_path = self.models_dir / f"{self.algorithm}_checkpoint_{current_step}"
                locals_.get('self').save(str(checkpoint_path))
                logger_manager.info(f"保存检查点: {checkpoint_path}")
            
            # 更新统计信息
            self.training_stats['total_steps'] = current_step
            
            # 记录训练进度
            if current_step % 1000 == 0:
                progress = (current_step / self.total_timesteps) * 100
                logger_manager.info(f"训练进度: {progress:.1f}% ({current_step}/{self.total_timesteps})")
            
            return True
        
        except Exception as e:
            logger_manager.error(f"训练回调异常: {e}")
            return False
    
    def evaluate_model(self, model_path: str, num_episodes: int = 10) -> Dict[str, float]:
        """
        评估训练好的模型
        
        Args:
            model_path: 模型路径
            num_episodes: 评估回合数
            
        Returns:
            评估结果
        """
        try:
            logger_manager.info(f"开始模型评估: {model_path}")
            
            # 创建评估环境
            env = self.create_environment()
            
            # 加载模型
            agent = self.create_agent(env)
            agent.load(model_path)
            
            # 执行评估
            episode_rewards = []
            episode_lengths = []
            success_count = 0
            collision_count = 0
            
            for episode in range(num_episodes):
                obs = env.reset()
                episode_reward = 0
                episode_length = 0
                done = False
                
                while not done:
                    action = agent.predict(obs, deterministic=True)
                    obs, reward, done, info = env.step(action)
                    episode_reward += reward
                    episode_length += 1
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                # 统计成功和碰撞
                if info.get('success', False):
                    success_count += 1
                if info.get('collision', False):
                    collision_count += 1
                
                logger_manager.info(f"评估回合 {episode + 1}: 奖励={episode_reward:.2f}, 长度={episode_length}")
            
            # 计算评估指标
            eval_results = {
                'mean_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'mean_length': np.mean(episode_lengths),
                'success_rate': success_count / num_episodes,
                'collision_rate': collision_count / num_episodes,
                'max_reward': np.max(episode_rewards),
                'min_reward': np.min(episode_rewards)
            }
            
            logger_manager.info(f"评估完成: {eval_results}")
            
            # 清理环境
            env.close()
            
            return eval_results
        
        except Exception as e:
            logger_manager.error(f"模型评估异常: {e}")
            raise
    
    def _generate_training_report(self) -> None:
        """
        生成训练报告
        """
        try:
            # 保存训练统计
            stats_file = self.output_dir / "training_stats.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.training_stats, f, indent=2, ensure_ascii=False)
            
            # 生成训练曲线图
            self._plot_training_curves()
            
            # 生成文本报告
            self._generate_text_report()
            
            logger_manager.info(f"训练报告生成完成: {self.output_dir}")
        
        except Exception as e:
            logger_manager.error(f"训练报告生成异常: {e}")
    
    def _plot_training_curves(self) -> None:
        """
        绘制训练曲线
        """
        try:
            # 创建子图
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'{self.algorithm.upper()} 训练结果', fontsize=16)
            
            # 奖励曲线
            if self.training_stats['episode_rewards']:
                axes[0, 0].plot(self.training_stats['episode_rewards'])
                axes[0, 0].set_title('回合奖励')
                axes[0, 0].set_xlabel('回合')
                axes[0, 0].set_ylabel('奖励')
                axes[0, 0].grid(True)
            
            # 回合长度曲线
            if self.training_stats['episode_lengths']:
                axes[0, 1].plot(self.training_stats['episode_lengths'])
                axes[0, 1].set_title('回合长度')
                axes[0, 1].set_xlabel('回合')
                axes[0, 1].set_ylabel('步数')
                axes[0, 1].grid(True)
            
            # 成功率曲线
            if self.training_stats['success_rates']:
                axes[1, 0].plot(self.training_stats['success_rates'])
                axes[1, 0].set_title('成功率')
                axes[1, 0].set_xlabel('回合')
                axes[1, 0].set_ylabel('成功率')
                axes[1, 0].grid(True)
            
            # 碰撞率曲线
            if self.training_stats['collision_rates']:
                axes[1, 1].plot(self.training_stats['collision_rates'])
                axes[1, 1].set_title('碰撞率')
                axes[1, 1].set_xlabel('回合')
                axes[1, 1].set_ylabel('碰撞率')
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            # 保存图像
            plot_file = self.plots_dir / f"{self.algorithm}_training_curves.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger_manager.info(f"训练曲线保存: {plot_file}")
        
        except Exception as e:
            logger_manager.error(f"训练曲线绘制异常: {e}")
    
    def _generate_text_report(self) -> None:
        """
        生成文本报告
        """
        try:
            report_file = self.output_dir / "training_report.txt"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(f"强化学习无人机导航训练报告\n")
                f.write(f"{'=' * 50}\n\n")
                
                # 基本信息
                f.write(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"算法: {self.algorithm.upper()}\n")
                f.write(f"总训练步数: {self.training_stats.get('total_steps', 0)}\n")
                f.write(f"总训练时间: {self.training_stats.get('total_training_time', 0):.2f}秒\n\n")
                
                # 性能统计
                if self.training_stats['episode_rewards']:
                    rewards = self.training_stats['episode_rewards']
                    f.write(f"回合奖励统计:\n")
                    f.write(f"  平均奖励: {np.mean(rewards):.2f}\n")
                    f.write(f"  最大奖励: {np.max(rewards):.2f}\n")
                    f.write(f"  最小奖励: {np.min(rewards):.2f}\n")
                    f.write(f"  标准差: {np.std(rewards):.2f}\n\n")
                
                if self.training_stats['success_rates']:
                    success_rates = self.training_stats['success_rates']
                    f.write(f"成功率统计:\n")
                    f.write(f"  平均成功率: {np.mean(success_rates):.2%}\n")
                    f.write(f"  最终成功率: {success_rates[-1]:.2%}\n\n")
                
                if self.training_stats['collision_rates']:
                    collision_rates = self.training_stats['collision_rates']
                    f.write(f"碰撞率统计:\n")
                    f.write(f"  平均碰撞率: {np.mean(collision_rates):.2%}\n")
                    f.write(f"  最终碰撞率: {collision_rates[-1]:.2%}\n\n")
                
                # 配置信息
                f.write(f"训练配置:\n")
                f.write(f"  配置文件: {self.config_path}\n")
                f.write(f"  输出目录: {self.output_dir}\n")
                f.write(f"  评估频率: {self.eval_freq}\n")
                f.write(f"  保存频率: {self.save_freq}\n")
            
            logger_manager.info(f"文本报告保存: {report_file}")
        
        except Exception as e:
            logger_manager.error(f"文本报告生成异常: {e}")


def parse_arguments() -> argparse.Namespace:
    """
    解析命令行参数
    
    Returns:
        解析后的参数
    """
    parser = argparse.ArgumentParser(
        description="强化学习无人机导航系统训练脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/navigation_config.yaml",
        help="配置文件路径"
    )
    
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["ppo", "sac", "td3", "a2c", "ddpg"],
        default="ppo",
        help="强化学习算法"
    )
    
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="总训练步数"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/training",
        help="输出目录"
    )
    
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="恢复训练的模型路径"
    )
    
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="评估回合数"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细输出"
    )
    
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="仅评估模式"
    )
    
    return parser.parse_args()


def main():
    """
    主函数
    """
    # 解析参数
    args = parse_arguments()
    
    # 设置随机种子
    np.random.seed(args.seed)
    
    # 配置日志
    if args.verbose:
        logger_manager.set_level("DEBUG")
    
    try:
        # 创建训练管理器
        trainer = TrainingManager(
            config_path=args.config,
            output_dir=args.output_dir
        )
        
        # 更新配置
        if hasattr(trainer.training_config, 'update'):
            trainer.training_config.update({
                'algorithm': args.algorithm,
                'total_timesteps': args.timesteps
            })
        
        if args.eval_only:
            # 仅评估模式
            if not args.resume_from:
                logger_manager.error("评估模式需要指定模型路径 --resume-from")
                sys.exit(1)
            
            logger_manager.info("开始模型评估")
            eval_results = trainer.evaluate_model(
                model_path=args.resume_from,
                num_episodes=args.eval_episodes
            )
            
            print("\n=== 评估结果 ===")
            for key, value in eval_results.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
        
        else:
            # 训练模式
            logger_manager.info("开始强化学习训练")
            training_results = trainer.train(resume_from=args.resume_from)
            
            print("\n=== 训练完成 ===")
            print(f"总训练步数: {training_results.get('total_steps', 0)}")
            print(f"训练时间: {training_results.get('total_training_time', 0):.2f}秒")
            
            if training_results['episode_rewards']:
                rewards = training_results['episode_rewards']
                print(f"平均奖励: {np.mean(rewards):.2f}")
                print(f"最大奖励: {np.max(rewards):.2f}")
            
            print(f"输出目录: {args.output_dir}")
    
    except KeyboardInterrupt:
        logger_manager.info("训练被用户中断")
        sys.exit(0)
    
    except Exception as e:
        logger_manager.error(f"训练异常: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()