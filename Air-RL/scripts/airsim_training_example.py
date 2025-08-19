#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AirSim训练环境使用示例

该脚本演示如何使用AirSim环境进行无人机强化学习训练，包括：
- 环境初始化和连接
- 飞行计划创建和执行
- 传感器数据采集
- 图像获取和处理

Author: wdblink
Date: 2024
"""

import asyncio
import sys
import os
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Tuple

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.environment.airsim_env import (
    AirSimEnvironment, 
    AirSimTrainingEnvironment,
    FlightPlan,
    Waypoint,
    MissionStatus
)
from src.utils.config import load_config, AirSimConfig
from src.utils.logger import setup_logger


class AirSimTrainingDemo:
    """
    AirSim训练演示类
    
    演示AirSim环境的基本功能和使用方法。
    """
    
    def __init__(self):
        """
        初始化演示类
        """
        self.logger = setup_logger("AirSimTrainingDemo")
        
        # 加载配置
        self.config = load_config()
        self.airsim_config = self.config.airsim
        
        # 创建环境
        self.env = None
        self.training_env = None
        
        # 数据存储
        self.collected_images = []
        self.flight_data = []
    
    async def run_basic_demo(self):
        """
        运行基础演示
        
        演示AirSim环境的基本功能
        """
        self.logger.info("开始AirSim基础演示")
        
        try:
            # 1. 初始化环境
            await self._initialize_environment()
            
            # 2. 创建飞行计划
            flight_plan = self._create_demo_flight_plan()
            
            # 3. 执行飞行任务
            await self._execute_flight_mission(flight_plan)
            
            # 4. 数据分析和可视化
            await self._analyze_flight_data()
            
        except Exception as e:
            self.logger.error(f"演示执行失败: {e}")
        finally:
            # 清理资源
            await self._cleanup()
    
    async def run_training_demo(self):
        """
        运行训练演示
        
        演示如何使用AirSim进行强化学习训练
        """
        self.logger.info("开始AirSim训练演示")
        
        try:
            # 1. 初始化训练环境
            await self._initialize_training_environment()
            
            # 2. 运行训练轮次
            await self._run_training_episodes(num_episodes=3)
            
        except Exception as e:
            self.logger.error(f"训练演示失败: {e}")
        finally:
            await self._cleanup()
    
    async def _initialize_environment(self):
        """
        初始化AirSim环境
        """
        self.logger.info("初始化AirSim环境...")
        
        self.env = AirSimEnvironment(self.airsim_config)
        success = await self.env.initialize()
        
        if not success:
            raise RuntimeError("AirSim环境初始化失败")
        
        self.logger.info("AirSim环境初始化成功")
    
    async def _initialize_training_environment(self):
        """
        初始化训练环境
        """
        self.logger.info("初始化AirSim训练环境...")
        
        self.training_env = AirSimTrainingEnvironment(self.airsim_config)
        await self.training_env.airsim_env.initialize()
        
        self.logger.info("AirSim训练环境初始化成功")
    
    def _create_demo_flight_plan(self) -> FlightPlan:
        """
        创建演示飞行计划
        
        Returns:
            FlightPlan: 飞行计划对象
        """
        self.logger.info("创建演示飞行计划...")
        
        # 定义基于地理坐标的航点
        # 这里使用相对于原点的小范围坐标
        origin_lat = self.airsim_config.origin_latitude
        origin_lon = self.airsim_config.origin_longitude
        origin_alt = self.airsim_config.origin_altitude
        
        # 创建一个矩形飞行路径
        waypoints = [
            (origin_lat + 0.001, origin_lon, origin_alt + 20),  # 北
            (origin_lat + 0.001, origin_lon + 0.001, origin_alt + 20),  # 东北
            (origin_lat, origin_lon + 0.001, origin_alt + 20),  # 东
            (origin_lat - 0.001, origin_lon + 0.001, origin_alt + 20),  # 东南
            (origin_lat - 0.001, origin_lon, origin_alt + 20),  # 南
            (origin_lat - 0.001, origin_lon - 0.001, origin_alt + 20),  # 西南
            (origin_lat, origin_lon - 0.001, origin_alt + 20),  # 西
            (origin_lat + 0.001, origin_lon - 0.001, origin_alt + 20),  # 西北
            (origin_lat, origin_lon, origin_alt + 20),  # 返回中心
        ]
        
        flight_plan = self.env.create_flight_plan(
            waypoints=waypoints,
            name="demo_rectangular_mission"
        )
        
        self.logger.info(f"创建了包含{len(waypoints)}个航点的飞行计划")
        return flight_plan
    
    async def _execute_flight_mission(self, flight_plan: FlightPlan):
        """
        执行飞行任务
        
        Args:
            flight_plan: 飞行计划
        """
        self.logger.info("开始执行飞行任务...")
        
        # 执行飞行计划
        success = await self.env.execute_flight_plan(flight_plan)
        
        if success:
            self.logger.info("飞行任务执行成功")
        else:
            self.logger.error("飞行任务执行失败")
            return
        
        # 监控飞行状态并收集数据
        await self._monitor_flight_progress()
    
    async def _monitor_flight_progress(self):
        """
        监控飞行进度并收集数据
        """
        self.logger.info("开始监控飞行进度...")
        
        while self.env.get_mission_status() == MissionStatus.EXECUTING:
            # 收集传感器数据
            sensor_data = await self.env._collect_sensor_data()
            
            if sensor_data:
                # 保存飞行数据
                flight_record = {
                    'timestamp': sensor_data.timestamp,
                    'position': {
                        'x': sensor_data.position.x,
                        'y': sensor_data.position.y,
                        'z': sensor_data.position.z
                    },
                    'attitude': {
                        'roll': sensor_data.attitude.roll,
                        'pitch': sensor_data.attitude.pitch,
                        'yaw': sensor_data.attitude.yaw
                    },
                    'velocity': {
                        'vx': sensor_data.velocity.vx,
                        'vy': sensor_data.velocity.vy,
                        'vz': sensor_data.velocity.vz
                    }
                }
                self.flight_data.append(flight_record)
                
                # 保存图像数据
                if sensor_data.camera_image is not None:
                    self.collected_images.append({
                        'timestamp': sensor_data.timestamp,
                        'camera_image': sensor_data.camera_image.copy(),
                        'satellite_image': sensor_data.satellite_image.copy() if sensor_data.satellite_image is not None else None
                    })
                
                self.logger.info(f"收集数据点: 位置({sensor_data.position.x:.2f}, {sensor_data.position.y:.2f}, {sensor_data.position.z:.2f})")
            
            # 等待一段时间再次检查
            await asyncio.sleep(1.0)
        
        self.logger.info(f"飞行监控完成，共收集{len(self.flight_data)}个数据点")
    
    async def _run_training_episodes(self, num_episodes: int = 3):
        """
        运行训练轮次
        
        Args:
            num_episodes: 训练轮次数
        """
        self.logger.info(f"开始运行{num_episodes}个训练轮次...")
        
        for episode in range(num_episodes):
            self.logger.info(f"开始第{episode + 1}轮训练")
            
            # 重置环境
            observation = await self.training_env.reset()
            self.logger.info(f"环境重置完成，初始观测: {list(observation.keys())}")
            
            # 运行一个轮次
            step_count = 0
            total_reward = 0.0
            
            while step_count < 50:  # 限制步数用于演示
                # 生成随机动作(实际训练中应该使用智能体策略)
                action = self._generate_random_action()
                
                # 执行动作
                observation, reward, done, info = await self.training_env.step(action)
                
                total_reward += reward
                step_count += 1
                
                self.logger.info(f"步骤{step_count}: 奖励={reward:.2f}, 累计奖励={total_reward:.2f}")
                
                if done:
                    break
                
                # 短暂等待
                await asyncio.sleep(0.1)
            
            self.logger.info(f"第{episode + 1}轮训练完成，总奖励: {total_reward:.2f}")
    
    def _generate_random_action(self) -> dict:
        """
        生成随机动作(用于演示)
        
        Returns:
            dict: 动作字典
        """
        return {
            'thrust': np.random.uniform(0.3, 0.7),
            'roll': np.random.uniform(-0.1, 0.1),
            'pitch': np.random.uniform(-0.1, 0.1),
            'yaw_rate': np.random.uniform(-0.1, 0.1)
        }
    
    async def _analyze_flight_data(self):
        """
        分析飞行数据并生成可视化
        """
        self.logger.info("开始分析飞行数据...")
        
        if not self.flight_data:
            self.logger.warning("没有飞行数据可供分析")
            return
        
        # 提取位置数据
        timestamps = [record['timestamp'] for record in self.flight_data]
        positions_x = [record['position']['x'] for record in self.flight_data]
        positions_y = [record['position']['y'] for record in self.flight_data]
        positions_z = [record['position']['z'] for record in self.flight_data]
        
        # 创建可视化
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('AirSim飞行数据分析', fontsize=16)
        
        # 3D轨迹图
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax1.plot(positions_x, positions_y, positions_z, 'b-', linewidth=2)
        ax1.scatter(positions_x[0], positions_y[0], positions_z[0], color='green', s=100, label='起点')
        ax1.scatter(positions_x[-1], positions_y[-1], positions_z[-1], color='red', s=100, label='终点')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D飞行轨迹')
        ax1.legend()
        
        # 2D轨迹图
        axes[0, 1].plot(positions_x, positions_y, 'b-', linewidth=2)
        axes[0, 1].scatter(positions_x[0], positions_y[0], color='green', s=100, label='起点')
        axes[0, 1].scatter(positions_x[-1], positions_y[-1], color='red', s=100, label='终点')
        axes[0, 1].set_xlabel('X (m)')
        axes[0, 1].set_ylabel('Y (m)')
        axes[0, 1].set_title('2D飞行轨迹')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 高度变化
        relative_time = [(t - timestamps[0]) for t in timestamps]
        axes[1, 0].plot(relative_time, positions_z, 'r-', linewidth=2)
        axes[1, 0].set_xlabel('时间 (s)')
        axes[1, 0].set_ylabel('高度 (m)')
        axes[1, 0].set_title('高度变化')
        axes[1, 0].grid(True)
        
        # 速度分析
        velocities = []
        for record in self.flight_data:
            vel = record['velocity']
            speed = np.sqrt(vel['vx']**2 + vel['vy']**2 + vel['vz']**2)
            velocities.append(speed)
        
        axes[1, 1].plot(relative_time, velocities, 'g-', linewidth=2)
        axes[1, 1].set_xlabel('时间 (s)')
        axes[1, 1].set_ylabel('速度 (m/s)')
        axes[1, 1].set_title('飞行速度')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # 保存图像
        output_dir = project_root / "output"
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "flight_analysis.png", dpi=300, bbox_inches='tight')
        self.logger.info(f"飞行分析图已保存到: {output_dir / 'flight_analysis.png'}")
        
        # 显示图像采集结果
        if self.collected_images:
            self._display_collected_images()
    
    def _display_collected_images(self):
        """
        显示采集的图像
        """
        self.logger.info(f"显示采集的{len(self.collected_images)}张图像...")
        
        # 选择几张代表性图像显示
        num_display = min(4, len(self.collected_images))
        indices = np.linspace(0, len(self.collected_images)-1, num_display, dtype=int)
        
        fig, axes = plt.subplots(2, num_display, figsize=(15, 8))
        fig.suptitle('采集的相机和卫星图像', fontsize=16)
        
        for i, idx in enumerate(indices):
            image_data = self.collected_images[idx]
            
            # 显示相机图像
            if image_data['camera_image'] is not None:
                if num_display == 1:
                    axes[0].imshow(cv2.cvtColor(image_data['camera_image'], cv2.COLOR_BGR2RGB))
                    axes[0].set_title(f'相机图像 {idx+1}')
                    axes[0].axis('off')
                else:
                    axes[0, i].imshow(cv2.cvtColor(image_data['camera_image'], cv2.COLOR_BGR2RGB))
                    axes[0, i].set_title(f'相机图像 {idx+1}')
                    axes[0, i].axis('off')
            
            # 显示卫星图像
            if image_data['satellite_image'] is not None:
                if num_display == 1:
                    axes[1].imshow(cv2.cvtColor(image_data['satellite_image'], cv2.COLOR_BGR2RGB))
                    axes[1].set_title(f'卫星图像 {idx+1}')
                    axes[1].axis('off')
                else:
                    axes[1, i].imshow(cv2.cvtColor(image_data['satellite_image'], cv2.COLOR_BGR2RGB))
                    axes[1, i].set_title(f'卫星图像 {idx+1}')
                    axes[1, i].axis('off')
        
        plt.tight_layout()
        
        # 保存图像
        output_dir = project_root / "output"
        plt.savefig(output_dir / "collected_images.png", dpi=300, bbox_inches='tight')
        self.logger.info(f"图像采集结果已保存到: {output_dir / 'collected_images.png'}")
    
    async def _cleanup(self):
        """
        清理资源
        """
        self.logger.info("清理资源...")
        
        if self.env:
            await self.env.shutdown()
        
        if self.training_env:
            await self.training_env.airsim_env.shutdown()
        
        self.logger.info("资源清理完成")


async def main():
    """
    主函数
    """
    demo = AirSimTrainingDemo()
    
    print("AirSim训练环境演示")
    print("=" * 50)
    print("请选择演示模式:")
    print("1. 基础飞行演示")
    print("2. 训练环境演示")
    print("3. 退出")
    
    while True:
        try:
            choice = input("请输入选择 (1-3): ").strip()
            
            if choice == '1':
                print("\n开始基础飞行演示...")
                await demo.run_basic_demo()
                break
            elif choice == '2':
                print("\n开始训练环境演示...")
                await demo.run_training_demo()
                break
            elif choice == '3':
                print("退出演示")
                break
            else:
                print("无效选择，请重新输入")
                
        except KeyboardInterrupt:
            print("\n用户中断，退出演示")
            break
        except Exception as e:
            print(f"演示过程中发生错误: {e}")
            break


if __name__ == "__main__":
    # 检查AirSim是否可用
    try:
        import airsim
        print("AirSim包已安装")
    except ImportError:
        print("警告: AirSim包未安装，请先安装AirSim Python包")
        print("安装命令: pip install airsim")
        sys.exit(1)
    
    # 运行演示
    asyncio.run(main())