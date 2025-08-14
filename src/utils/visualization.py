#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强化学习无人机定位导航系统 - 可视化工具模块

本模块提供系统运行状态的可视化功能，包括：
1. 实时轨迹可视化
2. 传感器数据可视化
3. 融合结果对比
4. 性能指标图表
5. 3D飞行路径显示
6. 传感器可靠性监控

Author: wdblink
Date: 2024
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from collections import deque
import threading
import queue

# 导入项目模块
from .data_types import (
    Position3D, FlightAttitude, SystemState, 
    OpticalMatchResult, ReliabilityMetrics
)
from .logger import logger_manager, performance_monitor

# 设置matplotlib样式
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class TrajectoryVisualizer:
    """轨迹可视化器
    
    实时显示无人机的飞行轨迹，包括：
    1. 真实轨迹
    2. 各传感器估计轨迹
    3. 融合轨迹
    4. 误差分析
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化轨迹可视化器
        
        Args:
            config: 可视化配置
        """
        self.config = config
        self.max_points = config.get('max_trajectory_points', 1000)
        self.update_interval = config.get('update_interval', 0.1)
        
        # 轨迹数据存储
        self.true_trajectory = deque(maxlen=self.max_points)
        self.imu_trajectory = deque(maxlen=self.max_points)
        self.gps_trajectory = deque(maxlen=self.max_points)
        self.optical_trajectory = deque(maxlen=self.max_points)
        self.fusion_trajectory = deque(maxlen=self.max_points)
        
        # 时间戳
        self.timestamps = deque(maxlen=self.max_points)
        
        # 图形对象
        self.fig = None
        self.ax_3d = None
        self.ax_2d = None
        self.lines = {}
        
        # 动画对象
        self.animation = None
        self.is_running = False
        
        logger_manager.info("轨迹可视化器初始化完成")
    
    def initialize_plots(self) -> None:
        """
        初始化绘图窗口
        """
        # 创建子图
        self.fig = plt.figure(figsize=(15, 10))
        
        # 3D轨迹图
        self.ax_3d = self.fig.add_subplot(221, projection='3d')
        self.ax_3d.set_title('3D Flight Trajectory')
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')
        
        # 2D俯视图
        self.ax_2d = self.fig.add_subplot(222)
        self.ax_2d.set_title('2D Top View')
        self.ax_2d.set_xlabel('X (m)')
        self.ax_2d.set_ylabel('Y (m)')
        self.ax_2d.grid(True)
        self.ax_2d.axis('equal')
        
        # 高度时间图
        self.ax_alt = self.fig.add_subplot(223)
        self.ax_alt.set_title('Altitude vs Time')
        self.ax_alt.set_xlabel('Time (s)')
        self.ax_alt.set_ylabel('Altitude (m)')
        self.ax_alt.grid(True)
        
        # 误差分析图
        self.ax_error = self.fig.add_subplot(224)
        self.ax_error.set_title('Position Error Analysis')
        self.ax_error.set_xlabel('Time (s)')
        self.ax_error.set_ylabel('Error (m)')
        self.ax_error.grid(True)
        
        # 初始化线条
        self._initialize_lines()
        
        plt.tight_layout()
    
    def _initialize_lines(self) -> None:
        """
        初始化绘图线条
        """
        # 3D线条
        self.lines['true_3d'], = self.ax_3d.plot([], [], [], 'k-', linewidth=2, label='True')
        self.lines['fusion_3d'], = self.ax_3d.plot([], [], [], 'r-', linewidth=2, label='Fusion')
        self.lines['gps_3d'], = self.ax_3d.plot([], [], [], 'b--', alpha=0.7, label='GPS')
        self.lines['optical_3d'], = self.ax_3d.plot([], [], [], 'g--', alpha=0.7, label='Optical')
        self.ax_3d.legend()
        
        # 2D线条
        self.lines['true_2d'], = self.ax_2d.plot([], [], 'k-', linewidth=2, label='True')
        self.lines['fusion_2d'], = self.ax_2d.plot([], [], 'r-', linewidth=2, label='Fusion')
        self.lines['gps_2d'], = self.ax_2d.plot([], [], 'b--', alpha=0.7, label='GPS')
        self.lines['optical_2d'], = self.ax_2d.plot([], [], 'g--', alpha=0.7, label='Optical')
        self.ax_2d.legend()
        
        # 高度线条
        self.lines['true_alt'], = self.ax_alt.plot([], [], 'k-', linewidth=2, label='True')
        self.lines['fusion_alt'], = self.ax_alt.plot([], [], 'r-', linewidth=2, label='Fusion')
        self.lines['gps_alt'], = self.ax_alt.plot([], [], 'b--', alpha=0.7, label='GPS')
        self.ax_alt.legend()
        
        # 误差线条
        self.lines['fusion_error'], = self.ax_error.plot([], [], 'r-', linewidth=2, label='Fusion Error')
        self.lines['gps_error'], = self.ax_error.plot([], [], 'b--', alpha=0.7, label='GPS Error')
        self.lines['optical_error'], = self.ax_error.plot([], [], 'g--', alpha=0.7, label='Optical Error')
        self.ax_error.legend()
    
    @performance_monitor
    def update_trajectory(self, system_state: SystemState) -> None:
        """
        更新轨迹数据
        
        Args:
            system_state: 系统状态
        """
        current_time = time.time()
        
        # 添加时间戳
        self.timestamps.append(current_time)
        
        # 添加真实轨迹（如果可用）
        if hasattr(system_state, 'true_position') and system_state.true_position:
            self.true_trajectory.append(system_state.true_position)
        
        # 添加各传感器轨迹
        if system_state.inertial_position:
            self.imu_trajectory.append(system_state.inertial_position)
        
        if system_state.gps_position:
            self.gps_trajectory.append(system_state.gps_position)
        
        if system_state.optical_position:
            self.optical_trajectory.append(system_state.optical_position)
        
        # 添加融合轨迹
        if hasattr(system_state, 'fusion_position') and system_state.fusion_position:
            self.fusion_trajectory.append(system_state.fusion_position)
    
    def animate_update(self, frame) -> List:
        """
        动画更新函数
        
        Args:
            frame: 帧编号
            
        Returns:
            更新的线条列表
        """
        if not self.is_running:
            return list(self.lines.values())
        
        # 更新3D轨迹
        self._update_3d_trajectory()
        
        # 更新2D轨迹
        self._update_2d_trajectory()
        
        # 更新高度图
        self._update_altitude_plot()
        
        # 更新误差图
        self._update_error_plot()
        
        return list(self.lines.values())
    
    def _update_3d_trajectory(self) -> None:
        """
        更新3D轨迹显示
        """
        # 真实轨迹
        if self.true_trajectory:
            x = [pos.x for pos in self.true_trajectory]
            y = [pos.y for pos in self.true_trajectory]
            z = [pos.z for pos in self.true_trajectory]
            self.lines['true_3d'].set_data_3d(x, y, z)
        
        # 融合轨迹
        if self.fusion_trajectory:
            x = [pos.x for pos in self.fusion_trajectory]
            y = [pos.y for pos in self.fusion_trajectory]
            z = [pos.z for pos in self.fusion_trajectory]
            self.lines['fusion_3d'].set_data_3d(x, y, z)
        
        # GPS轨迹
        if self.gps_trajectory:
            x = [pos.x for pos in self.gps_trajectory]
            y = [pos.y for pos in self.gps_trajectory]
            z = [pos.z for pos in self.gps_trajectory]
            self.lines['gps_3d'].set_data_3d(x, y, z)
        
        # 光学轨迹
        if self.optical_trajectory:
            x = [pos.x for pos in self.optical_trajectory]
            y = [pos.y for pos in self.optical_trajectory]
            z = [pos.z for pos in self.optical_trajectory]
            self.lines['optical_3d'].set_data_3d(x, y, z)
        
        # 自动调整3D视图范围
        self._auto_scale_3d()
    
    def _update_2d_trajectory(self) -> None:
        """
        更新2D轨迹显示
        """
        # 真实轨迹
        if self.true_trajectory:
            x = [pos.x for pos in self.true_trajectory]
            y = [pos.y for pos in self.true_trajectory]
            self.lines['true_2d'].set_data(x, y)
        
        # 融合轨迹
        if self.fusion_trajectory:
            x = [pos.x for pos in self.fusion_trajectory]
            y = [pos.y for pos in self.fusion_trajectory]
            self.lines['fusion_2d'].set_data(x, y)
        
        # GPS轨迹
        if self.gps_trajectory:
            x = [pos.x for pos in self.gps_trajectory]
            y = [pos.y for pos in self.gps_trajectory]
            self.lines['gps_2d'].set_data(x, y)
        
        # 光学轨迹
        if self.optical_trajectory:
            x = [pos.x for pos in self.optical_trajectory]
            y = [pos.y for pos in self.optical_trajectory]
            self.lines['optical_2d'].set_data(x, y)
        
        # 自动调整2D视图范围
        self._auto_scale_2d()
    
    def _update_altitude_plot(self) -> None:
        """
        更新高度图显示
        """
        if not self.timestamps:
            return
        
        # 计算相对时间
        start_time = self.timestamps[0]
        times = [(t - start_time) for t in self.timestamps]
        
        # 真实高度
        if self.true_trajectory and len(self.true_trajectory) == len(times):
            altitudes = [pos.z for pos in self.true_trajectory]
            self.lines['true_alt'].set_data(times, altitudes)
        
        # 融合高度
        if self.fusion_trajectory and len(self.fusion_trajectory) == len(times):
            altitudes = [pos.z for pos in self.fusion_trajectory]
            self.lines['fusion_alt'].set_data(times, altitudes)
        
        # GPS高度
        if self.gps_trajectory and len(self.gps_trajectory) == len(times):
            altitudes = [pos.z for pos in self.gps_trajectory]
            self.lines['gps_alt'].set_data(times, altitudes)
        
        # 自动调整高度图范围
        self.ax_alt.relim()
        self.ax_alt.autoscale_view()
    
    def _update_error_plot(self) -> None:
        """
        更新误差图显示
        """
        if not self.timestamps or not self.true_trajectory:
            return
        
        # 计算相对时间
        start_time = self.timestamps[0]
        times = [(t - start_time) for t in self.timestamps]
        
        # 计算融合误差
        if self.fusion_trajectory and len(self.fusion_trajectory) == len(self.true_trajectory):
            errors = []
            for true_pos, fusion_pos in zip(self.true_trajectory, self.fusion_trajectory):
                error = np.sqrt(
                    (true_pos.x - fusion_pos.x) ** 2 +
                    (true_pos.y - fusion_pos.y) ** 2 +
                    (true_pos.z - fusion_pos.z) ** 2
                )
                errors.append(error)
            
            if len(errors) == len(times):
                self.lines['fusion_error'].set_data(times, errors)
        
        # 计算GPS误差
        if self.gps_trajectory and len(self.gps_trajectory) == len(self.true_trajectory):
            errors = []
            for true_pos, gps_pos in zip(self.true_trajectory, self.gps_trajectory):
                error = np.sqrt(
                    (true_pos.x - gps_pos.x) ** 2 +
                    (true_pos.y - gps_pos.y) ** 2 +
                    (true_pos.z - gps_pos.z) ** 2
                )
                errors.append(error)
            
            if len(errors) == len(times):
                self.lines['gps_error'].set_data(times, errors)
        
        # 计算光学误差
        if self.optical_trajectory and len(self.optical_trajectory) == len(self.true_trajectory):
            errors = []
            for true_pos, optical_pos in zip(self.true_trajectory, self.optical_trajectory):
                error = np.sqrt(
                    (true_pos.x - optical_pos.x) ** 2 +
                    (true_pos.y - optical_pos.y) ** 2 +
                    (true_pos.z - optical_pos.z) ** 2
                )
                errors.append(error)
            
            if len(errors) == len(times):
                self.lines['optical_error'].set_data(times, errors)
        
        # 自动调整误差图范围
        self.ax_error.relim()
        self.ax_error.autoscale_view()
    
    def _auto_scale_3d(self) -> None:
        """
        自动调整3D视图范围
        """
        all_positions = []
        all_positions.extend(self.true_trajectory)
        all_positions.extend(self.fusion_trajectory)
        all_positions.extend(self.gps_trajectory)
        all_positions.extend(self.optical_trajectory)
        
        if all_positions:
            x_coords = [pos.x for pos in all_positions]
            y_coords = [pos.y for pos in all_positions]
            z_coords = [pos.z for pos in all_positions]
            
            self.ax_3d.set_xlim(min(x_coords) - 10, max(x_coords) + 10)
            self.ax_3d.set_ylim(min(y_coords) - 10, max(y_coords) + 10)
            self.ax_3d.set_zlim(min(z_coords) - 10, max(z_coords) + 10)
    
    def _auto_scale_2d(self) -> None:
        """
        自动调整2D视图范围
        """
        all_positions = []
        all_positions.extend(self.true_trajectory)
        all_positions.extend(self.fusion_trajectory)
        all_positions.extend(self.gps_trajectory)
        all_positions.extend(self.optical_trajectory)
        
        if all_positions:
            x_coords = [pos.x for pos in all_positions]
            y_coords = [pos.y for pos in all_positions]
            
            self.ax_2d.set_xlim(min(x_coords) - 10, max(x_coords) + 10)
            self.ax_2d.set_ylim(min(y_coords) - 10, max(y_coords) + 10)
    
    def start_animation(self) -> None:
        """
        启动动画显示
        """
        if self.fig is None:
            self.initialize_plots()
        
        self.is_running = True
        self.animation = animation.FuncAnimation(
            self.fig, self.animate_update, interval=int(self.update_interval * 1000),
            blit=False, cache_frame_data=False
        )
        
        plt.show()
        logger_manager.info("轨迹可视化动画已启动")
    
    def stop_animation(self) -> None:
        """
        停止动画显示
        """
        self.is_running = False
        if self.animation:
            self.animation.event_source.stop()
        logger_manager.info("轨迹可视化动画已停止")
    
    def save_trajectory_plot(self, filename: str) -> None:
        """
        保存轨迹图
        
        Args:
            filename: 保存文件名
        """
        if self.fig:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            logger_manager.info(f"轨迹图已保存: {filename}")


class SensorDataVisualizer:
    """传感器数据可视化器
    
    实时显示传感器数据和可靠性指标
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化传感器数据可视化器
        
        Args:
            config: 可视化配置
        """
        self.config = config
        self.max_points = config.get('max_sensor_points', 500)
        
        # 传感器数据存储
        self.sensor_data_history = {
            'imu': deque(maxlen=self.max_points),
            'gps': deque(maxlen=self.max_points),
            'optical': deque(maxlen=self.max_points)
        }
        
        # 可靠性数据存储
        self.reliability_history = {
            'imu': deque(maxlen=self.max_points),
            'gps': deque(maxlen=self.max_points),
            'optical': deque(maxlen=self.max_points),
            'fusion': deque(maxlen=self.max_points)
        }
        
        self.timestamps = deque(maxlen=self.max_points)
        
        logger_manager.info("传感器数据可视化器初始化完成")
    
    def create_sensor_dashboard(self) -> None:
        """
        创建传感器仪表板
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # IMU数据图
        axes[0, 0].set_title('IMU Acceleration')
        axes[0, 0].set_ylabel('Acceleration (m/s²)')
        axes[0, 0].grid(True)
        
        # GPS数据图
        axes[0, 1].set_title('GPS Position Accuracy')
        axes[0, 1].set_ylabel('HDOP')
        axes[0, 1].grid(True)
        
        # 光学匹配图
        axes[0, 2].set_title('Optical Match Score')
        axes[0, 2].set_ylabel('Match Score')
        axes[0, 2].grid(True)
        
        # 传感器可靠性图
        axes[1, 0].set_title('Sensor Reliability')
        axes[1, 0].set_ylabel('Reliability Score')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].grid(True)
        
        # 融合权重图
        axes[1, 1].set_title('Sensor Weights')
        axes[1, 1].set_ylabel('Weight')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].grid(True)
        
        # 系统状态图
        axes[1, 2].set_title('System Status')
        axes[1, 2].set_ylabel('Status')
        axes[1, 2].set_xlabel('Time (s)')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        return fig, axes
    
    @performance_monitor
    def update_sensor_data(self, system_state: SystemState, 
                          reliability_metrics: Optional[ReliabilityMetrics] = None) -> None:
        """
        更新传感器数据
        
        Args:
            system_state: 系统状态
            reliability_metrics: 可靠性指标
        """
        current_time = time.time()
        self.timestamps.append(current_time)
        
        # 更新传感器数据历史
        if hasattr(system_state, 'imu_data') and system_state.imu_data:
            self.sensor_data_history['imu'].append(system_state.imu_data)
        
        if hasattr(system_state, 'gps_data') and system_state.gps_data:
            self.sensor_data_history['gps'].append(system_state.gps_data)
        
        if hasattr(system_state, 'optical_data') and system_state.optical_data:
            self.sensor_data_history['optical'].append(system_state.optical_data)
        
        # 更新可靠性历史
        if reliability_metrics:
            self.reliability_history['imu'].append(reliability_metrics.imu_reliability)
            self.reliability_history['gps'].append(reliability_metrics.gps_reliability)
            self.reliability_history['optical'].append(reliability_metrics.optical_reliability)
            self.reliability_history['fusion'].append(reliability_metrics.fusion_reliability)


class PerformanceAnalyzer:
    """性能分析器
    
    分析和可视化系统性能指标
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化性能分析器
        
        Args:
            config: 分析配置
        """
        self.config = config
        self.performance_data = {
            'position_errors': [],
            'velocity_errors': [],
            'processing_times': [],
            'memory_usage': [],
            'cpu_usage': []
        }
        
        logger_manager.info("性能分析器初始化完成")
    
    def analyze_trajectory_accuracy(self, true_trajectory: List[Position3D],
                                  estimated_trajectory: List[Position3D]) -> Dict[str, float]:
        """
        分析轨迹精度
        
        Args:
            true_trajectory: 真实轨迹
            estimated_trajectory: 估计轨迹
            
        Returns:
            精度分析结果
        """
        if len(true_trajectory) != len(estimated_trajectory):
            logger_manager.warning("轨迹长度不匹配，无法进行精度分析")
            return {}
        
        errors = []
        for true_pos, est_pos in zip(true_trajectory, estimated_trajectory):
            error = np.sqrt(
                (true_pos.x - est_pos.x) ** 2 +
                (true_pos.y - est_pos.y) ** 2 +
                (true_pos.z - est_pos.z) ** 2
            )
            errors.append(error)
        
        analysis = {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'max_error': np.max(errors),
            'min_error': np.min(errors),
            'rmse': np.sqrt(np.mean(np.array(errors) ** 2)),
            'percentile_95': np.percentile(errors, 95)
        }
        
        return analysis
    
    def create_performance_report(self) -> str:
        """
        创建性能报告
        
        Returns:
            性能报告字符串
        """
        report = "\n=== 系统性能报告 ===\n"
        
        if self.performance_data['position_errors']:
            pos_errors = self.performance_data['position_errors']
            report += f"位置误差统计:\n"
            report += f"  平均误差: {np.mean(pos_errors):.3f} m\n"
            report += f"  标准差: {np.std(pos_errors):.3f} m\n"
            report += f"  最大误差: {np.max(pos_errors):.3f} m\n"
            report += f"  RMSE: {np.sqrt(np.mean(np.array(pos_errors) ** 2)):.3f} m\n"
        
        if self.performance_data['processing_times']:
            proc_times = self.performance_data['processing_times']
            report += f"\n处理时间统计:\n"
            report += f"  平均处理时间: {np.mean(proc_times):.3f} ms\n"
            report += f"  最大处理时间: {np.max(proc_times):.3f} ms\n"
            report += f"  处理频率: {1000.0 / np.mean(proc_times):.1f} Hz\n"
        
        return report
    
    def plot_performance_metrics(self) -> None:
        """
        绘制性能指标图表
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 位置误差分布
        if self.performance_data['position_errors']:
            axes[0, 0].hist(self.performance_data['position_errors'], bins=50, alpha=0.7)
            axes[0, 0].set_title('Position Error Distribution')
            axes[0, 0].set_xlabel('Error (m)')
            axes[0, 0].set_ylabel('Frequency')
        
        # 处理时间分布
        if self.performance_data['processing_times']:
            axes[0, 1].hist(self.performance_data['processing_times'], bins=50, alpha=0.7)
            axes[0, 1].set_title('Processing Time Distribution')
            axes[0, 1].set_xlabel('Time (ms)')
            axes[0, 1].set_ylabel('Frequency')
        
        # 内存使用趋势
        if self.performance_data['memory_usage']:
            axes[1, 0].plot(self.performance_data['memory_usage'])
            axes[1, 0].set_title('Memory Usage Trend')
            axes[1, 0].set_xlabel('Time')
            axes[1, 0].set_ylabel('Memory (MB)')
        
        # CPU使用趋势
        if self.performance_data['cpu_usage']:
            axes[1, 1].plot(self.performance_data['cpu_usage'])
            axes[1, 1].set_title('CPU Usage Trend')
            axes[1, 1].set_xlabel('Time')
            axes[1, 1].set_ylabel('CPU (%)')
        
        plt.tight_layout()
        plt.show()


class VisualizationManager:
    """可视化管理器
    
    统一管理所有可视化组件
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化可视化管理器
        
        Args:
            config: 可视化配置
        """
        self.config = config
        
        # 初始化各个可视化器
        self.trajectory_visualizer = TrajectoryVisualizer(config.get('trajectory', {}))
        self.sensor_visualizer = SensorDataVisualizer(config.get('sensor', {}))
        self.performance_analyzer = PerformanceAnalyzer(config.get('performance', {}))
        
        # 可视化状态
        self.is_running = False
        self.update_thread = None
        self.data_queue = queue.Queue()
        
        logger_manager.info("可视化管理器初始化完成")
    
    @performance_monitor
    def update_all_visualizations(self, system_state: SystemState,
                                reliability_metrics: Optional[ReliabilityMetrics] = None) -> None:
        """
        更新所有可视化
        
        Args:
            system_state: 系统状态
            reliability_metrics: 可靠性指标
        """
        # 更新轨迹可视化
        self.trajectory_visualizer.update_trajectory(system_state)
        
        # 更新传感器数据可视化
        self.sensor_visualizer.update_sensor_data(system_state, reliability_metrics)
        
        # 更新性能数据
        if hasattr(system_state, 'position_error'):
            self.performance_analyzer.performance_data['position_errors'].append(
                system_state.position_error
            )
        
        if hasattr(system_state, 'processing_time'):
            self.performance_analyzer.performance_data['processing_times'].append(
                system_state.processing_time * 1000  # 转换为毫秒
            )
    
    def start_real_time_visualization(self) -> None:
        """
        启动实时可视化
        """
        self.is_running = True
        
        # 启动轨迹动画
        self.trajectory_visualizer.start_animation()
        
        logger_manager.info("实时可视化已启动")
    
    def stop_real_time_visualization(self) -> None:
        """
        停止实时可视化
        """
        self.is_running = False
        
        # 停止轨迹动画
        self.trajectory_visualizer.stop_animation()
        
        logger_manager.info("实时可视化已停止")
    
    def generate_final_report(self, output_dir: str) -> None:
        """
        生成最终报告
        
        Args:
            output_dir: 输出目录
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存轨迹图
        trajectory_file = os.path.join(output_dir, 'trajectory_analysis.png')
        self.trajectory_visualizer.save_trajectory_plot(trajectory_file)
        
        # 保存性能图表
        performance_file = os.path.join(output_dir, 'performance_metrics.png')
        self.performance_analyzer.plot_performance_metrics()
        plt.savefig(performance_file, dpi=300, bbox_inches='tight')
        
        # 生成文本报告
        report_file = os.path.join(output_dir, 'performance_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(self.performance_analyzer.create_performance_report())
        
        logger_manager.info(f"最终报告已生成到: {output_dir}")
    
    def clear_all_data(self) -> None:
        """
        清除所有可视化数据
        """
        # 清除轨迹数据
        self.trajectory_visualizer.true_trajectory.clear()
        self.trajectory_visualizer.fusion_trajectory.clear()
        self.trajectory_visualizer.gps_trajectory.clear()
        self.trajectory_visualizer.optical_trajectory.clear()
        self.trajectory_visualizer.timestamps.clear()
        
        # 清除传感器数据
        for sensor_type in self.sensor_visualizer.sensor_data_history:
            self.sensor_visualizer.sensor_data_history[sensor_type].clear()
            self.sensor_visualizer.reliability_history[sensor_type].clear()
        
        # 清除性能数据
        for metric in self.performance_analyzer.performance_data:
            self.performance_analyzer.performance_data[metric].clear()
        
        logger_manager.info("所有可视化数据已清除")