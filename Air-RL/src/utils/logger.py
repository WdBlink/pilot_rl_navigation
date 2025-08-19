#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强化学习无人机定位导航系统 - 日志工具模块

本模块提供统一的日志管理功能，基于loguru库实现高性能的结构化日志记录。
支持多种日志输出格式、日志轮转、性能监控等功能。

Author: wdblink
Date: 2024
"""

import sys
import time
import functools
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from loguru import logger
import json


class PerformanceLogger:
    """性能监控日志器类
    
    用于监控函数执行时间和系统性能指标。
    """
    
    def __init__(self):
        """初始化性能日志器"""
        self.metrics: Dict[str, list] = {}
    
    def log_execution_time(self, func_name: str, execution_time: float) -> None:
        """记录函数执行时间
        
        Args:
            func_name: 函数名称
            execution_time: 执行时间（秒）
        """
        if func_name not in self.metrics:
            self.metrics[func_name] = []
        
        self.metrics[func_name].append(execution_time)
        
        # 记录到日志
        logger.debug(
            f"性能监控 | 函数: {func_name} | 执行时间: {execution_time:.4f}s"
        )
    
    def get_average_time(self, func_name: str) -> Optional[float]:
        """获取函数平均执行时间
        
        Args:
            func_name: 函数名称
            
        Returns:
            平均执行时间，如果没有记录则返回None
        """
        if func_name not in self.metrics or not self.metrics[func_name]:
            return None
        
        return sum(self.metrics[func_name]) / len(self.metrics[func_name])
    
    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """获取性能摘要
        
        Returns:
            包含各函数性能统计的字典
        """
        summary = {}
        
        for func_name, times in self.metrics.items():
            if times:
                summary[func_name] = {
                    'count': len(times),
                    'total_time': sum(times),
                    'average_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times)
                }
        
        return summary
    
    def reset_metrics(self) -> None:
        """重置性能指标"""
        self.metrics.clear()
        logger.info("性能指标已重置")


class StructuredLogger:
    """结构化日志器类
    
    提供结构化的日志记录功能，支持JSON格式输出和自定义字段。
    """
    
    def __init__(self, component_name: str):
        """初始化结构化日志器
        
        Args:
            component_name: 组件名称
        """
        self.component_name = component_name
        self.context: Dict[str, Any] = {}
    
    def set_context(self, **kwargs) -> None:
        """设置日志上下文
        
        Args:
            **kwargs: 上下文键值对
        """
        self.context.update(kwargs)
    
    def clear_context(self) -> None:
        """清除日志上下文"""
        self.context.clear()
    
    def log_event(self, level: str, event: str, **kwargs) -> None:
        """记录结构化事件
        
        Args:
            level: 日志级别
            event: 事件名称
            **kwargs: 事件数据
        """
        log_data = {
            'component': self.component_name,
            'event': event,
            'timestamp': time.time(),
            **self.context,
            **kwargs
        }
        
        message = f"[{self.component_name}] {event}"
        if kwargs:
            message += f" | 数据: {json.dumps(kwargs, ensure_ascii=False)}"
        
        getattr(logger, level.lower())(message)
    
    def log_state_change(self, from_state: str, to_state: str, **kwargs) -> None:
        """记录状态变化
        
        Args:
            from_state: 原状态
            to_state: 新状态
            **kwargs: 附加数据
        """
        self.log_event(
            'info',
            'state_change',
            from_state=from_state,
            to_state=to_state,
            **kwargs
        )
    
    def info(self, message: str, **kwargs) -> None:
        """记录信息级别日志
        
        Args:
            message: 日志消息
            **kwargs: 附加数据
        """
        self.log_event('info', message, **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """记录调试级别日志
        
        Args:
            message: 日志消息
            **kwargs: 附加数据
        """
        self.log_event('debug', message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """记录警告级别日志
        
        Args:
            message: 日志消息
            **kwargs: 附加数据
        """
        self.log_event('warning', message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """记录错误级别日志
        
        Args:
            message: 日志消息
            **kwargs: 附加数据
        """
        self.log_event('error', message, **kwargs)
    
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """记录错误信息
        
        Args:
            error: 异常对象
            context: 错误上下文
        """
        error_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'component': self.component_name
        }
        
        if context:
            error_data.update(context)
        
        logger.error(f"[{self.component_name}] 错误: {error} | 上下文: {json.dumps(error_data, ensure_ascii=False)}")


class LoggerManager:
    """日志管理器类
    
    统一管理系统的日志配置和输出。
    """
    
    def __init__(self):
        """初始化日志管理器"""
        self.performance_logger = PerformanceLogger()
        self.structured_loggers: Dict[str, StructuredLogger] = {}
        self.is_initialized = False
    
    def initialize(self, 
                  log_level: str = "INFO",
                  log_file: Optional[str] = None,
                  max_file_size: str = "10 MB",
                  backup_count: int = 5,
                  log_format: Optional[str] = None) -> None:
        """初始化日志系统
        
        Args:
            log_level: 日志级别
            log_file: 日志文件路径
            max_file_size: 最大文件大小
            backup_count: 备份文件数量
            log_format: 日志格式
        """
        if self.is_initialized:
            logger.warning("日志系统已经初始化")
            return
        
        # 移除默认处理器
        logger.remove()
        
        # 设置默认格式
        if log_format is None:
            log_format = (
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            )
        
        # 添加控制台输出
        logger.add(
            sys.stdout,
            level=log_level,
            format=log_format,
            colorize=True,
            backtrace=True,
            diagnose=True
        )
        
        # 添加文件输出
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.add(
                log_file,
                level=log_level,
                format=log_format,
                rotation=max_file_size,
                retention=backup_count,
                compression="zip",
                backtrace=True,
                diagnose=True,
                encoding="utf-8"
            )
        
        # 添加JSON格式的结构化日志文件
        if log_file:
            json_log_file = log_path.with_suffix('.json')
            logger.add(
                json_log_file,
                level="DEBUG",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
                rotation=max_file_size,
                retention=backup_count,
                serialize=True,
                backtrace=True,
                diagnose=True,
                encoding="utf-8"
            )
        
        self.is_initialized = True
        logger.info("日志系统初始化完成")
    
    def get_structured_logger(self, component_name: str) -> StructuredLogger:
        """获取结构化日志器
        
        Args:
            component_name: 组件名称
            
        Returns:
            结构化日志器实例
        """
        if component_name not in self.structured_loggers:
            self.structured_loggers[component_name] = StructuredLogger(component_name)
        
        return self.structured_loggers[component_name]
    
    def get_performance_logger(self) -> PerformanceLogger:
        """获取性能日志器
        
        Returns:
            性能日志器实例
        """
        return self.performance_logger
    
    def log_system_info(self) -> None:
        """记录系统信息"""
        import platform
        import psutil
        
        system_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
            'disk_usage': f"{psutil.disk_usage('/').percent:.1f}%"
        }
        
        logger.info(f"系统信息: {json.dumps(system_info, ensure_ascii=False)}")
    
    def create_performance_report(self) -> str:
        """创建性能报告
        
        Returns:
            性能报告字符串
        """
        summary = self.performance_logger.get_performance_summary()
        
        if not summary:
            return "暂无性能数据"
        
        report_lines = ["=== 性能报告 ==="]
        
        for func_name, stats in summary.items():
            report_lines.extend([
                f"\n函数: {func_name}",
                f"  调用次数: {stats['count']}",
                f"  总时间: {stats['total_time']:.4f}s",
                f"  平均时间: {stats['average_time']:.4f}s",
                f"  最小时间: {stats['min_time']:.4f}s",
                f"  最大时间: {stats['max_time']:.4f}s"
            ])
        
        report = "\n".join(report_lines)
        logger.info(report)
        
        return report


# 全局日志管理器实例
logger_manager = LoggerManager()


def performance_monitor(func: Callable) -> Callable:
    """性能监控装饰器
    
    用于自动监控函数执行时间。
    
    Args:
        func: 被装饰的函数
        
    Returns:
        装饰后的函数
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            execution_time = time.time() - start_time
            logger_manager.get_performance_logger().log_execution_time(
                func.__name__, execution_time
            )
    
    return wrapper


def log_function_call(component: str = "system"):
    """函数调用日志装饰器
    
    记录函数的调用和返回。
    
    Args:
        component: 组件名称
        
    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            structured_logger = logger_manager.get_structured_logger(component)
            
            # 记录函数调用
            structured_logger.log_event(
                'debug',
                'function_call',
                function=func.__name__,
                args_count=len(args),
                kwargs_keys=list(kwargs.keys())
            )
            
            try:
                result = func(*args, **kwargs)
                
                # 记录函数返回
                structured_logger.log_event(
                    'debug',
                    'function_return',
                    function=func.__name__,
                    success=True
                )
                
                return result
                
            except Exception as e:
                # 记录函数异常
                structured_logger.log_error(e, {
                    'function': func.__name__,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys())
                })
                raise
        
        return wrapper
    return decorator


def initialize_logging(config: Optional[Dict[str, Any]] = None) -> None:
    """初始化日志系统
    
    Args:
        config: 日志配置字典
    """
    if config is None:
        config = {
            'log_level': 'INFO',
            'log_file': 'logs/pilot_rl_navigation.log',
            'max_file_size': '10 MB',
            'backup_count': 5
        }
    
    logger_manager.initialize(**config)
    logger_manager.log_system_info()


def get_logger(component_name: str) -> StructuredLogger:
    """获取组件日志器
    
    Args:
        component_name: 组件名称
        
    Returns:
        结构化日志器实例
    """
    return logger_manager.get_structured_logger(component_name)


def get_performance_logger() -> PerformanceLogger:
    """获取性能日志器
    
    Returns:
        性能日志器实例
    """
    return logger_manager.get_performance_logger()


def create_performance_report() -> str:
    """创建性能报告
    
    Returns:
        性能报告字符串
    """
    return logger_manager.create_performance_report()


def setup_logger(component_name: str, log_level: str = "INFO") -> StructuredLogger:
    """设置并获取组件日志器
    
    Args:
        component_name: 组件名称
        log_level: 日志级别
        
    Returns:
        结构化日志器实例
    """
    # 初始化日志系统（如果尚未初始化）
    config = {
        'log_level': log_level,
        'log_file': f'logs/{component_name}.log',
        'max_file_size': '10 MB',
        'backup_count': 5
    }
    
    try:
        initialize_logging(config)
    except Exception:
        # 如果已经初始化，忽略错误
        pass
    
    return get_logger(component_name)