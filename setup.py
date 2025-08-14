#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强化学习无人机定位导航系统 - 安装配置文件

本文件定义了项目的安装配置，包括：
1. 项目基本信息和元数据
2. 依赖包管理
3. 入口点和命令行工具
4. 数据文件和资源管理
5. 开发和测试依赖

Author: wdblink
Date: 2024
"""

import os
import sys
from setuptools import setup, find_packages
from pathlib import Path

# 确保Python版本兼容性
if sys.version_info < (3, 8):
    raise RuntimeError("此项目需要Python 3.8或更高版本")

# 获取项目根目录
HERE = Path(__file__).parent.absolute()

# 读取README文件
with open(HERE / "README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# 基础依赖
install_requires = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "stable-baselines3>=2.0.0",
    "gymnasium>=0.28.0",
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    "opencv-python>=4.5.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "airsim>=1.8.1",
    "msgpack-rpc-python>=0.4.1",
    "pymavlink>=2.4.0",
    "pandas>=1.3.0",
    "pyyaml>=6.0",
    "loguru>=0.6.0",
    "psutil>=5.8.0",
    "tqdm>=4.62.0",
    "click>=8.0.0",
    "rich>=10.0.0",
    "pydantic>=1.9.0",
]

# 开发依赖
development_requires = [
    "pytest>=7.0.0",
    "pytest-cov>=3.0.0",
    "pytest-mock>=3.6.0",
    "black>=22.0.0",
    "flake8>=4.0.0",
    "isort>=5.10.0",
    "mypy>=0.950",
    "pre-commit>=2.15.0",
    "sphinx>=4.0.0",
    "sphinx-rtd-theme>=1.0.0",
    "jupyter>=1.0.0",
    "notebook>=6.0.0",
    "ipywidgets>=7.6.0",
]

# 可视化依赖
visualization_requires = [
    "plotly>=5.0.0",
    "bokeh>=2.4.0",
    "dash>=2.0.0",
    "streamlit>=1.0.0",
    "tensorboard>=2.8.0",
]

# GPU依赖
gpu_requires = [
    "torch>=2.0.0+cu118",
    "torchvision>=0.15.0+cu118",
]

# 硬件依赖
hardware_requires = [
    "dronekit>=2.9.0",
    "pyserial>=3.5",
    "RPi.GPIO>=0.7.0; platform_machine=='armv7l'",
]

def get_version():
    """获取版本号"""
    version_file = HERE / "src" / "__init__.py"
    if version_file.exists():
        with open(version_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    return "1.0.0"

# 项目分类
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Image Recognition",
    "Topic :: System :: Hardware :: Hardware Drivers",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

# 关键词
keywords = [
    "reinforcement learning",
    "drone navigation",
    "computer vision",
    "sensor fusion",
    "autonomous systems",
    "robotics",
    "airsim",
    "pytorch",
    "deep learning",
    "path planning",
    "optical positioning",
    "multi-sensor",
]

# 项目链接
project_urls = {
    "Homepage": "https://github.com/WdBlink/pilot_rl_navigation",
    "Bug Reports": "https://github.com/WdBlink/pilot_rl_navigation/issues",
    "Source": "https://github.com/WdBlink/pilot_rl_navigation",
    "Documentation": "https://pilot-rl-navigation.readthedocs.io/",
    "Changelog": "https://github.com/WdBlink/pilot_rl_navigation/blob/main/CHANGELOG.md",
}

# 安装配置
setup(
    # 基本信息
    name="pilot-rl-navigation",
    version=get_version(),
    author="wdblink",
    author_email="wdblink@example.com",
    description="基于强化学习的无人机智能定位导航系统",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/WdBlink/pilot_rl_navigation",
    project_urls=project_urls,
    
    # 包配置
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    
    # 依赖配置
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require={
        "dev": development_requires,
        "viz": visualization_requires,
        "gpu": gpu_requires,
        "hardware": hardware_requires,
        "all": development_requires + visualization_requires + hardware_requires,
    },
    
    # 元数据
    classifiers=classifiers,
    keywords=" ".join(keywords),
    license="MIT",
    platforms=["any"],
    
    # 其他配置
    zip_safe=False,
    test_suite="tests",
    tests_require=development_requires,
)