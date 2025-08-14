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

# 读取requirements.txt
def read_requirements(filename):
    """读取依赖文件"""
    requirements_file = HERE / filename
    if requirements_file.exists():
        with open(requirements_file, "r", encoding="utf-8") as f:
            return [
                line.strip() 
                for line in f.readlines() 
                if line.strip() and not line.startswith("#")
            ]
    return []

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

# GPU支持依赖
gpu_requires = [
    "torch>=2.0.0+cu118",
    "torchvision>=0.15.0+cu118",
]

# 硬件接口依赖
hardware_requires = [
    "dronekit>=2.9.0",
    "pyserial>=3.5",
    "RPi.GPIO>=0.7.0; platform_machine=='armv7l'",
]

# 获取版本信息
def get_version():
    """从版本文件获取版本号"""
    version_file = HERE / "src" / "utils" / "__init__.py"
    if version_file.exists():
        with open(version_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    return "1.0.0"

# 获取项目文件
def get_package_data():
    """获取包数据文件"""
    package_data = {
        "pilot_rl_navigation": [
            "config/*.yaml",
            "config/*.json",
            "data/maps/*.json",
            "data/models/*.pth",
            "docs/*.md",
            "scripts/*.py",
        ]
    }
    return package_data

# 获取数据文件
def get_data_files():
    """获取数据文件列表"""
    data_files = []
    
    # 配置文件
    config_files = []
    config_dir = HERE / "config"
    if config_dir.exists():
        for config_file in config_dir.glob("*.yaml"):
            config_files.append(str(config_file))
        for config_file in config_dir.glob("*.json"):
            config_files.append(str(config_file))
    
    if config_files:
        data_files.append(("config", config_files))
    
    # 文档文件
    doc_files = []
    docs_dir = HERE / "docs"
    if docs_dir.exists():
        for doc_file in docs_dir.glob("*.md"):
            doc_files.append(str(doc_file))
    
    if doc_files:
        data_files.append(("docs", doc_files))
    
    return data_files

# 命令行入口点
entry_points = {
    "console_scripts": [
        "rl-drone-train=src.scripts.train_rl_agent:main",
        "rl-drone-eval=src.scripts.evaluate_model:main",
        "rl-drone-demo=src.scripts.full_navigation_demo:main",
        "rl-drone-monitor=src.scripts.monitoring_dashboard:main",
        "rl-drone-collect=src.scripts.data_collection:main",
        "rl-drone-deploy=src.scripts.deployment:main",
    ],
}

# 分类器
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

# 项目URL
project_urls = {
    "Homepage": "https://github.com/wdblink/pilot_rl_navigation",
    "Bug Reports": "https://github.com/wdblink/pilot_rl_navigation/issues",
    "Source": "https://github.com/wdblink/pilot_rl_navigation",
    "Documentation": "https://pilot-rl-navigation.readthedocs.io/",
    "Changelog": "https://github.com/wdblink/pilot_rl_navigation/blob/main/CHANGELOG.md",
}

# 主要安装配置
setup(
    # 基本信息
    name="pilot-rl-navigation",
    version=get_version(),
    author="wdblink",
    author_email="wdblink@example.com",
    description="基于强化学习的无人机智能定位导航系统",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wdblink/pilot_rl_navigation",
    project_urls=project_urls,
    
    # 包配置
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data=get_package_data(),
    data_files=get_data_files(),
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
    
    # 入口点
    entry_points=entry_points,
    
    # 元数据
    classifiers=classifiers,
    keywords=" ".join(keywords),
    license="MIT",
    platforms=["any"],
    
    # 其他配置
    zip_safe=False,
    test_suite="tests",
    tests_require=development_requires,
    
    # 命令行选项
    options={
        "build_scripts": {
            "executable": "/usr/bin/python3",
        },
        "egg_info": {
            "tag_build": "",
            "tag_date": False,
        },
    },
)

# 安装后处理
class PostInstallCommand:
    """安装后执行的命令"""
    
    @staticmethod
    def create_directories():
        """创建必要的目录"""
        directories = [
            "logs",
            "models/checkpoints",
            "models/pretrained",
            "data/training",
            "data/validation",
            "data/maps",
        ]
        
        for directory in directories:
            dir_path = HERE / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"创建目录: {dir_path}")
    
    @staticmethod
    def setup_git_hooks():
        """设置Git钩子"""
        git_dir = HERE / ".git"
        if git_dir.exists():
            try:
                import subprocess
                subprocess.run(["pre-commit", "install"], cwd=HERE, check=True)
                print("Git pre-commit钩子安装成功")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("警告: 无法安装pre-commit钩子")
    
    @staticmethod
    def verify_installation():
        """验证安装"""
        try:
            import torch
            import cv2
            import numpy as np
            print("✓ 核心依赖验证成功")
            
            # 检查CUDA支持
            if torch.cuda.is_available():
                print(f"✓ CUDA支持: {torch.cuda.get_device_name(0)}")
            else:
                print("⚠ CUDA不可用，将使用CPU模式")
                
        except ImportError as e:
            print(f"✗ 依赖验证失败: {e}")
            sys.exit(1)
    
    @classmethod
    def run_post_install(cls):
        """运行安装后处理"""
        print("\n=== 强化学习无人机导航系统安装后配置 ===")
        
        cls.create_directories()
        cls.setup_git_hooks()
        cls.verify_installation()
        
        print("\n=== 安装完成 ===")
        print("使用 'rl-drone-demo --help' 查看可用命令")
        print("查看 README.md 了解详细使用说明")
        print("项目主页: https://github.com/wdblink/pilot_rl_navigation")

# 如果直接运行此文件，执行安装后处理
if __name__ == "__main__" and "install" in sys.argv:
    PostInstallCommand.run_post_install()