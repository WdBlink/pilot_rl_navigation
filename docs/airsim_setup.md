# AirSim训练环境搭建指南

本文档详细介绍如何搭建和使用AirSim训练环境进行无人机强化学习训练。

## 目录

- [环境要求](#环境要求)
- [安装步骤](#安装步骤)
- [配置说明](#配置说明)
- [使用方法](#使用方法)
- [API参考](#api参考)
- [故障排除](#故障排除)

## 环境要求

### 系统要求

- **操作系统**: Windows 10/11, Ubuntu 18.04+, macOS 10.14+
- **Python**: 3.7+
- **内存**: 8GB+ (推荐16GB)
- **显卡**: 支持DirectX 11的独立显卡 (推荐GTX 1060+)
- **存储**: 10GB+ 可用空间

### 软件依赖

- Unreal Engine 4.27+ (用于AirSim仿真)
- Visual Studio 2019+ (Windows)
- CMake 3.10+
- Git

## 安装步骤

### 1. 安装AirSim

#### Windows安装

```bash
# 克隆AirSim仓库
git clone https://github.com/Microsoft/AirSim.git
cd AirSim

# 构建AirSim
build.cmd

# 安装Python包
pip install airsim
```

#### Linux安装

```bash
# 安装依赖
sudo apt-get update
sudo apt-get install build-essential cmake git

# 克隆AirSim仓库
git clone https://github.com/Microsoft/AirSim.git
cd AirSim

# 构建AirSim
./setup.sh
./build.sh

# 安装Python包
pip install airsim
```

### 2. 下载预构建环境

从[AirSim发布页面](https://github.com/Microsoft/AirSim/releases)下载预构建的环境，推荐使用:

- **Neighborhood**: 适合基础训练
- **CityEnviron**: 适合复杂场景训练
- **LandscapeMountains**: 适合地形导航训练

### 3. 安装项目依赖

```bash
# 进入项目目录
cd /path/to/pilot_rl_navigation

# 安装Python依赖
pip install -r requirements.txt

# 安装额外的AirSim依赖
pip install airsim opencv-python matplotlib numpy scipy
```

## 配置说明

### AirSim设置文件

在用户目录下创建`settings.json`文件:

**Windows**: `%USERPROFILE%\Documents\AirSim\settings.json`
**Linux**: `~/Documents/AirSim/settings.json`

```json
{
  "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/master/docs/settings.md",
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "ClockSpeed": 1.0,
  "Vehicles": {
    "Drone1": {
      "VehicleType": "SimpleFlight",
      "X": 0, "Y": 0, "Z": -2,
      "Pitch": 0, "Roll": 0, "Yaw": 0
    }
  },
  "CameraDefaults": {
    "CaptureSettings": [
      {
        "ImageType": 0,
        "Width": 640,
        "Height": 480,
        "FOV_Degrees": 90
      }
    ]
  },
  "Recording": {
    "RecordOnMove": false,
    "RecordInterval": 0.05
  }
}
```

### 项目配置文件

编辑`configs/airsim_config.yaml`文件，根据您的环境调整以下参数:

```yaml
system:
  airsim:
    connection:
      ip: "127.0.0.1"  # AirSim服务器IP
      port: 41451      # API端口
      vehicle_name: "Drone1"  # 无人机名称
    
    origin:
      latitude: 47.641468   # 设置为您的地理原点
      longitude: -122.140165
      altitude: 122.0
```

## 使用方法

### 1. 启动AirSim环境

1. 运行下载的AirSim环境可执行文件
2. 选择"Computer Vision"或"Car"模式
3. 等待环境加载完成

### 2. 运行训练示例

```bash
# 进入项目目录
cd /path/to/pilot_rl_navigation

# 运行AirSim训练示例
python scripts/airsim_training_example.py
```

### 3. 选择演示模式

程序会提示选择演示模式:

- **基础飞行演示**: 展示环境初始化、飞行计划创建和执行
- **训练环境演示**: 展示强化学习训练流程

### 4. 查看结果

训练完成后，结果将保存在`output/`目录下:

- `flight_analysis.png`: 飞行轨迹分析图
- `collected_images.png`: 采集的图像样本
- 日志文件: 详细的训练日志

## API参考

### AirSimEnvironment类

主要的AirSim环境接口类。

#### 初始化

```python
from src.environment.airsim_env import AirSimEnvironment
from src.utils.config import load_config

config = load_config()
env = AirSimEnvironment(config.airsim)
```

#### 主要方法

```python
# 初始化环境
success = await env.initialize()

# 创建飞行计划
waypoints = [(lat1, lon1, alt1), (lat2, lon2, alt2), ...]
flight_plan = env.create_flight_plan(waypoints, name="mission1")

# 执行飞行计划
success = await env.execute_flight_plan(flight_plan)

# 获取传感器数据
sensor_data = await env._collect_sensor_data()

# 关闭环境
await env.shutdown()
```

### AirSimTrainingEnvironment类

用于强化学习训练的环境包装器。

```python
from src.environment.airsim_env import AirSimTrainingEnvironment

# 创建训练环境
training_env = AirSimTrainingEnvironment(config.airsim)

# 重置环境
observation = await training_env.reset()

# 执行动作
action = {"thrust": 0.5, "roll": 0.1, "pitch": 0.0, "yaw_rate": 0.0}
observation, reward, done, info = await training_env.step(action)
```

### 数据结构

#### SensorData

```python
@dataclass
class SensorData:
    timestamp: float
    position: Position3D
    attitude: Attitude
    velocity: Velocity3D
    camera_image: Optional[np.ndarray]
    satellite_image: Optional[np.ndarray]
    imu_data: Optional[dict]
    gps_data: Optional[dict]
```

#### FlightPlan

```python
@dataclass
class FlightPlan:
    name: str
    waypoints: List[Waypoint]
    created_at: datetime
    estimated_duration: float
```

## 故障排除

### 常见问题

#### 1. 连接失败

**问题**: `ConnectionError: Could not connect to AirSim`

**解决方案**:
- 确保AirSim环境正在运行
- 检查IP地址和端口配置
- 确认防火墙设置

```python
# 测试连接
import airsim
client = airsim.MultirotorClient()
client.confirmConnection()
```

#### 2. 图像采集失败

**问题**: 相机图像为空或损坏

**解决方案**:
- 检查相机配置
- 确认相机名称正确
- 验证图像类型设置

```python
# 测试相机
response = client.simGetImage("front_center", airsim.ImageType.Scene)
if response:
    print(f"图像大小: {len(response)}")
else:
    print("图像采集失败")
```

#### 3. 飞行控制异常

**问题**: 无人机不响应控制命令

**解决方案**:
- 检查无人机是否已解锁
- 确认API控制已启用
- 验证飞行模式设置

```python
# 解锁并启用API控制
client.enableApiControl(True)
client.armDisarm(True)
```

#### 4. 性能问题

**问题**: 仿真运行缓慢

**解决方案**:
- 降低图像分辨率
- 减少仿真复杂度
- 调整时钟速度

```json
// 在settings.json中调整
{
  "ClockSpeed": 2.0,  // 加速仿真
  "CameraDefaults": {
    "CaptureSettings": [
      {
        "Width": 320,    // 降低分辨率
        "Height": 240
      }
    ]
  }
}
```

### 调试技巧

#### 1. 启用详细日志

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### 2. 可视化调试

```python
# 显示实时图像
import cv2

while True:
    response = client.simGetImage("front_center", airsim.ImageType.Scene)
    if response:
        img = cv2.imdecode(np.frombuffer(response, np.uint8), cv2.IMREAD_COLOR)
        cv2.imshow("AirSim Camera", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
```

#### 3. 状态监控

```python
# 监控无人机状态
state = client.getMultirotorState()
print(f"位置: {state.kinematics_estimated.position}")
print(f"速度: {state.kinematics_estimated.linear_velocity}")
print(f"姿态: {state.kinematics_estimated.orientation}")
```

## 进阶使用

### 自定义环境

1. **创建自定义场景**: 使用Unreal Engine编辑器
2. **添加自定义传感器**: 修改AirSim插件
3. **实现自定义物理模型**: 扩展SimpleFlight控制器

### 分布式训练

```python
# 多环境并行训练
from multiprocessing import Pool

def train_worker(env_id):
    env = AirSimTrainingEnvironment(config, env_id=env_id)
    # 训练逻辑
    return results

# 启动多个训练进程
with Pool(processes=4) as pool:
    results = pool.map(train_worker, range(4))
```

### 云端部署

使用Docker容器化部署:

```dockerfile
FROM nvidia/cuda:11.0-devel-ubuntu20.04

# 安装依赖
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    libgl1-mesa-glx libglib2.0-0

# 复制项目文件
COPY . /app
WORKDIR /app

# 安装Python依赖
RUN pip3 install -r requirements.txt

# 启动训练
CMD ["python3", "scripts/airsim_training_example.py"]
```

## 参考资源

- [AirSim官方文档](https://microsoft.github.io/AirSim/)
- [AirSim API参考](https://microsoft.github.io/AirSim/api_docs/html/)
- [强化学习教程](https://spinningup.openai.com/)
- [项目GitHub仓库](https://github.com/WdBlink/pilot_rl_navigation)

## 贡献指南

欢迎提交问题报告和功能请求到项目的GitHub仓库。在提交代码之前，请确保:

1. 代码符合项目的编码规范
2. 添加了适当的测试
3. 更新了相关文档
4. 通过了所有现有测试

## 许可证

本项目采用MIT许可证，详见LICENSE文件。