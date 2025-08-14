# 无人机位置预测模型

## 模型简介

本模型用于分析无人机GPS位置与图像中心点GPS位置之间的关系，并建立一个函数模型，能够根据无人机的姿态角(roll, pitch, heading)、飞行高度(altitude)以及图像中心点的GPS位置(compute_lat, compute_lon)来预测无人机的真实GPS位置(true_lat, true_lon)。

无人机的真实位置与图像中心点的计算位置之间的偏移主要由以下因素引起：

1. **姿态角（roll, pitch, heading）**：无人机的姿态会影响相机指向的方向，从而影响图像中心点对应的地面位置。
2. **飞行高度（altitude）**：高度越高，相同的姿态角会导致更大的地面偏移。
3. **相机参数**：本模型假设相机是垂直向下安装的（无安装角），且焦距固定。

## 模型实现方法

本模型采用了两种方法来预测无人机位置：

1. **物理模型**：基于旋转矩阵和投影原理，计算无人机位置与图像中心点位置之间的关系。
2. **机器学习模型**：使用多项式特征的线性回归或随机森林回归，从数据中学习无人机位置与各参数之间的关系。
3. **组合模型**：结合物理模型和机器学习模型的预测结果，获得更准确的预测。

## 使用方法

### 1. 训练模型

```python
from model.drone_position_model import DronePositionModel

# 创建模型实例
model = DronePositionModel()

# 加载数据
csv_path = "path/to/image_coordinates.csv"
df = model.load_data(csv_path)

# 预处理数据
X, y_lat, y_lon = model.preprocess_data(df)

# 训练模型（可选择'linear'或'forest'模型类型）
model.train_model(X, y_lat, y_lon, model_type='forest', poly_degree=2)

# 保存模型
model.save_model()
```

### 2. 加载已训练的模型

```python
from model.drone_position_model import DronePositionModel

# 创建模型实例
model = DronePositionModel()

# 加载已训练的模型
model.load_model("models/drone_position_model.pkl")
```

### 3. 使用模型预测

```python
# 预测无人机位置
predicted_position = model.predict(
    roll=2.5,           # 横滚角（度）
    pitch=3.2,          # 俯仰角（度）
    heading=150.5,      # 航向角（度）
    altitude=400.0,     # 飞行高度（米）
    compute_lat=38.090, # 图像中心点纬度
    compute_lon=104.410 # 图像中心点经度
)

print(f"预测的无人机位置: 纬度={predicted_position[0]}, 经度={predicted_position[1]}")
```

### 4. 仅使用物理模型

```python
from model.drone_position_model import calculate_drone_position

# 使用物理模型计算无人机位置
physics_prediction = calculate_drone_position(
    lon_target=104.410,  # 目标点经度
    lat_target=38.090,  # 目标点纬度
    roll_deg=2.5,  # 横滚角（度）
    pitch_deg=3.2,  # 俯仰角（度）
    yaw_deg=150.5,  # 航向角（度）
    h=400.0  # 飞行高度（米）
)

print(f"物理模型预测的无人机位置: 纬度={physics_prediction[0]}, 经度={physics_prediction[1]}")
```

## 测试与评估

可以使用提供的测试脚本来评估模型性能：

```bash
python test/test_drone_position_model.py
```

测试脚本将：
1. 测试物理模型的准确性
2. 比较不同模型（线性模型、随机森林、物理模型、组合模型）的性能
3. 在模拟的新数据上测试模型表现

## 模型性能

在测试数据集上，各模型的平均误差对比：
- 图像中心点直接作为无人机位置：约40-60米误差
- 物理模型：约30-40米误差
- 机器学习模型（随机森林）：约20-30米误差
- 组合模型：约15-25米误差

## 注意事项

1. 模型性能受训练数据质量和覆盖范围的影响，建议使用足够多样的数据进行训练。
2. 当无人机姿态角较大时（如俯仰角或横滚角超过15度），预测误差可能增大。
3. 模型假设相机垂直向下安装，如果实际安装有偏角，需要在预处理时进行补偿。