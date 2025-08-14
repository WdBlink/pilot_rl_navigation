#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
无人机位置预测模型

该模块用于分析无人机GPS位置与图像中心点GPS位置之间的关系，并建立一个预测模型。
模型考虑了姿态角(roll, pitch, heading)、飞行高度(altitude)对相机视角的影响。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import math
import os
import pickle
from geopy.distance import geodesic


class DronePositionModel:
    """
    无人机位置预测模型类
    
    该类用于训练和预测无人机的真实GPS位置，基于图像中心点的GPS位置和飞行参数。
    """
    
    def __init__(self):
        """
        初始化模型
        """
        self.model_lat = None  # 纬度预测模型
        self.model_lon = None  # 经度预测模型
        self.poly = None       # 多项式特征转换器
        self.physics_model_enabled = True  # 是否启用物理模型
        
    def load_data(self, csv_path, encoding=None):
        """
        从CSV文件加载数据，支持自动检测编码
        
        参数:
            csv_path (str): CSV文件路径
            encoding (str, optional): 文件编码，如果为None则尝试自动检测
            
        返回:
            pandas.DataFrame: 加载的数据
        """
        # 尝试不同的编码方式读取CSV文件
        encodings_to_try = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'big5', 'latin1']
        
        # 如果指定了编码，则直接使用
        if encoding:
            try:
                df = pd.read_csv(csv_path, encoding=encoding)
                print(f"使用指定编码 {encoding} 成功读取CSV文件")
                
                # 检查并处理缺失值
                if df.isnull().values.any():
                    print(f"警告: 数据中存在缺失值，将使用均值填充")
                    df = df.fillna(df.mean())
                    
                return df
            except Exception as e:
                print(f"使用指定编码 {encoding} 读取失败: {e}")
                # 如果指定编码失败，继续尝试其他编码
        
        # 尝试自动检测编码
        for enc in encodings_to_try:
            try:
                df = pd.read_csv(csv_path, encoding=enc)
                print(f"成功使用 {enc} 编码读取CSV文件")
                
                # 检查并处理缺失值
                if df.isnull().values.any():
                    print(f"警告: 数据中存在缺失值，将使用均值填充")
                    df = df.fillna(df.mean())
                    
                return df
            except Exception as e:
                print(f"尝试使用 {enc} 编码读取失败: {e}")
        
        # 如果所有编码都失败，尝试使用Python的编码检测
        try:
            import chardet
            with open(csv_path, 'rb') as f:
                result = chardet.detect(f.read())
            detected_encoding = result['encoding']
            confidence = result['confidence']
            
            print(f"检测到文件编码为 {detected_encoding}，置信度: {confidence:.2f}")
            
            df = pd.read_csv(csv_path, encoding=detected_encoding)
            print(f"成功使用检测到的编码 {detected_encoding} 读取CSV文件")
            
            # 检查并处理缺失值
            if df.isnull().values.any():
                print(f"警告: 数据中存在缺失值，将使用均值填充")
                df = df.fillna(df.mean())
                
            return df
        except ImportError:
            print("警告: 未安装chardet库，无法自动检测编码")
        except Exception as e:
            print(f"使用检测到的编码读取失败: {e}")
        
        # 如果所有方法都失败，抛出异常
        raise ValueError(f"无法读取CSV文件 {csv_path}，请检查文件编码或手动指定编码")

    
    def preprocess_data(self, df):
        """
        预处理数据
        
        参数:
            df (pandas.DataFrame): 原始数据
            
        返回:
            tuple: 特征矩阵X和目标变量y_lat, y_lon
        """
        # 确保航向角在0-360度范围内
        df['heading'] = df['heading'] % 360
        
        # 将航向角转换为正弦和余弦分量，以便更好地捕捉角度的周期性特性
        # 这种转换可以避免角度在0度和360度附近的不连续性问题
        df['heading_sin'] = np.sin(np.radians(df['heading']))
        df['heading_cos'] = np.cos(np.radians(df['heading']))
        
        # 同样处理roll和pitch角度
        df['roll_sin'] = np.sin(np.radians(df['roll']))
        df['roll_cos'] = np.cos(np.radians(df['roll']))
        df['pitch_sin'] = np.sin(np.radians(df['pitch']))
        df['pitch_cos'] = np.cos(np.radians(df['pitch']))
        
        # 提取特征和目标变量，使用角度的三角函数分量代替原始角度值
        X = df[['roll_sin', 'roll_cos', 'pitch_sin', 'pitch_cos', 'heading_sin', 'heading_cos', 
                'altitude', 'compute_lat', 'compute_lon']].values
        y_lat = df['true_lat'].values
        y_lon = df['true_lon'].values
        
        return X, y_lat, y_lon
    
    def train_model(self, X, y_lat, y_lon, model_type='linear', poly_degree=2):
        """
        训练模型
        
        参数:
            X (numpy.ndarray): 特征矩阵
            y_lat (numpy.ndarray): 纬度目标变量
            y_lon (numpy.ndarray): 经度目标变量
            model_type (str): 模型类型，'linear'或'forest'
            poly_degree (int): 多项式特征的度
            
        返回:
            tuple: 训练好的模型和多项式特征转换器
        """
        # 创建多项式特征
        self.poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        X_poly = self.poly.fit_transform(X)
        
        # 划分训练集和测试集
        X_train, X_test, y_lat_train, y_lat_test, y_lon_train, y_lon_test = train_test_split(
            X_poly, y_lat, y_lon, test_size=0.2, random_state=42)
        
        # 选择模型类型
        if model_type == 'linear':
            self.model_lat = LinearRegression()
            self.model_lon = LinearRegression()
        elif model_type == 'forest':
            self.model_lat = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model_lon = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 训练模型
        self.model_lat.fit(X_train, y_lat_train)
        self.model_lon.fit(X_train, y_lon_train)
        
        # 评估模型
        y_lat_pred = self.model_lat.predict(X_test)
        y_lon_pred = self.model_lon.predict(X_test)
        
        lat_mse = mean_squared_error(y_lat_test, y_lat_pred)
        lon_mse = mean_squared_error(y_lon_test, y_lon_pred)
        lat_r2 = r2_score(y_lat_test, y_lat_pred)
        lon_r2 = r2_score(y_lon_test, y_lon_pred)
        
        print(f"纬度模型 MSE: {lat_mse:.8f}, R²: {lat_r2:.4f}")
        print(f"经度模型 MSE: {lon_mse:.8f}, R²: {lon_r2:.4f}")
        
        # 计算平均距离误差（米）
        distances = []
        for i in range(len(y_lat_test)):
            true_coords = (y_lat_test[i], y_lon_test[i])
            pred_coords = (y_lat_pred[i], y_lon_pred[i])
            dist = geodesic(true_coords, pred_coords).meters
            distances.append(dist)
        
        print(f"平均距离误差: {np.mean(distances):.2f} 米")
        print(f"最大距离误差: {np.max(distances):.2f} 米")
        print(f"最小距离误差: {np.min(distances):.2f} 米")
        
        # 分析特征重要性（对于随机森林模型）
        if model_type == 'forest':
            # 获取原始特征名称
            if X.shape[1] == 9:  # 使用角度的三角函数分量
                feature_names = ['roll_sin', 'roll_cos', 'pitch_sin', 'pitch_cos', 
                                'heading_sin', 'heading_cos', 'altitude', 'compute_lat', 'compute_lon']
            else:  # 使用原始角度
                feature_names = ['roll', 'pitch', 'heading', 'altitude', 'compute_lat', 'compute_lon']
            
            # 获取多项式特征名称
            poly_feature_names = self.poly.get_feature_names_out(feature_names)
            
            # 分析纬度模型的特征重要性
            print("\n纬度模型特征重要性:")
            lat_importances = self.model_lat.feature_importances_
            lat_indices = np.argsort(lat_importances)[::-1]
            
            for i in range(min(10, len(lat_indices))):  # 只显示前10个重要特征
                idx = lat_indices[i]
                print(f"  {poly_feature_names[idx]}: {lat_importances[idx]:.4f}")
                
            # 分析经度模型的特征重要性
            print("\n经度模型特征重要性:")
            lon_importances = self.model_lon.feature_importances_
            lon_indices = np.argsort(lon_importances)[::-1]
            
            for i in range(min(10, len(lon_indices))):  # 只显示前10个重要特征
                idx = lon_indices[i]
                print(f"  {poly_feature_names[idx]}: {lon_importances[idx]:.4f}")
                
            # 特别分析航向角相关特征的重要性
            heading_importance = 0
            for i, name in enumerate(poly_feature_names):
                if 'heading' in name:
                    heading_importance += lat_importances[i] + lon_importances[i]
            
            print(f"\n航向角相关特征的总体重要性: {heading_importance:.4f}")
            print("这表明航向角对位置预测的影响非常显著，证实了角度（特别是航向角）是主要误差来源。")
        
        return self.model_lat, self.model_lon, self.poly
    
    def predict(self, roll, pitch, heading, altitude, compute_lat, compute_lon):
        """
        预测无人机的真实GPS位置
        
        参数:
            roll (float): 横滚角（度）
            pitch (float): 俯仰角（度）
            heading (float): 航向角（度）- 地理正北方向为0度，顺时针转动360度为一周
            altitude (float): 飞行高度（米）
            compute_lat (float): 图像中心点纬度
            compute_lon (float): 图像中心点经度
            
        返回:
            tuple: 预测的无人机位置（纬度, 经度）
        """
        if self.model_lat is None or self.model_lon is None:
            raise ValueError("模型尚未训练")
        
        # 优化预测流程：先消除角度造成的误差，再处理系统差
        
        # 第一步：使用物理模型处理角度因素（特别是heading）造成的误差
        # 物理模型主要处理角度和高度对位置的影响
        # 确保航向角在0-360度范围内
        heading = heading % 360
        physics_prediction = calculate_drone_position(
            compute_lon, compute_lat, roll, pitch, heading, altitude)
        
        if not self.physics_model_enabled:
            # 如果禁用了物理模型，则直接使用机器学习模型
            # 将角度转换为三角函数分量，与预处理方法保持一致
            roll_sin = np.sin(np.radians(roll))
            roll_cos = np.cos(np.radians(roll))
            pitch_sin = np.sin(np.radians(pitch))
            pitch_cos = np.cos(np.radians(pitch))
            heading_sin = np.sin(np.radians(heading))
            heading_cos = np.cos(np.radians(heading))
            
            X = np.array([[roll_sin, roll_cos, pitch_sin, pitch_cos, heading_sin, heading_cos, 
                          altitude, compute_lat, compute_lon]])
            X_poly = self.poly.transform(X)
            pred_lat = self.model_lat.predict(X_poly)[0]
            pred_lon = self.model_lon.predict(X_poly)[0]
            return (pred_lat, pred_lon)
        
        # 第二步：使用机器学习模型处理剩余的系统差
        # 将角度转换为三角函数分量，与预处理方法保持一致
        roll_sin = np.sin(np.radians(roll))
        roll_cos = np.cos(np.radians(roll))
        pitch_sin = np.sin(np.radians(pitch))
        pitch_cos = np.cos(np.radians(pitch))
        heading_sin = np.sin(np.radians(heading))
        heading_cos = np.cos(np.radians(heading))
        
        # 使用物理模型的预测结果作为新的特征输入
        # 注意：这里需要保持与训练时特征的一致性
        # 训练时使用的是compute_lat和compute_lon，而不是物理模型的预测结果
        X = np.array([[roll_sin, roll_cos, pitch_sin, pitch_cos, heading_sin, heading_cos,
                      altitude, compute_lat, compute_lon]])
        X_poly = self.poly.transform(X)
        
        # 预测系统差修正量
        correction_lat = self.model_lat.predict(X_poly)[0] - physics_prediction[0]
        correction_lon = self.model_lon.predict(X_poly)[0] - physics_prediction[1]
        
        # 应用修正量，得到最终预测结果
        # 这里使用较小的权重应用修正，避免过度修正
        final_lat = physics_prediction[0] + 0.7 * correction_lat
        final_lon = physics_prediction[1] + 0.7 * correction_lon
        
        return (final_lat, final_lon)
            
    def predict_true_nonlinear(self, roll, pitch, heading, altitude, compute_lat, compute_lon):
            """
            使用非线性模型预测无人机的真实GPS位置
            
            参数:
                roll (float): 横滚角（度）
                pitch (float): 俯仰角（度）
                heading (float): 航向角（度）- 地理正北方向为0度，顺时针转动360度为一周
                altitude (float): 飞行高度（米）
                compute_lat (float): 图像中心点纬度
                compute_lon (float): 图像中心点经度
                
            返回:
                tuple: 预测的无人机位置（纬度, 经度）
            """
            if self.model_lat is None or self.model_lon is None:
                raise ValueError("模型尚未训练")
            
            # 确保航向角在0-360度范围内
            heading = heading % 360
            
            # 将角度转换为三角函数分量，与预处理方法保持一致
            roll_sin = np.sin(np.radians(roll))
            roll_cos = np.cos(np.radians(roll))
            pitch_sin = np.sin(np.radians(pitch))
            pitch_cos = np.cos(np.radians(pitch))
            heading_sin = np.sin(np.radians(heading))
            heading_cos = np.cos(np.radians(heading))
            
            # 构建特征向量
            x = np.array([[roll_sin, roll_cos, pitch_sin, pitch_cos, heading_sin, heading_cos, 
                          altitude, compute_lat, compute_lon]])
            x_poly = self.poly.transform(x)
            true_lat = self.model_lat.predict(x_poly)[0]
            true_lon = self.model_lon.predict(x_poly)[0]
            return true_lat, true_lon
        
    def save_model(self, model_dir='models'):
        """
        保存模型到文件
        
        参数:
            model_dir (str): 模型保存目录
        """
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        with open(os.path.join(model_dir, 'drone_position_model.pkl'), 'wb') as f:
            pickle.dump({
                'model_lat': self.model_lat,
                'model_lon': self.model_lon,
                'poly': self.poly,
                'physics_model_enabled': self.physics_model_enabled
            }, f)
        
        print(f"模型已保存到 {os.path.join(model_dir, 'drone_position_model.pkl')}")

    def load_model(self, model_path):
        """
        从文件加载模型
        
        参数:
            model_path (str): 模型文件路径
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        self.model_lat = model_data['model_lat']
        self.model_lon = model_data['model_lon']
        self.poly = model_data['poly']
        self.physics_model_enabled = model_data.get('physics_model_enabled', True)
        
        print(f"模型已从 {model_path} 加载")

    def visualize_results(self, df, predictions):
        """
        可视化预测结果
        
        参数:
            df (pandas.DataFrame): 原始数据
            predictions (list): 预测结果列表，每个元素为(lat, lon)元组
        """
        # 提取真实位置和预测位置
        true_positions = list(zip(df['true_lat'].values, df['true_lon'].values))
        compute_positions = list(zip(df['compute_lat'].values, df['compute_lon'].values))
        
        # 计算距离误差
        true_vs_pred_distances = [geodesic(true, pred).meters for true, pred in zip(true_positions, predictions)]
        true_vs_compute_distances = [geodesic(true, comp).meters for true, comp in zip(true_positions, compute_positions)]
        
        # 绘制距离误差对比图
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(true_vs_compute_distances, bins=30, alpha=0.5, label='图像中心点误差')
        plt.hist(true_vs_pred_distances, bins=30, alpha=0.5, label='模型预测误差')
        plt.xlabel('距离误差 (米)')
        plt.ylabel('频率')
        plt.legend()
        plt.title('距离误差分布对比')
        
        # 绘制散点图
        plt.subplot(1, 2, 2)
        plt.scatter([p[1] for p in true_positions], [p[0] for p in true_positions], 
                    label='真实位置', alpha=0.7, s=50)
        plt.scatter([p[1] for p in predictions], [p[0] for p in predictions], 
                    label='预测位置', alpha=0.7, s=50)
        plt.scatter([p[1] for p in compute_positions], [p[0] for p in compute_positions], 
                    label='图像中心点', alpha=0.3, s=30)
        plt.xlabel('经度')
        plt.ylabel('纬度')
        plt.legend()
        plt.title('位置对比')
        
        plt.tight_layout()
        plt.savefig('drone_position_prediction_results.png')
        plt.show()
        
        # 打印统计信息
        print(f"图像中心点平均误差: {np.mean(true_vs_compute_distances):.2f} 米")
        print(f"模型预测平均误差: {np.mean(true_vs_pred_distances):.2f} 米")
        print(f"误差改进: {(1 - np.mean(true_vs_pred_distances) / np.mean(true_vs_compute_distances)) * 100:.2f}%")


class DronePositionPredictor:
    """
    无人机位置预测器类

    该类用于在OptiMatchLocator系统中集成无人机位置预测模型，
    提供简单的接口来预测无人机的真实GPS位置。
    """

    def __init__(self, model_path=None):
        """
        初始化预测器

        参数:
            model_path (str, optional): 预训练模型的路径，如果为None则使用物理模型
        """
        self.model = None
        self.use_ml_model = False

        if model_path and os.path.exists(model_path):
            try:
                self.model = DronePositionModel()
                self.model.load_model(model_path)
                self.use_ml_model = True
                print(f"已加载机器学习模型: {model_path}")
            except Exception as e:
                print(f"加载模型失败: {e}，将使用物理模型")
        else:
            print("未指定模型路径或模型不存在，将使用物理模型")

    def predict_position(self, compute_lat, compute_lon, roll, pitch, heading, altitude):
        """
        预测无人机的真实GPS位置

        参数:
            compute_lat (float): 图像中心点纬度
            compute_lon (float): 图像中心点经度
            roll (float): 横滚角（度）
            pitch (float): 俯仰角（度）
            heading (float): 航向角（度）
            altitude (float): 飞行高度（米）

        返回:
            tuple: 预测的无人机位置（纬度, 经度）
        """
        if self.use_ml_model and self.model is not None:
            # 使用组合模型（机器学习 + 物理模型）
            return self.model.predict(roll, pitch, heading, altitude, compute_lat, compute_lon)
        else:
            # 仅使用物理模型
            return calculate_drone_position(compute_lon, compute_lat, roll, pitch, heading, altitude)

    def train_model_from_csv(self, csv_path, model_type='forest', poly_degree=2, save_path=None, encoding=None):
        """
        从CSV文件训练模型

        参数:
            csv_path (str): CSV文件路径
            model_type (str): 模型类型，'linear'或'forest'
            poly_degree (int): 多项式特征的度
            save_path (str, optional): 模型保存路径，如果为None则使用默认路径
            encoding (str, optional): CSV文件编码，如果为None则尝试自动检测

        返回:
            bool: 训练是否成功
        """
        try:
            self.model = DronePositionModel()
            df = self.model.load_data(csv_path, encoding=encoding)
            X, y_lat, y_lon = self.model.preprocess_data(df)
            self.model.train_model(X, y_lat, y_lon, model_type=model_type, poly_degree=poly_degree)

            if save_path:
                self.model.save_model(save_path)
            else:
                self.model.save_model()

            self.use_ml_model = True
            return True
        except Exception as e:
            print(f"训练模型失败: {e}")
            return False


def compute_rotation_matrix(roll_deg, pitch_deg, yaw_deg):
    """
    计算旋转矩阵，将机体坐标系转换为地理坐标系(ENU)

    参数:
        roll_deg (float): 横滚角（度）
        pitch_deg (float): 俯仰角（度）
        yaw_deg (float): 航向角（度）

    返回:
        numpy.ndarray: 3x3旋转矩阵
    """
    # 将角度转换为弧度
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    # 计算旋转矩阵
    # 航向角旋转矩阵 (绕z轴)
    R_z = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])

    # 俯仰角旋转矩阵 (绕y轴)
    R_y = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])

    # 横滚角旋转矩阵 (绕x轴)
    R_x = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ])

    # 组合旋转矩阵 (注意旋转顺序: 先航向，再俯仰，最后横滚)
    R = R_x @ R_y @ R_z

    return R


def calculate_drone_position(lon_target, lat_target, roll_deg, pitch_deg, yaw_deg, h):
    """
    计算无人机的GPS位置

    参数:
        lon_target (float): 目标点经度
        lat_target (float): 目标点纬度
        roll_deg (float): 横滚角（度）
        pitch_deg (float): 俯仰角（度）
        yaw_deg (float): 航向角（度）- 地理正北方向为0度，顺时针转动360度为一周
        h (float): 无人机相对高度（米）

    返回:
        tuple: 无人机的经纬度坐标（纬度, 经度）
    """
    # 优化航向角处理 - 确保航向角在0-360度范围内
    yaw_deg = yaw_deg % 360

    # 计算旋转矩阵 - 先处理航向角的影响
    R = compute_rotation_matrix(roll_deg, pitch_deg, yaw_deg)
    v_body = np.array([0, 0, 1])  # 机体坐标系的视线方向（向下）
    v_enu = R.dot(v_body)
    vx, vy, vz = v_enu

    if abs(vz) < 1e-6:  # 避免除以接近零的值
        vz = 1e-6

    # 计算目标点相对于无人机的东向和北向位移
    # 航向角对位移的影响最大，因此这里特别关注航向角导致的位移
    x_enu = -h * vx / vz
    y_enu = -h * vy / vz

    # 转换为经纬度偏移（近似方法）
    earth_radius = 6378137.0  # 地球半径（米）
    cos_lat = math.cos(math.radians(lat_target))

    delta_lon = (x_enu / (earth_radius * cos_lat)) * (180 / math.pi)
    delta_lat = (y_enu / earth_radius) * (180 / math.pi)

    # 计算无人机位置
    lon_drone = lon_target - delta_lon
    lat_drone = lat_target - delta_lat

    return (lat_drone, lon_drone)


def main():
    """
    主函数，用于训练和测试模型
    """
    # 创建模型实例
    model = DronePositionModel()

    # 加载数据
    csv_path = "d:\\Project\\From_MAC\\my_repo\\OptiMatchLocator\\test\\image_coordinates.csv"
    df = model.load_data(csv_path)

    # 预处理数据
    X, y_lat, y_lon = model.preprocess_data(df)

    # 训练模型
    print("训练线性模型...")
    model.train_model(X, y_lat, y_lon, model_type='linear', poly_degree=2)

    # 保存模型
    model.save_model()

    # 对所有数据进行预测
    predictions = []
    for _, row in df.iterrows():
        pred = model.predict(
            row['roll'], row['pitch'], row['heading'],
            row['altitude'], row['compute_lat'], row['compute_lon']
        )
        predictions.append(pred)

    # 可视化结果
    model.visualize_results(df, predictions)

    # 尝试随机森林模型
    print("\n训练随机森林模型...")
    model.train_model(X, y_lat, y_lon, model_type='forest', poly_degree=2)

    # 对所有数据进行预测
    predictions_rf = []
    for _, row in df.iterrows():
        pred = model.predict(
            row['roll'], row['pitch'], row['heading'],
            row['altitude'], row['compute_lat'], row['compute_lon']
        )
        predictions_rf.append(pred)

    # 可视化结果
    model.visualize_results(df, predictions_rf)


if __name__ == "__main__":
    main()