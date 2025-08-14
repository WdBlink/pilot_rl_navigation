#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强化学习无人机定位导航系统 - 光学定位模块

本模块实现基于计算机视觉的无人机位置估计功能，包括：
1. 特征提取和匹配
2. 位姿估计算法
3. 参考地图管理
4. 匹配质量评估
5. 实时位置计算

Author: wdblink
Date: 2024
"""

import numpy as np
import cv2
import time
import os
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
import torch
from PIL import Image
from geopy.distance import geodesic

from ..utils.data_types import Position3D, OpticalMatchResult
from ..utils.logger import get_logger, performance_monitor, log_function_call
from ..environment.model.superpoint import SuperPoint
from ..environment.model.lightglue import LightGlue


@dataclass
class OpticalConfig:
    """光学定位配置类
    
    包含光学定位系统的所有配置参数。
    
    Attributes:
        device: 计算设备 (cuda/cpu)
        superpoint_weights: SuperPoint模型权重路径
        lightglue_weights: LightGlue模型权重路径
        max_keypoints: 最大关键点数量
        keypoint_threshold: 关键点检测阈值
        match_threshold: 匹配阈值
        min_matches: 最小匹配点数量
        reference_map_path: 参考地图路径
        output_path: 输出路径
    """
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    superpoint_weights: str = "weights/superpoint_v1.pth"
    lightglue_weights: str = "weights/lightglue_outdoor.pth"
    max_keypoints: int = 1024
    keypoint_threshold: float = 0.005
    match_threshold: float = 0.2
    min_matches: int = 8
    reference_map_path: str = "data/reference_map.tif"
    output_path: str = "output/optical_positioning"


class FeatureExtractor:
    """特征提取器类
    
    使用SuperPoint网络提取图像特征点。
    """
    
    def __init__(self, config: OpticalConfig):
        """初始化特征提取器
        
        Args:
            config: 光学定位配置
        """
        self.config = config
        self.device = config.device
        
        # 初始化SuperPoint模型
        self.superpoint = SuperPoint({
            'max_keypoints': config.max_keypoints,
            'keypoint_threshold': config.keypoint_threshold,
            'nms_radius': 4,
            'remove_borders': 4
        }).to(self.device)
        
        # 加载预训练权重
        if os.path.exists(config.superpoint_weights):
            self.superpoint.load_state_dict(torch.load(config.superpoint_weights, map_location=self.device))
        
        self.superpoint.eval()
        self.logger = get_logger("feature_extractor")
    
    @performance_monitor
    def extract_features(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """提取图像特征
        
        Args:
            image: 输入图像张量 [1, 1, H, W]
            
        Returns:
            特征字典，包含关键点、描述符和分数
        """
        with torch.no_grad():
            features = self.superpoint({'image': image})
        
        return {
            'keypoints': features['keypoints'],
            'descriptors': features['descriptors'],
            'scores': features['keypoint_scores']
        }


class FeatureMatcher:
    """特征匹配器类
    
    使用LightGlue网络进行特征匹配。
    """
    
    def __init__(self, config: OpticalConfig):
        """初始化特征匹配器
        
        Args:
            config: 光学定位配置
        """
        self.config = config
        self.device = config.device
        
        # 初始化LightGlue模型
        self.lightglue = LightGlue({
            'width_confidence': 0.99,
            'depth_confidence': 0.95,
            'features': 'superpoint'
        }).to(self.device)
        
        # 加载预训练权重
        if os.path.exists(config.lightglue_weights):
            self.lightglue.load_state_dict(torch.load(config.lightglue_weights, map_location=self.device))
        
        self.lightglue.eval()
        self.logger = get_logger("feature_matcher")
    
    @performance_monitor
    def match_features(self, features0: Dict[str, torch.Tensor], 
                      features1: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """匹配两组特征
        
        Args:
            features0: 第一组特征
            features1: 第二组特征
            
        Returns:
            匹配结果字典
        """
        with torch.no_grad():
            matches = self.lightglue({
                'image0': features0,
                'image1': features1
            })
        
        return matches


class PoseEstimator:
    """位姿估计器类
    
    基于特征匹配结果估计无人机位姿。
    """
    
    def __init__(self, config: OpticalConfig):
        """初始化位姿估计器
        
        Args:
            config: 光学定位配置
        """
        self.config = config
        self.logger = get_logger("pose_estimator")
    
    def estimate_pose_pnp(self, 
                         matched_points_2d: np.ndarray,
                         matched_points_3d: np.ndarray,
                         camera_matrix: np.ndarray,
                         dist_coeffs: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
        """使用PnP算法估计位姿
        
        Args:
            matched_points_2d: 2D匹配点
            matched_points_3d: 3D匹配点
            camera_matrix: 相机内参矩阵
            dist_coeffs: 畸变系数
            
        Returns:
            旋转向量、平移向量和成功标志
        """
        if len(matched_points_2d) < self.config.min_matches:
            return None, None, False
        
        try:
            # 使用RANSAC PnP求解位姿
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                matched_points_3d.astype(np.float32),
                matched_points_2d.astype(np.float32),
                camera_matrix,
                dist_coeffs,
                reprojectionError=5.0,
                confidence=0.99
            )
            
            if success and inliers is not None and len(inliers) >= self.config.min_matches:
                return rvec, tvec, True
            else:
                return None, None, False
                
        except Exception as e:
            self.logger.log_event("error", "pnp_failed", {"error": str(e)})
            return None, None, False
    
    def estimate_pose_homography(self, 
                               matched_points_ref: np.ndarray,
                               matched_points_query: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """使用单应性矩阵估计位姿
        
        Args:
            matched_points_ref: 参考图像匹配点
            matched_points_query: 查询图像匹配点
            
        Returns:
            单应性矩阵和匹配质量分数
        """
        if len(matched_points_ref) < self.config.min_matches:
            return None, 0.0
        
        try:
            # 计算单应性矩阵
            H, mask = cv2.findHomography(
                matched_points_query.astype(np.float32),
                matched_points_ref.astype(np.float32),
                cv2.RANSAC,
                ransacReprojThreshold=5.0,
                confidence=0.99
            )
            
            if H is not None and mask is not None:
                # 计算内点比例作为质量分数
                inlier_ratio = np.sum(mask) / len(mask)
                return H, float(inlier_ratio)
            else:
                return None, 0.0
                
        except Exception as e:
            self.logger.log_event("error", "homography_failed", {"error": str(e)})
            return None, 0.0


class ReferenceMapManager:
    """参考地图管理器类
    
    管理参考地图数据和地理坐标转换。
    """
    
    def __init__(self, config: OpticalConfig):
        """初始化参考地图管理器
        
        Args:
            config: 光学定位配置
        """
        self.config = config
        self.reference_map = None
        self.geo_transform = None
        self.logger = get_logger("reference_map_manager")
        
        # 加载参考地图
        self.load_reference_map()
    
    def load_reference_map(self) -> bool:
        """加载参考地图
        
        Returns:
            加载成功标志
        """
        try:
            from osgeo import gdal
            
            # 打开GeoTIFF文件
            dataset = gdal.Open(self.config.reference_map_path)
            if dataset is None:
                self.logger.log_event("error", "map_load_failed", 
                                     {"path": self.config.reference_map_path})
                return False
            
            # 读取图像数据
            self.reference_map = dataset.ReadAsArray()
            if len(self.reference_map.shape) == 3:
                self.reference_map = np.transpose(self.reference_map, (1, 2, 0))
            
            # 获取地理变换参数
            self.geo_transform = dataset.GetGeoTransform()
            
            self.logger.log_event("info", "map_loaded", {
                "shape": self.reference_map.shape,
                "geo_transform": self.geo_transform
            })
            
            return True
            
        except Exception as e:
            self.logger.log_event("error", "map_load_exception", {"error": str(e)})
            return False
    
    def pixel_to_geo(self, x: float, y: float) -> Tuple[float, float]:
        """像素坐标转地理坐标
        
        Args:
            x: 像素X坐标
            y: 像素Y坐标
            
        Returns:
            经度和纬度
        """
        if self.geo_transform is None:
            return 0.0, 0.0
        
        lon = self.geo_transform[0] + x * self.geo_transform[1] + y * self.geo_transform[2]
        lat = self.geo_transform[3] + x * self.geo_transform[4] + y * self.geo_transform[5]
        
        return lon, lat
    
    def geo_to_pixel(self, lon: float, lat: float) -> Tuple[float, float]:
        """地理坐标转像素坐标
        
        Args:
            lon: 经度
            lat: 纬度
            
        Returns:
            像素X和Y坐标
        """
        if self.geo_transform is None:
            return 0.0, 0.0
        
        # 求解线性方程组
        det = self.geo_transform[1] * self.geo_transform[5] - self.geo_transform[2] * self.geo_transform[4]
        if abs(det) < 1e-10:
            return 0.0, 0.0
        
        x = ((lon - self.geo_transform[0]) * self.geo_transform[5] - 
             (lat - self.geo_transform[3]) * self.geo_transform[2]) / det
        y = ((lat - self.geo_transform[3]) * self.geo_transform[1] - 
             (lon - self.geo_transform[0]) * self.geo_transform[4]) / det
        
        return x, y
    
    def crop_map_region(self, center_lon: float, center_lat: float, 
                       crop_size: Tuple[int, int]) -> Tuple[Optional[np.ndarray], Optional[Tuple]]:
        """裁剪地图区域
        
        Args:
            center_lon: 中心经度
            center_lat: 中心纬度
            crop_size: 裁剪尺寸 (width, height)
            
        Returns:
            裁剪的图像和地理变换参数
        """
        if self.reference_map is None or self.geo_transform is None:
            return None, None
        
        # 转换为像素坐标
        center_x, center_y = self.geo_to_pixel(center_lon, center_lat)
        
        # 计算裁剪区域
        half_width, half_height = crop_size[0] // 2, crop_size[1] // 2
        x_start = max(0, int(center_x - half_width))
        y_start = max(0, int(center_y - half_height))
        x_end = min(self.reference_map.shape[1], x_start + crop_size[0])
        y_end = min(self.reference_map.shape[0], y_start + crop_size[1])
        
        # 裁剪图像
        cropped_map = self.reference_map[y_start:y_end, x_start:x_end]
        
        # 更新地理变换参数
        new_geo_transform = (
            self.geo_transform[0] + x_start * self.geo_transform[1] + y_start * self.geo_transform[2],
            self.geo_transform[1],
            self.geo_transform[2],
            self.geo_transform[3] + x_start * self.geo_transform[4] + y_start * self.geo_transform[5],
            self.geo_transform[4],
            self.geo_transform[5]
        )
        
        return cropped_map, new_geo_transform


class OpticalPositioning:
    """光学定位主类
    
    整合特征提取、匹配和位姿估计功能，实现完整的光学定位系统。
    """
    
    def __init__(self, config: OpticalConfig):
        """初始化光学定位系统
        
        Args:
            config: 光学定位配置
        """
        self.config = config
        self.logger = get_logger("optical_positioning")
        
        # 初始化各个组件
        self.feature_extractor = FeatureExtractor(config)
        self.feature_matcher = FeatureMatcher(config)
        self.pose_estimator = PoseEstimator(config)
        self.map_manager = ReferenceMapManager(config)
        
        # 创建输出目录
        os.makedirs(config.output_path, exist_ok=True)
        
        self.logger.log_event("info", "optical_positioning_initialized")
    
    @log_function_call
    def process_image(self, 
                     query_image: np.ndarray,
                     approximate_position: Optional[Position3D] = None,
                     camera_params: Optional[Dict[str, np.ndarray]] = None) -> OpticalMatchResult:
        """处理查询图像并估计位置
        
        Args:
            query_image: 查询图像
            approximate_position: 近似位置（用于裁剪参考地图）
            camera_params: 相机参数字典
            
        Returns:
            光学匹配结果
        """
        start_time = time.time()
        
        try:
            # 预处理查询图像
            query_tensor = self._preprocess_image(query_image)
            
            # 获取参考地图区域
            if approximate_position is not None:
                ref_image, ref_geo_transform = self.map_manager.crop_map_region(
                    approximate_position.x, approximate_position.y, (2000, 2000)
                )
            else:
                ref_image = self.map_manager.reference_map
                ref_geo_transform = self.map_manager.geo_transform
            
            if ref_image is None:
                return self._create_failed_result(start_time, "参考地图不可用")
            
            # 预处理参考图像
            ref_tensor = self._preprocess_image(ref_image)
            
            # 提取特征
            query_features = self.feature_extractor.extract_features(query_tensor)
            ref_features = self.feature_extractor.extract_features(ref_tensor)
            
            # 特征匹配
            matches = self.feature_matcher.match_features(ref_features, query_features)
            
            # 解析匹配结果
            match_indices = matches['matches0'].cpu().numpy()
            valid_matches = match_indices > -1
            
            if np.sum(valid_matches) < self.config.min_matches:
                return self._create_failed_result(start_time, "匹配点数量不足")
            
            # 获取匹配点坐标
            ref_keypoints = ref_features['keypoints'][0].cpu().numpy()
            query_keypoints = query_features['keypoints'][0].cpu().numpy()
            
            matched_ref_points = ref_keypoints[valid_matches]
            matched_query_points = query_keypoints[match_indices[valid_matches]]
            
            # 位姿估计
            if camera_params is not None:
                # 使用PnP算法
                position = self._estimate_position_pnp(
                    matched_ref_points, matched_query_points, 
                    camera_params, ref_geo_transform
                )
            else:
                # 使用单应性矩阵
                position = self._estimate_position_homography(
                    matched_ref_points, matched_query_points, ref_geo_transform
                )
            
            # 计算匹配质量
            match_quality = self._calculate_match_quality(
                matched_ref_points, matched_query_points, matches
            )
            
            processing_time = time.time() - start_time
            
            result = OpticalMatchResult(
                position=position,
                confidence=min(1.0, match_quality),
                num_matches=int(np.sum(valid_matches)),
                match_quality=match_quality,
                processing_time=processing_time,
                timestamp=time.time()
            )
            
            self.logger.log_event("info", "image_processed", {
                "num_matches": result.num_matches,
                "match_quality": result.match_quality,
                "processing_time": result.processing_time
            })
            
            return result
            
        except Exception as e:
            self.logger.log_event("error", "image_processing_failed", {"error": str(e)})
            return self._create_failed_result(start_time, str(e))
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """预处理图像
        
        Args:
            image: 输入图像
            
        Returns:
            预处理后的张量
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 归一化到[0, 1]
        gray = gray.astype(np.float32) / 255.0
        
        # 转换为张量
        tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(self.config.device)
        
        return tensor
    
    def _estimate_position_pnp(self, 
                              matched_ref_points: np.ndarray,
                              matched_query_points: np.ndarray,
                              camera_params: Dict[str, np.ndarray],
                              ref_geo_transform: Tuple) -> Position3D:
        """使用PnP算法估计位置"""
        # 将参考点转换为3D坐标（假设地面高度为0）
        matched_3d_points = np.zeros((len(matched_ref_points), 3))
        for i, (x, y) in enumerate(matched_ref_points):
            lon, lat = self.map_manager.pixel_to_geo(x, y)
            matched_3d_points[i] = [lon, lat, 0.0]  # 简化处理，实际应用中需要高程数据
        
        # PnP求解
        rvec, tvec, success = self.pose_estimator.estimate_pose_pnp(
            matched_query_points, matched_3d_points,
            camera_params['camera_matrix'], camera_params['dist_coeffs']
        )
        
        if success and tvec is not None:
            return Position3D(
                x=float(tvec[0, 0]),
                y=float(tvec[1, 0]),
                z=float(tvec[2, 0]),
                timestamp=time.time(),
                confidence=0.8
            )
        else:
            return Position3D(0.0, 0.0, 0.0, time.time(), 0.0)
    
    def _estimate_position_homography(self, 
                                    matched_ref_points: np.ndarray,
                                    matched_query_points: np.ndarray,
                                    ref_geo_transform: Tuple) -> Position3D:
        """使用单应性矩阵估计位置"""
        H, quality = self.pose_estimator.estimate_pose_homography(
            matched_ref_points, matched_query_points
        )
        
        if H is not None:
            # 计算查询图像中心点在参考图像中的对应位置
            query_center = np.array([[matched_query_points.shape[1] / 2, 
                                    matched_query_points.shape[0] / 2]], dtype=np.float32)
            ref_center = cv2.perspectiveTransform(query_center.reshape(1, -1, 2), H)
            
            # 转换为地理坐标
            ref_x, ref_y = ref_center[0, 0]
            lon, lat = self.map_manager.pixel_to_geo(ref_x, ref_y)
            
            return Position3D(
                x=float(lon),
                y=float(lat),
                z=0.0,
                timestamp=time.time(),
                confidence=quality
            )
        else:
            return Position3D(0.0, 0.0, 0.0, time.time(), 0.0)
    
    def _calculate_match_quality(self, 
                               matched_ref_points: np.ndarray,
                               matched_query_points: np.ndarray,
                               matches: Dict[str, torch.Tensor]) -> float:
        """计算匹配质量分数"""
        # 基于匹配点数量的质量
        num_matches = len(matched_ref_points)
        quantity_score = min(1.0, num_matches / 100.0)
        
        # 基于匹配分布的质量
        if num_matches > 4:
            ref_std = np.std(matched_ref_points, axis=0)
            query_std = np.std(matched_query_points, axis=0)
            distribution_score = min(1.0, (np.mean(ref_std) + np.mean(query_std)) / 100.0)
        else:
            distribution_score = 0.0
        
        # 综合质量分数
        overall_quality = 0.6 * quantity_score + 0.4 * distribution_score
        
        return float(overall_quality)
    
    def _create_failed_result(self, start_time: float, reason: str) -> OpticalMatchResult:
        """创建失败结果"""
        processing_time = time.time() - start_time
        
        self.logger.log_event("warning", "positioning_failed", {"reason": reason})
        
        return OpticalMatchResult(
            position=Position3D(0.0, 0.0, 0.0, time.time(), 0.0),
            confidence=0.0,
            num_matches=0,
            match_quality=0.0,
            processing_time=processing_time,
            timestamp=time.time()
        )
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息
        
        Returns:
            系统信息字典
        """
        return {
            "device": str(self.config.device),
            "max_keypoints": self.config.max_keypoints,
            "min_matches": self.config.min_matches,
            "reference_map_loaded": self.map_manager.reference_map is not None,
            "output_path": self.config.output_path
        }