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

from typing import Tuple, Union
from dataclasses import dataclass
from model.superpoint import SuperPoint
from model.lightglue import LightGlue
from model.drone_position_model import DronePositionPredictor
from server.mavlink import CustomSITL
from utils.logger import Logger
from utils.pair import inference, get_center_aim, pixel_to_geolocation, visualize_and_save_matches, get_bbox_geo, visualize_and_save_bboxes,save_origin_img
from torchvision.transforms import ToTensor
from utils.pair import (crop_geotiff_by_center_point, save_coordinates_to_csv, get_m_nums, crop_geotiff_by_pixel_point)
from utils.elevation import get_elevation_from_hgt, get_elevation_bilinear, get_elevation_from_tif_file, \
    get_elevation_bilinear_from_file
import argparse
import torch
import timeit
import numpy as np
import os
import csv
from osgeo import gdal
from server.detection import YoloConfig, YoloeDetector
from geopy.distance import geodesic

# from camera.camera import DCamera
# from camera.gimbal import Gimbal
from datetime import datetime


@dataclass
class AppConfig:
    ste_path : str
    output_path : str
    dsm_path : str
    device: torch.device
    args: argparse.Namespace
    extractor: SuperPoint
    matcher: LightGlue
    logger: Logger
    sitl: CustomSITL
    detector_config: YoloConfig
    detector: YoloeDetector
    calibrator: DronePositionPredictor
    # camera:Union[DCamera, Gimbal, None] = None

class OptMatch:

    def __init__(self, app_config: AppConfig):
        self.config = app_config  # 添加配置存储
        self.output_path = self.config.args['data']['output_path']
        self.camera=None
        self.input_ste = gdal.Open(self.config.args['data']['image_ste_path'])
        self.res_path = os.path.join(self.output_path, 'res_img')
        self.fault_path = os.path.join(self.output_path, 'fault_img')
        os.makedirs(self.res_path, exist_ok=True)
        os.makedirs(self.fault_path, exist_ok=True)

    def process_image_matching(self, image_ste, real_img):
        """核心图像匹配处理流程
        Args:
            image_ste: 卫星基准图像
            real_img: 无人机实时图像
        Returns:
            matches_S_U: 匹配的特征点对
            matches_num: 匹配的特征点对数量
            m_kpts_ste: 匹配的特征点在卫星基准图像中的索引
            m_kpts_uav: 匹配的特征点在无人机实时图像中的索引
            ste_keypoints: 卫星基准图像中的特征点
            ste_scores: 卫星基准图像中特征点的置信度
            uav_keypoints: 无人机实时图像中的特征点
            uav_scores: 无人机实时图像中特征点的置信度
            matches_scores: 匹配的特征点对的置信度
        """
        start_time = timeit.default_timer()

        matches_S_U, matches_num, m_kpts_ste, m_kpts_uav, ste_keypoints, ste_scores, uav_keypoints, uav_scores, matches_scores \
            = inference(
            image_ste, real_img,
            self.config.extractor,
            self.config.matcher,
            self.config.device
        )

        elapsed_time = (timeit.default_timer() - start_time) * 1000
        self.config.logger.log(f"推理时间: {elapsed_time:.2f} 毫秒, FPS={1000 / elapsed_time:.1f}")
        return matches_S_U, matches_num, m_kpts_ste, m_kpts_uav, ste_keypoints, ste_scores, uav_keypoints, uav_scores, matches_scores

    def save_match_keypoints_geo_to_csv(self, m_kpts_ste, ste_keypoints, ste_scores, keypoint_csv_file, img_ste_geo):
        """保存匹配到的关键点的地理信息到CSV文件"""
        table_title = ["成功匹配的特征点latitude", "成功匹配的特征点longitude"]
        file_exists = os.path.exists(keypoint_csv_file)
        with open(keypoint_csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(table_title)  # 表头
            for i, ste_keypoints in enumerate(m_kpts_ste):
                lon, lat = pixel_to_geolocation(
                    ste_keypoints[0] + 0.5,
                    ste_keypoints[1] + 0.5,
                    img_ste_geo
                )
                content = [lat, lon]
                writer.writerow(content)  # 写入图像名称和对应的坐标

    def save_keypoints_geo_to_csv(self, ste_keypoints, ste_scores, keypoint_csv_file, img_ste_geo):
        """保存关键点的地理信息到CSV文件"""
        table_title = ["卫星图特征点latitude", "卫星图特征点longitude", "卫星图特征点置信度"]
        file_exists = os.path.exists(keypoint_csv_file)
        with open(keypoint_csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(table_title)  # 表头
            for i, ste_keypoints in enumerate(ste_keypoints):
                lon, lat = pixel_to_geolocation(
                    ste_keypoints[0] + 0.5,
                    ste_keypoints[1] + 0.5,
                    img_ste_geo
                )
                content = [lat, lon, ste_scores[i]]
                writer.writerow(content)  # 写入图像名称和对应的坐标
                
    def get_elevation_data(self, lat, lon):
        """获取指定经纬度位置的高程数据
        
        Args:
            lat (float): 纬度
            lon (float): 经度
            
        Returns:
            float: 高程值（单位：米），如果无法获取则返回None
        """
        # 首先尝试使用双线性插值获取更精确的高程数据
        elevation = get_elevation_bilinear(lat, lon)
        
        # 如果双线性插值失败，尝试使用普通方法获取
        if elevation is None:
            elevation = get_elevation_from_hgt(lat, lon)
            
        if elevation is not None:
            self.config.logger.log(f"获取高程数据成功：{elevation} 米")
        else:
            self.config.logger.log(f"无法获取高程数据，使用默认高度")
            
        return elevation

    def process_ste_extractor_point(self, config: AppConfig, win_size: Tuple[int, int]):
        """处理提取器点"""
        winx, winy = win_size
        #读取self.input_ste图像并从左到右使用win_size大小的窗口进行滑窗
        for x in range(0, self.input_ste.RasterXSize, winx):
            self.config.logger.log(f"正在处理第{x}/{self.input_ste.RasterXSize}行")
            for y in range(0, self.input_ste.RasterYSize, winy):
                image_ste, img_ste_geo = crop_geotiff_by_pixel_point(
                    x, y,
                    input_tif=self.input_ste,
                    crop_size_px=winx,
                    crop_size_py=winy
                )
                # 将图像转换为Tensor并移动到指定设备
                transform = ToTensor()
                image_ste = transform(image_ste).to(self.config.device)

                # 提取两张图像的特征
                feats_ste = self.config.extractor.extract(image_ste)

                ste_keypoints, ste_scores = feats_ste['keypoints'].cpu().numpy(), feats_ste['keypoint_scores'].cpu().numpy()
                keypoint_csv_file = os.path.join(self.output_path, 'keypoint_geo.csv')
                self.save_keypoints_geo_to_csv(ste_keypoints[0], ste_scores[0], keypoint_csv_file,
                                               img_ste_geo)

    def process_frame_matching(self, config: AppConfig, position_data, frame_img, win_size: Tuple[int, int], file_name):
        """处理无人机图像测试流程单帧图像匹配"""
        true_lat, true_lon, true_alt, roll, pitch, heading = position_data
        # 获取该位置的高程数据
        if config.dsm_path is not None:
            # 从DSM.tif文件中读取高程信息
            elevation = get_elevation_from_tif_file(true_lat, true_lon, config.dsm_path)
        else:
            elevation = self.get_elevation_data(true_lat, true_lon)
        true_alt = true_alt - 1483
        winx, winy = int(7000 * (true_alt/150)), int(4000 * (true_alt/150))
        winx, winy = int(3000 * (true_alt / 150)), int(2000 * (true_alt / 150))
        config.logger.log(f'winx: {winx}, winy: {winy}')
        # winx, winy = (8000, 5000)
        try:
            image_ste, img_ste_geo, _, _ = crop_geotiff_by_center_point(
                longitude=true_lon, latitude=true_lat,
                input_tif=self.input_ste,
                crop_size_px=winx,
                crop_size_py=winy
            )
        except Exception as e:
            config.logger.log(f"图像裁剪失败: {e}")
            return None, None

        # 核心匹配处理
        # 使用opencv对 frame_img 下采样处理节省计算开销
        frame_img = frame_img.resize((int(frame_img.width/10), int(frame_img.height/10)))
        matches_S_U, matches_num, m_kpts_ste, m_kpts_uav, ste_keypoints, ste_scores, uav_keypoints, uav_scores, matches_scores \
            = self.process_image_matching(
            image_ste, frame_img
        )
        #检测目标
        detector = config.detector
        labels, detections = detector.detect(frame_img)
        keypoint_csv_file = os.path.join(self.config.output_path, 'matched_keypoint_geo.csv')
        self.save_match_keypoints_geo_to_csv(m_kpts_ste.cpu().numpy(), ste_keypoints[0], ste_scores[0], keypoint_csv_file, img_ste_geo)

        if matches_num > 8:
            aim = get_center_aim(winx, winy, m_kpts_ste, m_kpts_uav, matches_scores)
            #判断是否有目标
            if len(labels) == 0:
                config.logger.log(f"未检测到目标：{file_name}")
            else:
                config.logger.log(f"检测到目标：{file_name}")
                detections = get_bbox_geo(winx, winy, m_kpts_ste, m_kpts_uav, labels, detections, img_ste_geo)
                vis_path = os.path.join(self.output_path, f"{file_name}_bbox.jpg")
                visualize_and_save_bboxes(image_ste, frame_img, m_kpts_ste, m_kpts_uav, matches_S_U, vis_path, detections, labels)
            lon, lat = pixel_to_geolocation(
                aim[0] + 0.5,
                aim[1] + 0.5,
                img_ste_geo
            )

            
            current_lat, current_lon = int(lat * 1e7), int(lon * 1e7)
            
            # 将高程数据用于后续处理
            elevation_value = elevation
            
            config.logger.log(f"匹配成功：{file_name}，地表高程：{elevation_value} 米，真高：{true_alt} 米")
            # output_geo = self.config.calibrator.predict_position(lat, lon, roll, pitch, heading, true_alt)
            # 计算两个坐标之间的距离（单位：米）
            output_geo = (lat, lon)
            true_geo = (true_lat, true_lon)
            distance = geodesic(true_geo, output_geo).meters
            distance = np.float32(distance)
            config.logger.log(
                f"文件名：{file_name} 真实坐标: {true_lat}, {true_lon}, 计算坐标: {lat}, {lon}"
                f"距离差: {distance:.2f}米")
            
            # 更新CSV内容
            content = [
                f"{file_name}", true_lat, true_lon,
                lat, lon, 
                distance,
                matches_num,
                roll, pitch, heading, true_alt, elevation_value
            ]
            title = [
                "Image Name", "true_lat", "true_lon", 
                "compute_lat", "compute_lon", 
                "distance(m)",
                "匹配点数",
                "roll", "pitch", "heading", "altitude", "elevation(m)"
            ]
            csv_file = os.path.join(self.config.output_path, 'image_coordinates.csv')
            save_coordinates_to_csv(csv_file, content, title)

            vis_path = os.path.join(self.res_path, f"{file_name}.jpg")
            visualize_and_save_matches(image_ste, frame_img, m_kpts_ste, m_kpts_uav, matches_S_U, vis_path)
            return current_lat, current_lon

        else:
            vis_path = os.path.join(self.fault_path, f"{file_name}.jpg")
            content = [f"{file_name}.jpg", true_lat, true_lon, matches_num]
            title = ["Image Name", "true_lat", "true_lon", "匹配点数"]
            csv_file = os.path.join(self.fault_path, 'image_coordinates.csv')  # 使用实例配置
            save_coordinates_to_csv(csv_file, content, title)
            visualize_and_save_matches(image_ste, frame_img, m_kpts_ste, m_kpts_uav, matches_S_U, vis_path)
            config.logger.log(f"匹配失败：{file_name}.jpg, 匹配点数为: {matches_num}")
            return None, None

    def process_camera_matching(self,config:AppConfig, camera_img, image_ste, win_size: Tuple[int, int], csv_file: str):
        """处理相机测试流程传入图像匹配"""
        winx, winy = win_size
        #传入的相机影像数据直接是numpy的mat形式，要将底图input_ste从gdal转换成numpy形式
        # 核心匹配处理

        matches_S_U, matches_num, m_kpts_ste, m_kpts_uav, ste_keypoints, ste_scores, uav_keypoints, uav_scores, matches_scores \
            = self.process_image_matching(
            image_ste,camera_img
        )

        if matches_num > config.args.num_keypoints / 100:
            aim = get_center_aim(winx, winy, m_kpts_ste, m_kpts_uav,matches_scores)
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')[:-3]

            if config.args.image_save=="visual":
                start_time = timeit.default_timer()
                vis_path = os.path.join(self.output_path, f"{timestamp}.jpg")
                visualize_and_save_matches(image_ste,camera_img, m_kpts_ste, m_kpts_uav, matches_S_U, vis_path)
                # content = [timestamp, aim[0], aim[1]]
                elapsed_time = (timeit.default_timer() - start_time) * 1000  # 单位：毫秒
                config.logger.log(f"保存图片耗时：{elapsed_time}")
                # title = ["Time_stamp", "Aim_Longitude", "Aim_Latitude"]
                # save_coordinates_to_csv(csv_file, content, title)
            return aim
        else:
            config.logger.log(f"搜寻失败, 匹配点数为: {matches_num}")
            return None, None

    def process_image_data(self, config: AppConfig, position_data, camera_img, csv_file: str):
        """处理飞行流程图像数据"""

        REAL_lat, REAL_lon, PRESS_alt, GLOBAL_alt, GLOBAL_lat, GLOBAL_lon = position_data
        config.logger.log(f"开始处理{GLOBAL_lon}, {GLOBAL_lat}")

        output_path = config.args.save_path
        fault_path = config.args.fault_path
        # 图像裁剪处理
        start_time = timeit.default_timer()
        matches_num=0
        winx,winy=0,0
        if config.args.camera_mode =="none" and config.args.fly_mode == "sim":
        #没有相机输入，裁剪底图两次作为对比
            win_size = (1024, 1024)
            winx, winy = win_size
            image_ste, img_ste_geo, _, _ = crop_geotiff_by_center_point(
            longitude=GLOBAL_lon, latitude=GLOBAL_lat,
            input_tif=self.input_ste,
            crop_size_px=winx,
            crop_size_py=winy
            )
            frame_img, real_geo, _, _ = crop_geotiff_by_center_point(
                longitude=REAL_lon, latitude=REAL_lat,
                input_tif=self.input_ste,
                crop_size_px=winx,
                crop_size_py=winy
            )
        else:
            win_size = (camera_img.shape[1], camera_img.shape[0])
            winx, winy = win_size
            image_ste, img_ste_geo, _, _ = crop_geotiff_by_center_point(
            longitude=GLOBAL_lon, latitude=GLOBAL_lat,
            # longitude=114.0414837, latitude=33.9917234, #测试暂时，后续删除
            input_tif=self.input_ste,
            crop_size_px=winx,
            crop_size_py=winy
            )
            frame_img = camera_img
        # 核心匹配处理

        elapsed_time = (timeit.default_timer() - start_time) * 1000
        config.logger.log(f"裁剪时间: {elapsed_time:.2f} 毫秒, FPS={1000 / elapsed_time:.1f}")
        # 核心匹配处理
        matches_S_U, matches_num, m_kpts_ste, m_kpts_uav, ste_keypoints, ste_scores, uav_keypoints, uav_scores, matches_scores = self.process_image_matching(
            image_ste, frame_img
        )
        #获取高程数据
        COMPUTED_alt = PRESS_alt if abs(PRESS_alt-GLOBAL_alt)>8 else GLOBAL_alt

        if matches_num > config.args.num_keypoints / 15:
            aim = get_center_aim(winx, winy, m_kpts_ste, m_kpts_uav, matches_scores)

            lon, lat = pixel_to_geolocation(
                aim[0] + 0.5,  
                aim[1] + 0.5,  
                img_ste_geo
            )

            config.sitl.update_global_position(
                current_lat = lat,
                current_lon = lon,
                current_alt = COMPUTED_alt
            )

            aim_geo = (lon, lat)
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')[:-3]

            # 获取该位置的高程数据
            elevation = self.get_elevation_data(aim_geo[1], aim_geo[0])
            elevation_value = elevation if elevation is not None else COMPUTED_alt
            title = ["Image Name", "compute_lat", "compute_lon", "compute_alt","GLOBAL_lat", "GLOBAL_lon", "GLOBAL_alt", "PRESS_alt","SIM_lon", "SIM_alt"]
            content = [f"{timestamp}", aim_geo[1], aim_geo[0], COMPUTED_alt, GLOBAL_lat,GLOBAL_lon,GLOBAL_alt, PRESS_alt, REAL_lon, REAL_lat]
            if config.args.fly_mode == "sim":
                config.logger.log(
                    f"真实坐标: {REAL_lat}, {REAL_lon}, 计算坐标: {aim_geo[1]}, {aim_geo[0]}, 飞控仿真坐标: {GLOBAL_lat}, {GLOBAL_lon}")
            # 将图像名称和对应的地理坐标保存到 CSV 文件

            save_coordinates_to_csv(csv_file, content, title)

            config.logger.log(f"匹配成功,总耗时：{(timeit.default_timer() - start_time) * 1000:.2f}")
            vis_path = os.path.join(output_path,f"{timestamp}.jpg")
            if config.args.image_save == "visual" and config.args.match_mode == "match":
                visualize_and_save_matches(image_ste, frame_img, m_kpts_ste, m_kpts_uav, matches_S_U, vis_path)
            elif config.args.image_save == "origin":
                #或仅保存原图，速度会快一些
                save_origin_img(frame_img, vis_path)
            if config.args.match_mode == "yolo":
                # 进行目标检测
                detector = config.detector
                labels, detections = detector.detect(frame_img)
                keypoint_csv_file = os.path.join(self.output_path, 'matched_keypoint_geo.csv')
                self.save_match_keypoints_geo_to_csv(m_kpts_ste.cpu().numpy(), ste_keypoints[0], ste_scores[0], keypoint_csv_file, img_ste_geo)
                if len(labels) > 0:
                    config.logger.log(f"检测到目标：{labels}")
                    #目标信息位于detections的lonlat_geo字段中，
                    detections = get_bbox_geo(winx, winy, m_kpts_ste, m_kpts_uav, labels, detections, img_ste_geo)
                    # 可视化检测结果
                    if config.args.image_save == "visual":
                        vis_path = os.path.join(output_path, f"{timestamp}_detected.jpg")
                        visualize_and_save_bboxes(image_ste, frame_img, m_kpts_ste, m_kpts_uav, matches_S_U, vis_path, detections, labels)
                else:
                    config.logger.log(f"未检测到目标")
        else:
            if config.args.boundary_extension:
                directions = [(0, 1000), (0, -1000), (-1000, 0), (1000, 0)]
                aims = []
                for dx, dy in directions:
                    n_coord = (GLOBAL_lon + dx * img_ste_geo[1], GLOBAL_lat + dy * img_ste_geo[5])

                    image_ste, img_ste_geo, _, _ = crop_geotiff_by_center_point(
                        longitude=n_coord[0], latitude=n_coord[1],
                        input_tif=self.input_ste,
                        crop_size_px=winx,
                        crop_size_py=winy)
                    matches_S_U, matches_num, m_kpts_ste, m_kpts_uav, ste_keypoints, ste_scores, uav_keypoints, uav_scores, matches_scores = self.process_image_matching(
                        image_ste, frame_img
                    )
                    aims.append((n_coord, matches_num))
                # 选取匹配数量最高的结果
                max_aim = max(aims, key=get_m_nums)
                config.logger.log(f"搜寻失败，尝试边界拓展：{GLOBAL_lat}, {GLOBAL_lon}.jpg")
            else:
                # config.sitl.update_global_position(
                # current_lat = GLOBAL_lat,
                # current_lon = GLOBAL_lon,
                # current_alt = COMPUTED_alt
                # )#测试暂用，后续删除
                timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')[:-3]
                title = ["Image Name", "compute_lat", "compute_lon", "compute_alt","GLOBAL_lat", "GLOBAL_lon", "GLOBAL_alt", "PRESS_alt","SIM_lon", "SIM_alt"]
                content = [f"{timestamp}", "", "", COMPUTED_alt, GLOBAL_lat,GLOBAL_lon,GLOBAL_alt, PRESS_alt, REAL_lon, REAL_lat]
                save_coordinates_to_csv(csv_file, content, title)
                vis_path = os.path.join(output_path, f"{timestamp}.jpg")
                if config.args.image_save == "visual":

                    visualize_and_save_matches(image_ste, frame_img, m_kpts_ste, m_kpts_uav, matches_S_U, vis_path)
                elif config.args.image_save == "origin":
                #或仅保存原图，速度会快一些
                    save_origin_img(frame_img, vis_path)
                config.logger.log(f"搜寻失败，跳过：{GLOBAL_lat}, {GLOBAL_lon}")

    def run(self):
        """主控制循环"""
        # csv_file = os.path.join(self.output_path, 'image_coordinates.csv')  # 使用实例配置
        #修改保存csv的逻辑
        csv_dir = os.path.join(self.output_path, 'csv')
        os.makedirs(csv_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        csv_filename = f'{timestamp}.csv'
        csv_file = os.path.join(csv_dir, csv_filename)


        camera_img = None
        inx =0
        while True:
            try:
                if self.config.args['mavlink']['fly_mode']=="sim":
                    #模拟飞行时在空中关闭GPS
                    inx += 1
                    if inx == 10:
                        self.config.sitl.SIM_GPS_DISABLE(1)
                    if inx ==20:
                        self.config.sitl.set_mode("AUTO")
                    if inx == 30 :
                        self.config.sitl.SIM_RC_FAIL(1)
                if self.config.camera is not None:
                    print(f"尝试获取影像")              
                    try:
                        camera_img = self.config.camera.get_img()
                        print(f"获取相机图像成功")               
                    except Exception as e:
                        self.config.logger.log(f"获取相机图像时出错: {str(e)}")
                        camera_img = None
                # 获取定位数据
                self.config.sitl.refresh_msg(self.config.args['mavlink']['fly_mode'])
                position_data = self.config.sitl.get_global_position()

                # 图像处理流程
                self.process_image_data(self.config, position_data, camera_img, csv_file)
            except KeyboardInterrupt:
                print("程序中断，退出")
                if self.config.camera is not None:
                    self.config.camera.release()
                break
