"""测试无人机拍摄的图像与底图的匹配效果
测试方法：
1. 读取无人机拍摄的图像，和无人机的位置信息
2. 读取底图
3. crop底图，使其与无人机拍摄的图像在同一位置
4. 使用LightGlue模型进行匹配
5. 保存匹配结果
"""
import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# from lightglue import LightGlue, SuperPoint
from model.superpoint import SuperPoint
from model.lightglue import LightGlue
from server.mavlink import CustomSITL
from server.optmatch import AppConfig
from server.detection import YoloConfig, YoloeDetector
from model.drone_position_model import DronePositionPredictor
import torch
from utils.logger import Logger
from typing import Tuple
from dataclasses import dataclass
from server.optmatch import OptMatch
import json
from PIL import Image
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
import numpy as np
import yaml
import argparse


def load_config(config_path):
    """从yaml配置文件加载参数"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_environment(config) -> Tuple[Logger, torch.device]:
    """初始化环境和硬件连接"""
    try:
        os.makedirs(config['data']['output_path'], exist_ok=True)
    except OSError as e:
        print(f"创建保存路径时出错: {e}")

    logger = Logger(log_file=os.path.join(config['data']['output_path'], 'log.txt'))
    device = torch.device(config['device']) if config['device'] != "auto" else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.log(f"Running inference on device: {device}")
    torch.set_grad_enabled(False)
    return logger, device

def initialize_models(config, device) -> Tuple[SuperPoint, LightGlue]:
    """初始化模型"""
    extractor = SuperPoint(
        max_num_keypoints=config['extractor']['num_keypoints'],
        weight_path=os.path.join(os.path.dirname(__file__), "weights", "superpoint_v1.pth")
    ).eval().to(device)

    matcher = LightGlue(features=None, weights=config["matcher"]['weights_path']).eval().to(device)
    return extractor, matcher


def parse_args():
    """解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的命令行参数
    """
    parser = argparse.ArgumentParser(description="测试无人机拍摄的图像与底图的匹配效果")
    parser.add_argument(
        "--config", 
        type=str, 
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'edge_data_test.yaml'),
        help="配置文件路径"
    )
    return parser.parse_args()


def main():
    # 解析命令行参数
    args = parse_args()
    # 加载配置文件
    config_path = args.config
    config = load_config(config_path)
    logger, device = setup_environment(config)
    #将names中的内容转换为列表
    config['detector']['names'] = config['detector']['names'].split(',') \
        if isinstance(config['detector']['names'], str) else config['detector']['names']
    names = list(config['detector']['names'])
    # 读取标注文件
    annotation_file = config["detector"]["annotation_file"]
    with open(annotation_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    # 从标注文件中提取bboxes和cls信息
    bboxes = np.array(annotations['bboxes'], dtype=np.float32)
    cls = np.array(annotations['cls'], dtype=np.int64)
    refer_image = annotations['image']
    # 创建visual_prompts字典
    visual_prompts = dict(
        bboxes=bboxes,  # 边界框坐标
        cls=cls,  # 类别ID
    )
    extractor, matcher = initialize_models(config, device)
    yolo_config = YoloConfig(
        weight=config['detector']['weights_path'],
        device=device,
        names=names,
        refer_image=refer_image,  # Reference image used to get visual prompts
        visual_prompts=visual_prompts,
        predictor=YOLOEVPSegPredictor,
    )
    app_config = AppConfig(
        device=device,
        args=config,
        ste_path=config['data']['image_ste_path'],
        output_path=config['data']['output_path'],
        dsm_path=config['data']['dsm_path'],
        extractor=extractor,
        matcher=matcher,
        logger=logger,
        sitl=None,
        detector_config=yolo_config,
        detector=YoloeDetector(yolo_config),
        calibrator=DronePositionPredictor(config['calibrator']['weights_path'])
    )

    opt_matcher = OptMatch(app_config)

    #从UAV路径中读取文件树
    uav_files = os.listdir(config['data']['image_uav_path'])
    
    # 查找数据文件（支持txt和csv格式）
    txt_files = [file for file in uav_files if file.endswith('.txt')]
    csv_files = [file for file in uav_files if file.endswith('.csv')]
    
    parsed_data = []
    
    # 处理TXT文件
    if txt_files:
        txt_file = txt_files[0]
        logger.log(f"使用TXT文件: {txt_file}")
        with open(os.path.join(config['data']['image_uav_path'], txt_file), 'r') as f:
            uav_files = f.readlines()
            for line in uav_files:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue

                # 第一列固定为文件名
                file_name = parts[0]
                roll = parts[-3]
                heading = parts[-1]
                pitch = parts[-2]
                coords = []

                # 从剩余部分识别坐标
                for part in parts[1:]:
                    try:
                        num = float(part)
                        # 识别经纬度(通常小数点后有6位以上)
                        if '.' in part and len(part.split('.')[1]) >= 4:
                            coords.append(num)
                        # 识别高度(通常在500-2000之间)
                        elif 500 <= num <= 2000:
                            coords.append(num)
                    except ValueError:
                        continue

                # 确保找到3个坐标值(经度、纬度、高度)
                if len(coords) == 3:
                    parsed_data.append((file_name, coords[0], coords[1], coords[2], roll, heading, pitch))
    
    # 处理CSV文件
    elif csv_files:
        csv_file = csv_files[0]
        logger.log(f"使用CSV文件: {csv_file}")
        with open(os.path.join(config['data']['image_uav_path'], csv_file), 'r') as f:
            uav_files = f.readlines()
            for line in uav_files:
                # 跳过可能的标题行
                if line.strip().startswith('#') or ',' not in line:
                    continue
                    
                parts = line.strip().split(',')
                if len(parts) < 5:  # 至少需要文件名、纬度、经度、高度
                    continue
                    
                # CSV格式：第一列是文件名，第三列是纬度，第四列是经度，第五列是高度
                file_name = parts[0].strip() + ".jpg"
                # file_name = os.path.join(file_name, ".jpg")
                lat = float(parts[4].strip())
                lon = float(parts[5].strip())
                alt = float(parts[6].strip())
                
                # 如果有姿态信息，则提取
                roll = 0.0
                pitch = 0.0
                heading = 0.0
                if len(parts) > 7:  # 假设姿态信息在后面的列
                    try:
                        roll = float(parts[5].strip())
                        pitch = float(parts[6].strip())
                        heading = float(parts[7].strip())
                    except (ValueError, IndexError):
                        pass
                        
                parsed_data.append((file_name, lat, lon, alt, roll, heading, pitch))
    else:
        logger.log("错误：未找到TXT或CSV格式的数据文件")
        return
        
    # 转换为字典格式
    uav_files = {file_name: {'path': os.path.join(config['data']['image_uav_path'], file_name),
                             'lat': float(lat),
                             'lon': float(lon),
                             'alt': float(alt),
                             'roll': float(roll),
                             'heading': float(heading),
                             'pitch': float(pitch)} for file_name, lat, lon, alt, roll, heading, pitch in parsed_data}

    #从image_uav_path中逐个读取后缀为.JPG的文件
    for file_name in os.listdir(config['data']['image_uav_path']):
        if file_name.endswith('.jpg'):
            #使用PIL读取文件
            frame_img = Image.open(os.path.join(config['data']['image_uav_path'], file_name))
            #将图片缩小到1/4
            frame_img = frame_img.resize((int(frame_img.width/1), int(frame_img.height/1)))
            #获取文件的地理信息
            position_data = (uav_files[file_name]['lat'], uav_files[file_name]['lon'], uav_files[file_name]['alt'],
                             uav_files[file_name]['roll'], uav_files[file_name]['pitch'], uav_files[file_name]['heading'])
            # 获取frame_img的窗口大小
            win_size = frame_img.size[0], frame_img.size[1]
            current_lat, current_lon = opt_matcher.process_frame_matching(app_config, position_data, frame_img, win_size, file_name)


if __name__ == "__main__":
    main()

