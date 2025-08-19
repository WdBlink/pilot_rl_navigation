import cv2, torch
from model.utils import load_image, rbd
from utils import viz2d
import matplotlib.pyplot as plt
import csv
import numpy as np
import glob
import os
from pathlib import Path
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, Normalize, ColorJitter, Grayscale
import torchvision.transforms.functional as TF
from osgeo import gdal, osr, ogr
import csv
import math

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']

import random


def get_variable(coord, p):
    probability = random.random()  # 生成0到1之间的随机数

    # 假设a为1的概率是p，那么a为2的概率就是1-p
    # 这里我们假设a为1的概率是0.5，a为2的概率也是0.5，你可以根据实际情况调整这些概率
    if probability < p:
        return coord
    else:
        return (-100000000, 100000000)


def draw_points_on_image(image_path, points, output_path):
    # 读取输入图像
    image = cv2.imread(image_path)

    # 将图像转换为RGB格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 在图像上标点

    x, y = int(points[0]), int(points[1])
    cv2.circle(image_rgb, (x, y), 25, (255, 0, 0), -1)  # 在图像上标点
    image_rgb = cv2.resize(image_rgb, (5000, 5000))

    # 显示带有标点的图像
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()

    # 保存带有标点的图像
    cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def pad_to_match_dimension(tensor_to_pad, reference_tensor):
    """
    将 tensor_to_pad 填充至与 reference_tensor 相同的维度。

    参数:
        tensor_to_pad (torch.Tensor): 需要填充的张量。
        reference_tensor (torch.Tensor): 作为参考的张量，被用于确定填充后的维度。

    返回:
        torch.Tensor: 填充后的张量，其维度与 reference_tensor 相同。
    """
    in_shape = len(tensor_to_pad.shape)
    if in_shape > 2:
        tensor_to_pad = tensor_to_pad.squeeze(0)
        reference_tensor = reference_tensor.squeeze(0)
    # 计算填充的行数和列数
    padding_rows = reference_tensor.shape[0] - tensor_to_pad.shape[0]
    padding_cols = reference_tensor.shape[1] - tensor_to_pad.shape[1]

    # 使用 torch.nn.functional.pad() 函数填充
    # 该函数以（左填充，右填充，上填充，下填充）的方式进行填充
    # 这里我们将右填充和下填充设置为相应的行数和列数，其余填充设置为0
    padded_tensor = torch.nn.functional.pad(tensor_to_pad, (0, padding_cols, 0, padding_rows))
    if in_shape > 2:
        padded_tensor = padded_tensor.unsqueeze(0)
    return padded_tensor


def get_m_nums(aim):
    return aim[1]


def visualize_and_save_matches(image_ste, image_uav, m_kpts_ste, m_kpts_uav, matches_S_U, output_path):
    """
    Visualize images and their matches and save the result to the output_path.

    Parameters:
        image_ste (numpy.ndarray): Image from source.
        image_uav (numpy.ndarray): Image from target.
        m_kpts_ste (list): Keypoints from source.
        m_kpts_uav (list): Keypoints from target.
        matches_S_U (dict): Matches between source and target.
        output_path (str): Path to save the visualization.
    """
    image_ste = np.array(image_ste)
    # image_ste = image_ste.cpu().permute(1, 2, 0).numpy()
    image_uav = np.array(image_uav)
    axes = viz2d.plot_images([image_ste, image_uav])
    viz2d.plot_matches(m_kpts_ste, m_kpts_uav, color="lime", lw=0.15)
    # viz2d.add_text(0, f'Stop after {matches_S_U["stop"]} layers', fs=20)
    plt.savefig(output_path)
    plt.close()

def visualize_and_save_bboxes(image_ste, image_uav, m_kpts_ste, m_kpts_uav, matches_S_U, vis_path, detections, labels):
    """
    Visualize images and their matches and bounding box and save the result to the output_path.
    Parameters:
        image_ste (numpy.ndarray): Image from source.
        image_uav (numpy.ndarray): Image from target.
        m_kpts_ste (list): Keypoints from source.
        m_kpts_uav (list): Keypoints from target.
        matches_S_U (dict): Matches between source and target.
        vis_path (str): Path to save the visualization.
        detections(dict): Detections from source.
        labels(list): Labels from source.
    """
    image_ste = np.array(image_ste)
    # image_ste = image_ste.cpu().permute(1, 2, 0).numpy()
    image_uav = np.array(image_uav)
    
    # Create figure and axes explicitly
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display images on the axes
    axes[0].imshow(image_ste)
    axes[1].imshow(image_uav)
    
    # Turn off axis labels
    for ax in axes:
        ax.set_axis_off()
    
    # Plot matches between the images
    viz2d.plot_matches(m_kpts_ste, m_kpts_uav, color="lime", lw=0.15, axes=axes)
    
    # Draw bounding boxes on satellite image
    if 'xyxy_ste' in detections and len(detections['xyxy_ste']) > 0:
        for i, bbox in enumerate(detections['xyxy_ste']):
            x1, y1, x2, y2 = bbox
            class_id = detections['class_id'][i] if 'class_id' in detections and i < len(detections['class_id']) else 0
            label = labels[class_id] if class_id < len(labels) else "未知"
            confidence = detections['confidence'][i] if 'confidence' in detections and i < len(detections['confidence']) else 0
            
            # Draw rectangle on satellite image
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2)
            axes[0].add_patch(rect)
            axes[0].text(x1, y1 - 5, f'{label} {confidence:.2f}', color='red', fontsize=8, 
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
    
    # Draw bounding boxes on UAV image
    if 'xyxy' in detections and len(detections['xyxy']) > 0:
        for i, bbox in enumerate(detections['xyxy']):
            x1, y1, x2, y2 = bbox
            class_id = detections['class_id'][i] if 'class_id' in detections and i < len(detections['class_id']) else 0
            label = labels[class_id] if class_id < len(labels) else "未知"
            confidence = detections['confidence'][i] if 'confidence' in detections and i < len(detections['confidence']) else 0
            
            # Draw rectangle on UAV image
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='blue', linewidth=2)
            axes[1].add_patch(rect)
            axes[1].text(x1, y1 - 5, f'{label} {confidence:.2f}', color='blue', fontsize=8, 
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(vis_path)
    plt.close()

def save_origin_img(image_uav,output_path):


    # 保存图像
    cv2.imwrite(output_path, image_uav)



def extract_number(filename):
    return int(filename.split('/')[-1].split('.')[0])


def read_coordinates(file_path):
    """
    从给定路径的txt文件中读取坐标信息，并以字典列表的形式返回，其中每个字典包含id、经度和纬度。
    :param file_path: 包含坐标的txt文件路径
    :return: 字典列表，格式为 [{'id': id, 'longitude': longitude, 'latitude': latitude}, ...]
    """
    # 初始化结果列表
    coordinates_by_id = []
    with open(file_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        # # 跳过表头（如果有的话）
        # next(reader)  # 如果第一行是标题行，则取消注释此行
        for row in reader:
            try:
                # 假设第一列是id，后面两列分别是经度和纬度
                id_ = row[0]
                longitude = float(row[2])
                latitude = float(row[1])
                # 将每行的数据打包成一个字典，并添加到列表中
                coordinates_by_id.append({
                    'id': id_,
                    'longitude': longitude,
                    'latitude': latitude
                })
            except (IndexError, ValueError):
                # 如果某一行数据格式不正确，则忽略该行
                pass

    return coordinates_by_id


def inference(image_ste, image_uav, extractor, matcher, device):
    """
    对输入的两张图像进行特征提取和匹配，并返回匹配结果。

    Args:
        image_ste (PIL.Image.Image): 基准图像。
        image_uav (PIL.Image.Image): 实拍图像。
        extractor (object): 特征提取器对象。
        matcher (object): 特征匹配器对象。
        device (torch.device): 运行设备，如 'cuda' 或 'cpu'。

    Returns:
        dict: 匹配结果字典，包含匹配信息。
        int: 匹配点的数量。
        torch.Tensor: 基准图像的匹配关键点坐标。
        torch.Tensor: 实拍图像的匹配关键点坐标。
    """
    # 将图像转换为Tensor并移动到指定设备
    transform = ToTensor()
    image_ste = transform(image_ste).to(device)
    image_uav = transform(image_uav).to(device)

    # 提取两张图像的特征
    feats_ste = extractor.extract(image_ste)
    feats_uav = extractor.extract(image_uav)

    # 检查两张图像的关键点形状是否一致，如果不一致则进行填充
    if feats_ste['keypoints'].shape != feats_uav['keypoints'].shape:
        # 填充关键点坐标
        feats_ste['keypoints'] = pad_to_match_dimension(feats_ste['keypoints'], feats_uav['keypoints'])
        # 填充关键点分数
        feats_ste['keypoint_scores'] = pad_to_match_dimension(feats_ste['keypoint_scores'],
                                                              feats_uav['keypoint_scores'])
        # 填充特征描述符
        feats_ste['descriptors'] = pad_to_match_dimension(feats_ste['descriptors'], feats_uav['descriptors'])

    ste_keypoints, ste_scores = feats_ste['keypoints'], feats_ste['keypoint_scores']
    uav_keypoints, uav_scores = feats_uav['keypoints'], feats_uav['keypoint_scores']
    # 进行特征匹配
    matches_S_U = matcher({"image0": feats_ste, "image1": feats_uav})

    # 将匹配结果从设备中取出并转换为numpy数组
    feats_ste, feats_uav, matches_S_U = [rbd(x) for x in [feats_ste, feats_uav, matches_S_U]]

    # 提取关键点和匹配信息
    kpts_ste, kpts_uav, matches, maches_scores = feats_ste["keypoints"], feats_uav["keypoints"], matches_S_U["matches"], matches_S_U["scores"]

    # 提取匹配的关键点
    m_kpts_ste, m_kpts_uav = kpts_ste[matches[..., 0]], kpts_uav[matches[..., 1]]

    # 计算匹配点的数量
    matches_num = matches_S_U["matches"].shape[0]

    return matches_S_U, matches_num, m_kpts_ste, m_kpts_uav, ste_keypoints, ste_scores, uav_keypoints, uav_scores, maches_scores


def get_center_aim(h, w, m_kpts_ste, m_kpts_uav, matches_scores):
    """通过单应性矩阵计算实拍图中心在基准图中的对应像素坐标
    Args:
        h (int): 基准图高度（行数）
        w (int): 基准图宽度（列数）
        m_kpts_ste: 基准图匹配关键点坐标 (tensor)
        m_kpts_uav: 实拍图匹配关键点坐标 (tensor)
        matches_scores: 匹配点的置信度分数 (tensor)
    Returns:
        tuple: 实拍图中心在基准图中的坐标 (x, y)
    """
    # 将GPU上的tensor转换为numpy数组（如果需要的话）
    if hasattr(m_kpts_ste, 'cpu'):
        m_kpts_ste = m_kpts_ste.cpu().numpy()
    if hasattr(m_kpts_uav, 'cpu'):
        m_kpts_uav = m_kpts_uav.cpu().numpy()
    
    # 输入点顺序应为 (实拍图坐标, 基准图坐标)
    Ma, _ = cv2.findHomography(m_kpts_uav, m_kpts_ste, cv2.RANSAC, 5.0)
    
    # 实拍图中心坐标（亚像素精度）
    uav_center_x = (m_kpts_uav[:,0].max() + m_kpts_uav[:,0].min()) / 2
    uav_center_y = (m_kpts_uav[:,1].max() + m_kpts_uav[:,1].min()) / 2
    pts = np.float32([[[uav_center_x, uav_center_y]]])  # 仅转换实拍图中心点
    
    # 应用单应性变换到基准图坐标系
    dst = cv2.perspectiveTransform(pts, Ma)
    cX, cY = dst[0][0]
    
    # 边界检查
    cX = np.clip(cX, 0, w-1)
    cY = np.clip(cY, 0, h-1)
    return (cX, cY)


def get_bbox_geo(h, w, m_kpts_ste, m_kpts_uav, labels, detections, geotransform):
    """通过单应性矩阵计算目标检测框在基准图中的对应地理坐标
    Args:
        h (int): 基准图高度（行数）
        w (int): 基准图宽度（列数）
        m_kpts_ste: 基准图匹配关键点坐标 (tensor)
        m_kpts_uav: 实拍图匹配关键点坐标 (tensor)
        labels: 基准图目标检测框标签 (list)
        detections: 实拍图目标检测框信息 (dict)
        geotransform: 基准图地理变换元组 (tuple)
    Returns:
        tuple: 实拍图中心在基准图中的坐标 (x, y)
    """
    # 将GPU上的tensor转换为numpy数组
    m_kpts_ste, m_kpts_uav = m_kpts_ste.cpu().numpy(), m_kpts_uav.cpu().numpy()

    # 输入点顺序应为 (实拍图坐标, 基准图坐标)
    Ma, _ = cv2.findHomography(m_kpts_uav, m_kpts_ste, cv2.RANSAC, 5.0)

    # 将detections转换为字典
    detections = {
        'class_id': detections.class_id,
        'confidence': detections.confidence,
        'xyxy': detections.xyxy,
        'mask': detections.mask,
        'xyxy_ste': [],
        'lonlat_geo': []
    }
    # 所有检测框的xyxy像素坐标
    xyxy = detections['xyxy']

    #遍历所有检测框的xyxy像素坐标，应用单应性变换到基准图坐标系之后，再存入到detections[xyxy_ste]中
    for i in range(len(xyxy)):
        pts = np.float32([[[xyxy[i][0], xyxy[i][1]]]])
        dst = cv2.perspectiveTransform(pts, Ma)
        x1, y1 = dst[0][0]
        x1, y1= np.clip(x1, 0, w - 1), np.clip(y1, 0, h - 1) #边界检查
        lon_1, lat_1 = pixel_to_geolocation(x1, y1, geotransform)
        pts = np.float32([[[xyxy[i][2], xyxy[i][3]]]])
        dst = cv2.perspectiveTransform(pts, Ma)
        x2, y2 = dst[0][0]
        x2, y2 = np.clip(x2, 0, w - 1), np.clip(y2, 0, h - 1) #边界检查
        lon_2, lat_2 = pixel_to_geolocation(x2, y2, geotransform)
        detections['xyxy_ste'].append([x1, y1, x2, y2])
        detections['lonlat_geo'].append([lon_1, lat_1, lon_2, lat_2])
    return detections


def list_files(directory):
    p = str(Path(directory).absolute())  # os-agnostic absolute path
    if '*' in p:
        files = sorted(glob.glob(p, recursive=True))  # glob
    elif os.path.isdir(p):
        files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
    elif os.path.isfile(p):
        files = [p]  # files
    else:
        raise Exception(f'ERROR: {p} does not exist')
    image_format = files[0].split('.')[-1].lower()
    file_list = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]

    file_list = sorted(file_list, key=extract_number)
    return file_list, image_format


def pixel_to_geolocation(x_pixel, y_pixel, geotransform):
    """
    将图像坐标系中的像素坐标转换为实际地理坐标（经纬度）

    参数：
    x_pixel (float): 图像x轴上的像素位置
    y_pixel (float): 图像y轴上的像素位置
    geotransform (tuple of 6 floats): 地理变换元组，形式如：(top_left_x, pixel_width, rotation0, top_left_y, rotation1, pixel_height)

    返回：
    (lon, lat): 经纬度坐标对
    """

    # 地理变换参数解释
    # top_left_x, top_left_y 是图像左上角在地理坐标系中的坐标
    # pixel_width 和 pixel_height 分别是每个像素对应的地理单位长度（通常是米）
    # rotation0 和 rotation1 是旋转参数，在大多数情况下它们是0

    origin_x = geotransform[0]
    pixel_width = geotransform[1]
    origin_y = geotransform[3]
    pixel_height = geotransform[5]

    # 考虑到GDAL中图像的原点位于左上角且y轴方向向下增长，需要做调整
    lon = origin_x + x_pixel * pixel_width
    lat = origin_y + y_pixel * pixel_height  # 注意这里是减法

    return lon, lat


def crop_image_by_center_point(x, y, img, crop_size_px, crop_size_py):
    # 计算裁剪区域的左上角坐标
    left = max(x - crop_size_px // 2, 0)
    top = max(y - crop_size_py // 2, 0)

    # 计算裁剪区域的右下角坐标
    right = min(x + crop_size_px // 2, img.width)
    bottom = min(y + crop_size_py // 2, img.height)

    # 裁剪图像
    cropped_img = img.crop((left, top, right, bottom))

    return cropped_img, left, top


def center_crop_with_coords(image_tensor, center, crop_size_x, crop_size_y):
    """
    根据中心坐标裁剪图像，并返回裁剪后的图像以及裁剪图像的左上角坐标在原图中的位置

    :param image_tensor: 原始图像（tensor形式）
    :param center: 中心坐标 (x, y)
    :param crop_size: 要裁剪的大小 (width, height)
    :return: 裁剪后的图像（tensor形式），裁剪图像的左上角坐标在原图中的位置 (x, y)
    """
    # 获取图像的尺寸
    image_height, image_width = image_tensor.shape[-2:]

    # 计算裁剪区域的左上角和右下角坐标
    crop_width, crop_height = crop_size_x, crop_size_y
    x_center, y_center = center
    x1 = max(0, int(x_center - crop_width / 2))
    y1 = max(0, int(y_center - crop_height / 2))
    x2 = min(image_width, int(x_center + crop_width / 2))
    y2 = min(image_height, int(y_center + crop_height / 2))

    # 裁剪图像
    cropped_image = TF.crop(image_tensor, y1, x1, y2 - y1, x2 - x1)

    return cropped_image, x1, y1


def geo_to_pixel(geo_x, geo_y, tfw_path):
    with open(tfw_path, 'r') as tfw_file:
        lines = tfw_file.readlines()
        pixel_size_x = float(lines[0])
        pixel_size_y = float(lines[3])
        origin_x = float(lines[4])
        origin_y = float(lines[5])

    pixel_x = int((geo_x - origin_x) / pixel_size_x)
    pixel_y = int((origin_y - geo_y) / pixel_size_y * (-1))

    return pixel_x, pixel_y

def geo2pixel(geotransform, lon, lat):
    lon = float(lon)
    lat = float(lat)
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(lon, lat)

    x = int((point.GetX() - geotransform[0]) / geotransform[1])
    y = abs(int((geotransform[3] - point.GetY()) / geotransform[5]))
    return x, y


def pixel_to_geo(pixel_x, pixel_y, tfw_path):
    with open(tfw_path, 'r') as tfw_file:
        lines = tfw_file.readlines()
        pixel_size_x = float(lines[0])
        pixel_size_y = float(lines[3])
        origin_x = float(lines[4])
        origin_y = float(lines[5])

    geo_x = origin_x + pixel_x * pixel_size_x
    geo_y = origin_y + pixel_y * pixel_size_y

    return geo_x, geo_y

def save_coordinates_to_csv(csv_file, content, table_title):
    """将图像文件名和对应的地理坐标保存到 CSV 文件"""
    # 如果文件不存在，则创建文件并写入表头
    file_exists = os.path.exists(csv_file)
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(table_title)  # 表头
        writer.writerow(content)  # 写入图像名称和对应的坐标

def crop_geotiff_by_center_point(longitude, latitude, input_tif, crop_size_px, crop_size_py):

    if input_tif is None:
        raise ValueError("无法打开输入的GeoTIFF文件")

    # 获取原数据集的地理参考信息
    geotransform = input_tif.GetGeoTransform()

    # 将经纬度坐标转换为图像坐标
    x, y = geo2pixel(geotransform, longitude, latitude)

    # 根据裁剪半径计算实际裁剪矩形框大小（这里简化为正方形裁剪）
    block_xsize = int(min(crop_size_px, input_tif.RasterXSize - x))
    block_ysize = int(min(crop_size_py, input_tif.RasterYSize - y))

    # 调整裁剪区域以确保裁剪圆心位于裁剪矩形中心
    offset_x = int(max(x - block_xsize // 2, 0))
    offset_y = int(max(y - block_ysize // 2, 0))

    # 从每个波段中读取裁剪区域的数据
    in_band1 = input_tif.GetRasterBand(1)
    in_band2 = input_tif.GetRasterBand(2)
    in_band3 = input_tif.GetRasterBand(3)
    out_band1 = in_band1.ReadAsArray(offset_x, offset_y, block_xsize, block_ysize)
    out_band2 = in_band2.ReadAsArray(offset_x, offset_y, block_xsize, block_ysize)
    out_band3 = in_band3.ReadAsArray(offset_x, offset_y, block_xsize, block_ysize)

    # 设置裁剪后图像的仿射变换参数
    top_left_x = geotransform[0] + offset_x * geotransform[1]
    top_left_y = geotransform[3] + offset_y * geotransform[5]

    dst_transform = (
        top_left_x, 
        geotransform[1],  # 保持原始X方向像素宽度
        geotransform[2],  # 保持原始X方向旋转参数
        top_left_y,
        geotransform[4],  # 保持原始Y方向旋转参数 
        geotransform[5]   # 保持原始Y方向像素高度（应为负数）
    )

    if geotransform[5] > 0:
        raise ValueError(f"检测到异常像素高度值 {geotransform[5]}，应为负数")
    
    rgb_crop = np.dstack((out_band1, out_band2, out_band3))
    return rgb_crop, dst_transform, offset_x, offset_y

def crop_geotiff_by_pixel_point(x, y, input_tif, crop_size_px, crop_size_py):
    """
    根据像素坐标裁剪GeoTIFF文件，并返回裁剪后的图像和地理参考信息。
    参数:
    x (int): 裁剪区域左上角的x像素坐标
    y (int): 裁剪区域左上角的y像素坐标
    input_tif (gdal.Dataset): 输入的GeoTIFF数据
    crop_size_px (int): 裁剪区域的宽度（像素）
    crop_size_py (int): 裁剪区域的高度（像素）
    返回：
    tuple: (裁剪后的RGB图像数组, 地理变换参数)
    """
    # 获取图像原始尺寸
    img_width = input_tif.RasterXSize
    img_height = input_tif.RasterYSize
    num_bands = min(input_tif.RasterCount, 3)  # 最多读取3个波段

    # 计算实际可裁剪的区域
    read_width = min(crop_size_px, img_width - x) if x >= 0 else min(crop_size_px + x, img_width)
    read_height = min(crop_size_py, img_height - y) if y >= 0 else min(crop_size_py + y, img_height)

    # 调整读取位置（处理负坐标情况）
    read_x = max(x, 0)
    read_y = max(y, 0)

    # 初始化输出数组（用0填充）
    cropped_image = np.zeros((crop_size_py, crop_size_px, 3), dtype=np.uint8)

    if read_width > 0 and read_height > 0:
        # 读取实际可读取的区域数据
        bands_data = []
        for band in range(1, num_bands + 1):
            band_data = input_tif.GetRasterBand(band).ReadAsArray(
                read_x, read_y, read_width, read_height)
            bands_data.append(band_data)

        # 将读取的数据放入正确位置
        valid_data = np.dstack(bands_data)

        # 计算在输出图像中的放置位置
        output_x = -min(x, 0)
        output_y = -min(y, 0)

        # 确保不越界
        place_width = min(valid_data.shape[1], crop_size_px - output_x)
        place_height = min(valid_data.shape[0], crop_size_py - output_y)

        # 将有效数据放入输出图像
        cropped_image[output_y:output_y + place_height,
        output_x:output_x + place_width, :] = valid_data[:place_height, :place_width, :]

    # 处理单/双波段情况
    if num_bands == 1:
        cropped_image[:, :, 1:] = cropped_image[:, :, 0:1]  # 复制单波段到所有通道
    elif num_bands == 2:
        cropped_image[:, :, 2] = 0  # 第三通道补零

    # 计算新的地理变换参数
    original_geotransform = input_tif.GetGeoTransform()
    new_geotransform = (
        original_geotransform[0] + x * original_geotransform[1],
        original_geotransform[1],
        original_geotransform[2],
        original_geotransform[3] + y * original_geotransform[5],
        original_geotransform[4],
        original_geotransform[5]
    )

    return cropped_image, new_geotransform


def compute_rotation_matrix(roll_deg, pitch_deg, yaw_deg):
    """计算从机体坐标系到ENU坐标系的旋转矩阵"""
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    # 绕Z轴（航向）的旋转矩阵
    Rz = np.array([
        [math.cos(yaw), math.sin(yaw), 0],
        [-math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])

    # 绕Y轴（俯仰）的旋转矩阵
    Ry = np.array([
        [math.cos(pitch), 0, -math.sin(pitch)],
        [0, 1, 0],
        [math.sin(pitch), 0, math.cos(pitch)]
    ])

    # 绕X轴（横滚）的旋转矩阵
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(roll), math.sin(roll)],
        [0, -math.sin(roll), math.cos(roll)]
    ])

    # 组合旋转顺序：Z -> Y -> X
    return Rz @ Ry @ Rx


def calculate_drone_position(lon_target, lat_target, roll_deg, pitch_deg, yaw_deg, h):
    """
    计算无人机的GPS位置
    :param lon_target: 目标点经度
    :param lat_target: 目标点纬度
    :param roll_deg: 横滚角（度）
    :param pitch_deg: 俯仰角（度）
    :param yaw_deg: 航向角（度）
    :param h: 无人机相对高度（米）
    :return: 无人机的经纬度坐标（经度, 纬度）
    """
    R = compute_rotation_matrix(roll_deg, pitch_deg, yaw_deg)
    v_body = np.array([0, 0, 1])  # 机体坐标系的视线方向（向下）
    v_enu = R.dot(v_body)
    vx, vy, vz = v_enu

    if vz == 0:
        raise ValueError("视线方向垂直，无法计算，vz不能为零。")

    # 计算目标点相对于无人机的东向和北向位移
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
    compute_drone = (lat_drone, lon_drone)
    return compu