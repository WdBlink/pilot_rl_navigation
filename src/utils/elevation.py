# -*- coding: utf-8 -*-
"""
高程数据处理模块

该模块提供了从HGT文件中读取高程数据的功能。
HGT文件是SRTM（Shuttle Radar Topography Mission）项目生成的数字高程模型（DEM）文件。
文件命名规则为：N/SxxE/Wyyy.hgt，其中xx表示纬度，yyy表示经度。
"""

import os
import math
import numpy as np
from osgeo import gdal
import sys
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_elevation_from_tif_file(lat, lon, tif_file):
    # 使用GDAL打开DSM文件
    dsm_dataset = gdal.Open(tif_file)
    if dsm_dataset:
        # 获取地理变换参数
        geotransform = dsm_dataset.GetGeoTransform()

        # 计算像素坐标（注意：经度对应x，纬度对应y）
        x = int((lon - geotransform[0]) / geotransform[1])
        y = int((lat - geotransform[3]) / geotransform[5])

        # 使用双线性插值获取更精确的高程值
        # 计算精确的像素坐标（浮点数）
        x_float = (lon - geotransform[0]) / geotransform[1]
        y_float = (lat - geotransform[3]) / geotransform[5]

        # 获取周围四个像素的整数坐标
        x0 = int(math.floor(x_float))
        y0 = int(math.floor(y_float))
        x1 = x0 + 1
        y1 = y0 + 1

        # 确保所有坐标在有效范围内
        if (0 <= x0 < dsm_dataset.RasterXSize - 1 and 0 <= y0 < dsm_dataset.RasterYSize - 1):
            # 计算插值权重
            wx = x_float - x0
            wy = y_float - y0

            # 读取四个角点的高程值
            band = dsm_dataset.GetRasterBand(1)
            data = band.ReadAsArray(x0, y0, 2, 2)

            # 检查是否有无效值
            if np.any(data == -32768):
                # 如果有无效值，回退到最近邻插值
                elevation_data = band.ReadAsArray(x, y, 1, 1)
                elevation = float(elevation_data[0, 0])
            else:
                # 执行双线性插值
                elevation = ((1 - wx) * (1 - wy) * data[0, 0] +
                             wx * (1 - wy) * data[0, 1] +
                             (1 - wx) * wy * data[1, 0] +
                             wx * wy * data[1, 1])
            return elevation
        else:
            # 坐标超出范围，使用单点读取
            if 0 <= x < dsm_dataset.RasterXSize and 0 <= y < dsm_dataset.RasterYSize:
                band = dsm_dataset.GetRasterBand(1)
                elevation_data = band.ReadAsArray(x, y, 1, 1)
                elevation = float(elevation_data[0, 0])
                if elevation != -32768:  # 如果不是无效值
                    return elevation
                else:
                    return None
            else:
                # 坐标完全超出范围，回退到常规高程获取方法
                print("坐标完全超出范围")
                return None
    else:
        # 无法打开DSM文件，回退到常规高程获取方法
        print("无法打开DSM文件")
        return None


def get_hgt_file_path(lat, lon, hgt_dir):
    """
    根据经纬度获取对应的HGT文件路径
    
    Args:
        lat (float): 纬度
        lon (float): 经度
        hgt_dir (str): HGT文件所在目录
        
    Returns:
        str: HGT文件路径，如果文件不存在则返回None
    """
    # 确定文件名前缀（N/S表示北/南纬，E/W表示东/西经）
    lat_prefix = 'N' if lat >= 0 else 'S'
    lon_prefix = 'E' if lon >= 0 else 'W'
    
    # 获取整数部分作为文件名
    lat_int = int(abs(lat))
    lon_int = int(abs(lon))
    
    # 构建文件名，格式为N/SxxE/Wyyy.hgt
    file_name = f"{lat_prefix}{lat_int:02d}{lon_prefix}{lon_int:03d}.hgt"
    file_path = os.path.join(hgt_dir, file_name)
    
    # 检查文件是否存在
    if os.path.exists(file_path):
        return file_path
    
    # 如果直接的.hgt文件不存在，尝试查找压缩文件
    zip_file_path = file_path + ".zip"
    if os.path.exists(zip_file_path):
        # 这里可以添加解压逻辑，但通常建议预先解压好文件
        return None
    
    return None


def get_elevation_from_hgt_file(lat, lon, hgt_file):
    """
    从指定的HGT文件中获取指定经纬度位置的高程值
    
    HGT文件是一个1度×1度的区域，分辨率为1弧秒（约30米），
    文件大小为3601×3601个16位整数值，表示海拔高度（单位：米）。
    
    Args:
        lat (float): 纬度
        lon (float): 经度
        hgt_file (str): HGT文件的完整路径
        
    Returns:
        float: 高程值（单位：米），如果无法获取则返回None
    """
    if not os.path.exists(hgt_file):
        print(f"HGT文件不存在: {hgt_file}")
        return None
    
    try:
        # 使用GDAL打开HGT文件
        dataset = gdal.Open(hgt_file)
        if not dataset:
            return None
        
        # 获取地理变换参数
        geotransform = dataset.GetGeoTransform()
        
        # 计算像素坐标
        # HGT文件的原点在西北角，x方向为经度，y方向为纬度
        x = int((lon - geotransform[0]) / geotransform[1])
        y = int((lat - geotransform[3]) / geotransform[5])
        
        # 确保坐标在有效范围内
        if x < 0 or y < 0 or x >= dataset.RasterXSize or y >= dataset.RasterYSize:
            return None
        
        # 读取高程值
        band = dataset.GetRasterBand(1)
        elevation_data = band.ReadAsArray(x, y, 1, 1)
        elevation = float(elevation_data[0, 0])
        
        # 关闭数据集
        dataset = None
        
        # 如果高程值为-32768（SRTM数据中的无效值），则返回None
        if elevation == -32768:
            return None
        
        return elevation
    
    except Exception as e:
        print(f"获取高程数据时出错: {e}")
        return None


def get_elevation_from_hgt(lat, lon, hgt_dir="data/DSM"):
    """
    从HGT文件中获取指定经纬度位置的高程值
    
    HGT文件是一个1度×1度的区域，分辨率为1弧秒（约30米），
    文件大小为3601×3601个16位整数值，表示海拔高度（单位：米）。
    
    Args:
        lat (float): 纬度
        lon (float): 经度
        hgt_dir (str): HGT文件所在目录，默认为项目中的LY-dem目录
        
    Returns:
        float: 高程值（单位：米），如果无法获取则返回None
    """
    # 获取HGT文件路径
    hgt_file = get_hgt_file_path(lat, lon, hgt_dir)
    if not hgt_file:
        return None
    
    return get_elevation_from_hgt_file(lat, lon, hgt_file)


def get_elevation_bilinear_from_file(lat, lon, hgt_file):
    """
    使用双线性插值从指定的HGT文件中获取指定经纬度位置的高程值
    
    双线性插值可以提供更平滑的高程数据，特别是在两个像素之间的位置
    
    Args:
        lat (float): 纬度
        lon (float): 经度
        hgt_file (str): HGT文件的完整路径
        
    Returns:
        float: 插值后的高程值（单位：米），如果无法获取则返回None
    """
    if not os.path.exists(hgt_file):
        print(f"HGT文件不存在: {hgt_file}")
        return None

    # 使用GDAL打开HGT文件
    dataset = gdal.Open(hgt_file)
    if not dataset:
        return None

    # 获取地理变换参数
    geotransform = dataset.GetGeoTransform()

    # 计算精确的像素坐标（浮点数）
    x = (lon - geotransform[0]) / geotransform[1]
    y = (lat - geotransform[3]) / geotransform[5]

    # 获取周围四个像素的整数坐标
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1

    # 确保所有坐标在有效范围内
    if (x0 < 0 or y0 < 0 or x1 >= dataset.RasterXSize or y1 >= dataset.RasterYSize):
        return None

    # 计算插值权重
    wx = x - x0
    wy = y - y0

    # 读取四个角点的高程值
    band = dataset.GetRasterBand(1)
    data = band.ReadAsArray(x0, y0, 2, 2)

    # 检查是否有无效值（SRTM数据中的-32768）
    if np.any(data == -32768):
        # 如果有无效值，回退到最近邻插值
        x_nearest = int(round(x))
        y_nearest = int(round(y))
        nearest_data = band.ReadAsArray(x_nearest, y_nearest, 1, 1)
        elevation = float(nearest_data[0, 0])
        if elevation == -32768:
            return None
        return elevation

    # 执行双线性插值
    # f(x,y) = (1-wx)(1-wy)f(x0,y0) + wx(1-wy)f(x1,y0) + (1-wx)wy*f(x0,y1) + wx*wy*f(x1,y1)
    elevation = ((1-wx)*(1-wy)*data[0,0] +
                wx*(1-wy)*data[0,1] +
                (1-wx)*wy*data[1,0] +
                wx*wy*data[1,1])

    # 关闭数据集
    dataset = None

    return float(elevation)


def get_elevation_bilinear(lat, lon):
    """
    使用双线性插值从HGT文件中获取指定经纬度位置的高程值
    
    双线性插值可以提供更平滑的高程数据，特别是在两个像素之间的位置
    
    Args:
        lat (float): 纬度
        lon (float): 经度
        hgt_dir (str): HGT文件所在目录，默认为项目中的LY-dem目录
        
    Returns:
        float: 插值后的高程值（单位：米），如果无法获取则返回None
    """
    hgt_dir = "data/DSM"
    # 获取HGT文件路径
    hgt_file = get_hgt_file_path(lat, lon, hgt_dir)
    if not hgt_file:
        return None
    
    return get_elevation_bilinear_from_file(lat, lon, hgt_file)