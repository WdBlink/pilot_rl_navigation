from osgeo import gdal, osr

def get_center_geographic_coordinates(file_path):
    """
    获取TIFF文件中心点的地理坐标（WGS84经纬度）
    
    Args:
        file_path (str): TIFF文件路径
        
    Returns:
        tuple[float, float]: 包含经度和纬度的元组 (longitude, latitude)
        
    Raises:
        Exception: 当无法打开TIFF文件时抛出
        
    Examples:
    >>> lon, lat = get_center_geographic_coordinates('test.tif')
    >>> print(f"经度: {lon}, 纬度: {lat}")
    经度: 120.123456, 纬度: 30.654321
    """
    # 打开tif文件
    dataset = gdal.Open(file_path)
    if dataset is None:
        raise Exception(f"Could not open {file_path}")

    # 获取tif文件的地理变换信息
    geotransform = dataset.GetGeoTransform()

    # 计算tif文件的尺寸
    width = dataset.RasterXSize
    height = dataset.RasterYSize

    # 计算中心点的像素坐标
    center_pixel_x = width / 2.0
    center_pixel_y = height / 2.0

    # 将像素坐标转换为地理坐标（经纬度）
    center_geo_x = geotransform[0] + (center_pixel_x * geotransform[1]) + (center_pixel_y * geotransform[2])
    center_geo_y = geotransform[3] + (center_pixel_x * geotransform[4]) + (center_pixel_y * geotransform[5])

    # 创建空间参考对象
    srs = osr.SpatialReference()
    srs.ImportFromWkt(dataset.GetProjectionRef())

    # 如果需要将坐标转换为WGS84（经纬度），可以创建一个坐标转换对象
    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(4326)
    transform = osr.CoordinateTransformation(srs, target_srs)

    # 进行坐标转换
    center_geo_x, center_geo_y, _ = transform.TransformPoint(center_geo_x, center_geo_y)

    return center_geo_x, center_geo_y

