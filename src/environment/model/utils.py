import collections.abc as collections
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple, Union

import cv2
import kornia
import numpy as np
import torch


class ImagePreprocessor:
    """
    图像预处理类，用于调整图像尺寸和预处理
    
    配置参数:
        resize (int): 目标边长，None表示不调整大小
        side (str): 调整基准边，"long"表示长边，"short"表示短边
        interpolation (str): 插值方法
        align_corners (bool): 对齐像素角点设置
        antialias (bool): 是否启用抗锯齿
    
    Examples:
    >>> preprocessor = ImagePreprocessor(resize=1024)
    >>> img_tensor = torch.randn(3, 512, 512)
    >>> processed_img, scale = preprocessor(img_tensor)
    """
    default_conf = {
        "resize": None,  # target edge length, None for no resizing
        "side": "long",
        "interpolation": "bilinear",
        "align_corners": None,
        "antialias": True,
    }

    def __init__(self, **conf) -> None:
        super().__init__()
        self.conf = {**self.default_conf, **conf}
        self.conf = SimpleNamespace(**self.conf)

    def __call__(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resize and preprocess an image, return image and resize scale"""
        h, w = img.shape[-2:]
        if self.conf.resize is not None:
            img = kornia.geometry.transform.resize(
                img,
                self.conf.resize,
                side=self.conf.side,
                antialias=self.conf.antialias,
                align_corners=self.conf.align_corners,
            )
        scale = torch.Tensor([img.shape[-1] / w, img.shape[-2] / h]).to(img)
        return img, scale


def map_tensor(input_, func: Callable):
    """
    递归地对张量数据结构应用处理函数
    
    Args:
        input_: 输入数据结构，支持张量/字典/列表
        func: 要应用的函数
        
    Returns:
        处理后的数据结构
        
    Examples:
    >>> data = {'img': torch.tensor([1,2,3]), 'mask': [torch.tensor(4)]}
    >>> map_tensor(data, lambda x: x*2)
    """
    string_classes = (str, bytes)
    if isinstance(input_, string_classes):
        return input_
    elif isinstance(input_, collections.Mapping):
        return {k: map_tensor(sample, func) for k, sample in input_.items()}
    elif isinstance(input_, collections.Sequence):
        return [map_tensor(sample, func) for sample in input_]
    elif isinstance(input_, torch.Tensor):
        return func(input_)
    else:
        return input_


def batch_to_device(batch: dict, device: str = "cpu", non_blocking: bool = True):
    """
    将数据批转移到指定设备
    
    Args:
        batch: 输入数据字典
        device: 目标设备
        non_blocking: 是否使用非阻塞传输
        
    Returns:
        设备上的数据字典
        
    Examples:
    >>> data = {'image': torch.randn(4,3,256,256)}
    >>> batch_to_device(data, 'cuda')
    """

    def _func(tensor):
        return tensor.to(device=device, non_blocking=non_blocking).detach()

    return map_tensor(batch, _func)


def rbd(data: dict) -> dict:
    """
    移除数据中的批次维度（第0维）
    
    Args:
        data: 包含张量/数组/列表的字典
        
    Returns:
        dict: 移除批次维度后的字典
        
    Examples:
    >>> batch_data = {'keypoints': torch.randn(1, 100, 2)}
    >>> rbd(batch_data)['keypoints'].shape  # (100, 2)
    """
    return {
        k: v[0] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v
        for k, v in data.items()
    }


def read_image(path: Path, grayscale: bool = False) -> np.ndarray:
    """
    读取图像文件为numpy数组
    
    Args:
        path: 图像文件路径
        grayscale: 是否以灰度模式读取
        
    Returns:
        np.ndarray: 图像数组（RGB或灰度）
        
    Raises:
        FileNotFoundError: 文件不存在时抛出
        IOError: 读取失败时抛出
        
    Examples:
    >>> img = read_image('test.jpg')
    >>> gray_img = read_image('test.png', grayscale=True)
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"No image at path {path}.")
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise IOError(f"Could not read image at {path}.")
    if not grayscale:
        image = image[..., ::-1]
    return image


def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """
    将numpy图像转换为标准化PyTorch张量
    
    Args:
        image: 输入numpy数组（HxWxC或HxW）
        
    Returns:
        torch.Tensor: 标准化后的张量（CxHxW），值域[0,1]
        
    Raises:
        ValueError: 输入维度不符合要求时抛出
        
    Examples:
    >>> np_img = np.random.randint(0,255,(256,256,3))
    >>> torch_img = numpy_image_to_torch(np_img)
    """
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)


def resize_image(
    image: np.ndarray,
    size: Union[List[int], int],
    fn: str = "max",
    interp: Optional[str] = "area",
) -> np.ndarray:
    """
    调整图像尺寸并返回缩放比例
    
    Args:
        image: 输入图像数组（HxWxC）
        size: 目标尺寸，整数（按比例缩放）或元组（固定尺寸）
        fn: 缩放基准边，"max"按最长边，"min"按最短边
        interp: 插值方法，可选[linear/cubic/nearest/area]
        
    Returns:
        tuple: (调整后的图像, (宽缩放比例, 高缩放比例))
        
    Examples:
    >>> img, scale = resize_image(img, 1024)  # 按最长边缩放
    >>> img, scale = resize_image(img, (512, 768), interp='linear')
    """
    h, w = image.shape[:2]

    fn = {"max": max, "min": min}[fn]
    if isinstance(size, int):
        scale = size / fn(h, w)
        h_new, w_new = int(round(h * scale)), int(round(w * scale))
        scale = (w_new / w, h_new / h)
    elif isinstance(size, (tuple, list)):
        h_new, w_new = size
        scale = (w_new / w, h_new / h)
    else:
        raise ValueError(f"Incorrect new size: {size}")
    mode = {
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "nearest": cv2.INTER_NEAREST,
        "area": cv2.INTER_AREA,
    }[interp]
    return cv2.resize(image, (w_new, h_new), interpolation=mode), scale


def load_image(path: Path, resize: int = None, **kwargs) -> torch.Tensor:
    """
    加载图像文件并转换为张量
    
    Args:
        path: 图像文件路径
        resize: 可选调整尺寸参数
        **kwargs: 传递给resize_image的参数
        
    Returns:
        torch.Tensor: 预处理后的图像张量
        
    Examples:
    >>> tensor = load_image('test.jpg')
    >>> tensor = load_image('test.png', resize=1024, fn='min')
    """
    image = read_image(path)
    if resize is not None:
        image, _ = resize_image(image, resize, **kwargs)
    return numpy_image_to_torch(image)


class Extractor(torch.nn.Module):
    """
    特征提取器基类
    
    配置参数:
        preprocess_conf: 图像预处理配置
        default_conf: 默认配置参数
        
    Methods:
        extract: 执行特征提取
        forward: 前向传播方法（需子类实现）
        
    Examples:
    >>> extractor = Extractor(preprocess_conf={'resize': 1024})
    >>> features = extractor.extract(torch.randn(1,3,512,512))
    """
    def __init__(self, **conf):
        super().__init__()
        self.conf = SimpleNamespace(**{**self.default_conf, **conf})

    @torch.no_grad()
    def extract(self, img: torch.Tensor, **conf) -> dict:
        """Perform extraction with online resizing"""
        if img.dim() == 3:
            img = img[None]  # add batch dim
        assert img.dim() == 4 and img.shape[0] == 1
        shape = img.shape[-2:][::-1]
        img, scales = ImagePreprocessor(**{**self.preprocess_conf, **conf})(img)
        feats = self.forward({"image": img})
        feats["image_size"] = torch.tensor(shape)[None].to(img).float()
        feats["keypoints"] = (feats["keypoints"] + 0.5) / scales[None] - 0.5
        return feats


def match_pair(
    extractor,
    matcher,
    image0: torch.Tensor,
    image1: torch.Tensor,
    device: str = "cpu",
    **preprocess,
):
    """Match a pair of images (image0, image1) with an extractor and matcher"""
    feats0 = extractor.extract(image0, **preprocess)
    feats1 = extractor.extract(image1, **preprocess)
    matches01 = matcher({"image0": feats0, "image1": feats1})
    data = [feats0, feats1, matches01]
    # remove batch dim and move to target device
    feats0, feats1, matches01 = [batch_to_device(rbd(x), device) for x in data]
    return feats0, feats1, matches01
