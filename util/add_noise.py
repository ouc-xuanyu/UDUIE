import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
def add_separate_gaussian_noise(image):
    """
    向图像的每个通道分别添加不同的高斯噪声。
    
    参数:
    image -- 输入的图像，应为一个3D数组（彩色图像）
    means -- 一个长度为3的列表或数组，分别对应R、G、B通道的噪声均值，默认为[0, 0, 0]
    vars -- 一个长度为3的列表或数组，分别对应R、G、B通道的噪声方差，默认为[10, 10, 10]
    
    返回:
    noisy_image -- 添加了高斯噪声的图像
    """
    means = [0.21903785, 0.48115808, 0.4569597]
    vars = [0.1807035, 0.2047696, 0.21352784]
    
    # 确保噪声参数是正确的长度
    assert len(means) == 3 and len(vars) == 3, "means and vars must be lists or arrays of length 3."
    
    # 分离通道并分别添加噪声
    b, g, r = cv2.split(image)
    #print(np.mean(r),np.mean(g),np.mean(b))
    rr,gg,bb=np.random.normal(means[0], vars[0]),np.random.normal(means[1], vars[1]),np.random.normal(means[2], vars[2])
    noisy_r = r/255. + rr * random.randint(-1,1)
    noisy_g = g/255. + gg * random.randint(-1,1)
    noisy_b = b/255. + bb * random.randint(-1,1)

    # noisy_r =  np.ones(r.shape)*rr
    # noisy_g = np.ones(g.shape)*gg
    # noisy_b = np.ones(b.shape)*bb
    noisy_r = 255. / (1 + np.exp(-noisy_r))
    noisy_g = 255. / (1 + np.exp(-noisy_g))
    noisy_b = 255. / (1 + np.exp(-noisy_b))
    # noisy_r = np.clip(r + 255*np.random.normal(means[0], vars[0]**0.5, r.shape), 0, 255).astype(np.uint8)
    # noisy_g = np.clip(g + 255*np.random.normal(means[1], vars[1]**0.5, g.shape), 0, 255).astype(np.uint8)
    # noisy_b = np.clip(b + 255*np.random.normal(means[2], vars[2]**0.5, b.shape), 0, 255).astype(np.uint8)
    
    # 合并噪声后的通道
    noisy_image = cv2.merge((noisy_b, noisy_g, noisy_r))
    noisy_image=cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)
    return noisy_image

def add_gaussian_noise(tensor, mean=0, std=1):
    noise = torch.randn(tensor.size()) * std + mean
    noisy_tensor = tensor + noise.to(tensor.device)
    return noisy_tensor