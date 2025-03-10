#!/usr/bin/env python
"""
# > Modules for computing the Underwater Image Quality Measure (UIQM)
#
# Maintainer: Jahid (email: islam034@umn.edu)
# Interactive Robotics and Vision Lab (http://irvlab.cs.umn.edu/)
# Any part of this repo can be used for academic and educational purposes only
"""
from scipy import ndimage
from PIL import Image
import numpy as np
import math
import niqe


def uciqe(loc):
    img_bgr = cv2.imread(loc)  # Used to read image files
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)  # Transform to Lab color space

    coe_metric = [0.4680, 0.2745, 0.2576]  # Obtained coefficients are: c1=0.4680, c2=0.2745, c3=0.2576.
    img_lum = img_lab[..., 0] / 255
    img_a = img_lab[..., 1] / 255
    img_b = img_lab[..., 2] / 255

    img_chr = np.sqrt(np.square(img_a) + np.square(img_b))  # Chroma

    img_sat = img_chr / np.sqrt(np.square(img_chr) + np.square(img_lum))  # Saturation
    aver_sat = np.mean(img_sat)  # Average of saturation

    aver_chr = np.mean(img_chr)  # Average of Chroma

    var_chr = np.sqrt(np.mean(abs(1 - np.square(aver_chr / img_chr))))  # Variance of Chroma

    dtype = img_lum.dtype  # Determine the type of img_lum
    if dtype == 'uint8':
        nbins = 256
    else:
        nbins = 65536

    hist, bins = np.histogram(img_lum, nbins)  # Contrast of luminance
    cdf = np.cumsum(hist) / np.sum(hist)

    ilow = np.where(cdf > 0.0100)
    ihigh = np.where(cdf >= 0.9900)
    tol = [(ilow[0][0] - 1) / (nbins - 1), (ihigh[0][0] - 1) / (nbins - 1)]
    con_lum = tol[1] - tol[0]

    quality_val = coe_metric[0] * var_chr + coe_metric[1] * con_lum + coe_metric[
        2] * aver_sat  # get final quality value
    # print("quality_val is", quality_val)
    return quality_val

from PIL import Image
# x=Image.open(r'D:\download\ziqiang\realistic_411\9FKA5IFOE2_03533.jpg')
import cv2
import glob
import os
from tqdm import tqdm, trange
import os
import cv2
import tqdm
from pathlib import Path
from uiqm_batch import getUIQM

def write_image_to_directory(image_path, score, root_directory):
    """
    根据分数将图片写入不同的子目录。
    """
    directory_name = f"{score:.1f}"
    target_directory = os.path.join(root_directory, directory_name)
    os.makedirs(target_directory, exist_ok=True)
    filename = os.path.basename(image_path)
    target_path = os.path.join(target_directory, filename)
    cv2.imwrite(target_path, cv2.imread(image_path))

f=open('/data/luohan/query-selected-attention-main/nonono.txt', 'a')
def process_images(rootdir, output_rootdir):
    """
    处理图片，根据UIQM分数将图片分类到不同的目录。
    """
    if not os.path.isdir(rootdir) or not os.path.isdir(output_rootdir):
        print(f"无效的路径: {rootdir} 或 {output_rootdir}")
        return

    filelist = os.listdir(rootdir)
    for img in tqdm.tqdm(filelist):
        try:
            full_path = os.path.join(rootdir, img)
            if not os.path.isfile(full_path):
                continue  # 跳过非文件条目
            img1 = cv2.imread(full_path, cv2.IMREAD_COLOR)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            score = getUIQM(img1)  # 假设此函数已实现
            score1 = uciqe(full_path)
            #print(score)
            # if score > 2.9:
            f.write(full_path+'    '+str(score)+'    '+str(score1)+'\n')
            # 根据分数写入不同的目录
            #write_image_to_directory(full_path, score, output_rootdir)

        except Exception as e:
            print(f"处理图片---- {img} 时出错: {e}")


if __name__ == "__main__":
    rootdir = '/data/luohan/query-selected-attention-main/dataset_426/trainA'  # 可以改为接收命令行参数或从配置文件读取
    output_rootdir = rootdir  # 同上
    process_images(rootdir, output_rootdir)
    # rootdir = '/data/luohan/query-selected-attention-main/mydataset/0.5'  # 可以改为接收命令行参数或从配置文件读取
    # output_rootdir = rootdir  # 同上
    # process_images(rootdir, output_rootdir)    
