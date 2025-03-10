import multiprocessing
from functools import partial

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
def process_image(full_path, output_rootdir, f):
    try:
        img1 = cv2.imread(full_path, cv2.IMREAD_COLOR)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        score = getUIQM(img1)  # 假设此函数已实现
        score1 = uciqe(full_path)
        f.write(full_path + '    ' + str(score) + '    ' + str(score1) + '\n')
    except Exception as e:
        print(f"处理图片---- {full_path} 时出错: {e}")

def process_images_multiprocessing(rootdir, output_rootdir, f):
    if not os.path.isdir(rootdir) or not os.path.isdir(output_rootdir):
        print(f"无效的路径: {rootdir} 或 {output_rootdir}")
        return

    filelist = os.listdir(rootdir)
    
    with multiprocessing.Pool() as pool:
        # partial函数用于创建一个带固定参数的新函数
        process_image_partial = partial(process_image, output_rootdir=output_rootdir, f=f)
        pool.map(process_image_partial, [os.path.join(rootdir, img) for img in filelist])

# ...（保持原始代码的主程序部分不变）

if __name__ == "__main__":
    rootdir = '/data/luohan/query-selected-attention-main/dataset_426/trainA'
    output_rootdir = rootdir
    with open('/data/luohan/query-selected-attention-main/nonono.txt', 'a') as f:
        process_images_multiprocessing(rootdir, output_rootdir, f)
