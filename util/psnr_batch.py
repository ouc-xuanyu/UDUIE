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
from skimage.measure import shannon_entropy
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from PIL import Image
# x=Image.open(r'D:\download\ziqiang\realistic_411\9FKA5IFOE2_03533.jpg')
import cv2
import glob
import os
from tqdm import tqdm, trange
import torch

import multiprocessing as mp

def compute_single_image_ssim(img, img1):
    return compare_ssim(img1, img,channel_axis=2,win_size=11)

def compute_single_image_mse(img, img1):
    return compare_mse(img1, img)

def compute_single_image_psnr(img, img1):
    return compare_psnr(img1, img)
def parallel_compute(batch_images, batch_images1):
    # 创建进程池
    pool = mp.Pool(mp.cpu_count())#

    # 将图像批次拆分为多个任务
    tasks = [(img1, img2) for img1, img2 in zip(batch_images, batch_images1)]
    # 并行计算
    #results_ssim = pool.starmap(compute_single_image_ssim, tasks)
    results_psnr = pool.starmap(compute_single_image_psnr, tasks)
    results_mse = pool.starmap(compute_single_image_mse, tasks)
    return np.mean(results_mse), np.mean(results_psnr)
 
def read_images_to_batch(image_paths, image_ref, target_shape=None):
    t=0
    images = []
    images1 = []
    for path in os.listdir(image_paths):
        path0=image_paths+'//'+path
        path1 = image_ref+'//'+ path[:-3]+'jpg'
        if os.path.exists(path1) and os.path.exists(path0):
            img = cv2.imread(path0, cv2.IMREAD_COLOR)  # 读取彩色图像
            img1 = cv2.imread(path1, cv2.IMREAD_COLOR)
            # 如果需要统一尺寸，可以在此处进行缩放
            if target_shape is not None:
                img = cv2.resize(img, target_shape)
                img1 = cv2.resize(img1, target_shape)
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            images.append(img)
            images1.append(img1)

    # 堆叠图像，形成BHWC格式
    batch_images = np.stack(images, axis=0)
    batch_images1 = np.stack(images1, axis=0)
    #print('sum:',t)
    return batch_images, batch_images1


# 示例用法
import time
def fun(i):
    image_paths = '/data/luohan/query-selected-attention-main/saud/50*l1+426+new_aug+100domain/val_'+str(i)+'/images/fake_B'
    image_paths = '/data/luohan/CLUIE-Net-CLUIE-Net/output_saud_256'
    image_paths = '/data/luohan/PDDNet-main/PDDNet/results/PDDNet-Re/test_UIEB/test_results'
    image_paths = '/data/luohan/PUIE-Net-main/results/saud_256/mc'
    image_paths = '/data/luohan/U_shape/test/saud_256'
    image_paths = '/data/luohan/USUIR-main/results/saud_256/J'

    #image_paths = '/data/luohan/FUnIE-GAN-master/PyTorch/data/output_uwi'
    #image_paths = '/data/luohan/query-selected-attention-main/results/lambda10+ours/val_62/images/fake_B'
    #image_paths = '/data/luohan/CLAHE/uwi'
    image_paths = '/data/luohan/CLUIE-Net-CLUIE-Net/output_uwi'
    image_paths = '/data/luohan/PUIE-Net-main/results/UIEBD/mc'
    image_paths = '/data/luohan/U_shape/test/output'
    image_paths = '/data/luohan/USUIR-main/results/uwi/J'
    image_paths = '/data/luohan/CLUIE-Net-CLUIE-Net/output_saud_256'
    image_paths = '/data/luohan/U_shape/test/output_saud_256'
    image_paths= '/data/luohan/PDDNet-main/PDDNet/results/PDDNet-Re/test_UIEB/test_results_256'
    image_paths = '/data/luohan/Shallow-UWnet-main/data/output'
    image_paths = '/data/luohan/CLAHE/uwi'
    image_paths= '/data/luohan/Single-Underwater-Image-Enhancement-and-Color-Restoration-master/IE/GC/uwi_OutputImages'
    image_paths = '/data/luohan/Single-Underwater-Image-Enhancement-and-Color-Restoration-master/CR/UDCP/uwi_OutputImages'
    image_paths = '/data/luohan/Water_Net-code_pytorch-main/uwi1'
    image_paths = '/data/luohan/PUIE-Net-main/results/ufo/mc'
    image_paths = '/data/luohan/PUIE-Net-main/results/ufo_mp/mp'
    image_paths = '/data/luohan/query-selected-attention-main/ufo/50*l1+426+new_aug+100domain/val_46/images/fake_B'
    image_paths = '/data/luohan/USUIR-main/results/ufo/J'
    image_paths = '/data/luohan/CLUIE-Net-CLUIE-Net/output_ufo'
    image_paths = '/data/luohan/U_shape/test/ufo'
    image_paths = '/data/luohan/Water_Net-code_pytorch-main/ufo1'
    image_paths = '/data/luohan/CLAHE/ufo'
    image_paths = '/data/luohan/query-selected-attention-main/uveb/50*l1+426+new_aug+100domain/val_46/images/fake_B'
    image_paths = '/data/luohan/USUIR-main/results/uveb/J'
    image_paths = '/data/luohan/CLUIE-Net-CLUIE-Net/output_uveb'
    image_paths = '/data/luohan/U_shape/test/uveb'
    image_paths = '/data/luohan/query-selected-attention-main/xiaorong_novgg/ufo_256/50*l1+426+new_aug+100domain+wovgg/val_'+str(i)+'/images/fake_B'
    image_paths = '/data/luohan/old_method/CR/UDCP/OutputImages/valA'
    image_paths = '/data/luohan/query-selected-attention-main/xiaorong_nodomain/uwi_256/50*l1+426+new_aug+0domain/val_53/images/fake_B'
    image_paths = '/data/luohan/query-selected-attention-main/xiaorong_novgg_nodomain/uwi_256/50*l1+426+new_aug+0domain+wovgg/val_39/images/fake_B'
    image_paths = '/data/luohan/USUIR-main/add/uwi/'+str(i)+'/J'
    image_paths = '/data/luohan/CLUIE-Net-CLUIE-Net/output_ufo'
    image_paths = '/data/luohan/query-selected-attention-main/qs/uwi/muddy2clean_ori_newdataset/val_'+str(i)+'/images/fake_B'
    image_paths = '/data/luohan/MUNIT-master/out_summer2winter/ufo'
    
    image_paths = '/root/autodl-tmp/uduie/ssm/ufo/ttk0/val_'+str(i)+'/images/fake_B'
    
    image_ref='/root/autodl-tmp/uduie/dataset_ufo/valB'

    #image_paths = '/root/autodl-tmp/uduie/ssm/uwi/ttk0/val_'+str(i)+'/images/fake_B'
    #image_paths = '/root/autodl-tmp/uduie/ssm2/uwi/ttk2/val_45/images/fake_B'
    #image_ref='/root/autodl-tmp/uduie/dataset_uwi/valB'
    print(image_paths, end=' ')
    batch_images, batch_images1 = read_images_to_batch(image_paths,image_ref)
    # time1=time.time()
    # print(time1-time0)
    xx = parallel_compute(batch_images, batch_images1)
    print(xx)
    # if not 'muddy2clean' in image_paths:
    #     exit(0)
if __name__ == '__main__':
    for i in range(21,59,1):
        i=30
        fun(i)
        exit(0)
    # epochs=[]
    # pool = mp.Pool()
    # for i in range(100, 220, 5):#muddy2clean 220 muddy2cleanori 275
    #     epochs.append(i)
    # result_parallel = pool.map(fun, epochs)
    # pool.close()
    # pool.join()
    #/data/luohan/MUNIT-master/out/uwi (5192, 11.70)
    #/data/luohan/MUNIT-master/out/ufo (5373, 11.23)
    #/data/luohan/MUNIT-master/out/c60 (3.32, 0.59, 5.21, 1.94)
    #/data/luohan/MUNIT-master/out/u45 (3.32, 0.60, 5.22, 2.01)
    #/data/luohan/MUNIT-master/out/uccs (3.12, 0.59, 5.26, 2.11)
    #/data/luohan/MUNIT-master/out/ours (3.15, 0.59, 5.26, 2.12)