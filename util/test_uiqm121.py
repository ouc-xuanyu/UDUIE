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
def mu_a(x, alpha_L=0.1, alpha_R=0.1):
    """
      Calculates the asymetric alpha-trimmed mean
    """
    # sort pixels by intensity - for clipping
    x = sorted(x)
    # get number of pixels
    K = len(x)
    # calculate T alpha L and T alpha R
    T_a_L = math.ceil(alpha_L*K)
    T_a_R = math.floor(alpha_R*K)
    # calculate mu_alpha weight
    weight = (1/(K-T_a_L-T_a_R))
    # loop through flattened image starting at T_a_L+1 and ending at K-T_a_R
    s   = int(T_a_L+1)
    e   = int(K-T_a_R)
    val = sum(x[s:e])
    val = weight*val
    return val

def s_a(x, mu):
    val = 0
    for pixel in x:
        val += math.pow((pixel-mu), 2)
    return val/len(x)

def _uicm(x):
    R = x[:,:,0].flatten()
    G = x[:,:,1].flatten()
    B = x[:,:,2].flatten()
    RG = R-G
    YB = ((R+G)/2)-B
    mu_a_RG = mu_a(RG)
    mu_a_YB = mu_a(YB)
    s_a_RG = s_a(RG, mu_a_RG)
    s_a_YB = s_a(YB, mu_a_YB)
    l = math.sqrt( (math.pow(mu_a_RG,2)+math.pow(mu_a_YB,2)) )
    r = math.sqrt(s_a_RG+s_a_YB)
    return (-0.0268*l)+(0.1586*r)

def sobel(x):
    dx = ndimage.sobel(x,0)
    dy = ndimage.sobel(x,1)
    mag = np.hypot(dx, dy)
    mag *= 255.0 / np.max(mag)
    return mag

def eme(x, window_size):
    """
      Enhancement measure estimation
      x.shape[0] = height
      x.shape[1] = width
    """
    # if 4 blocks, then 2x2...etc.
    k1 = int(x.shape[1]/window_size)
    k2 = int(x.shape[0]/window_size)
    # weight
    w = 2./(k1*k2)
    blocksize_x = window_size
    blocksize_y = window_size
    # make sure image is divisible by window_size - doesn't matter if we cut out some pixels
    wheight = int(blocksize_y*k2)
    wwith = int(blocksize_x*k1)
    x = x[:wheight, :wwith]
    val = 0
    for l in range(k1):
        for k in range(k2):
            block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1)]
            max_ = np.max(block)
            min_ = np.min(block)
            # bound checks, can't do log(0)
            if min_ == 0.0: val += 0
            elif max_ == 0.0: val += 0
            else: val += math.log(max_/min_)
    return w*val

def _uism(x):
    """
      Underwater Image Sharpness Measure
    """
    # get image channels
    R = x[:,:,0]
    G = x[:,:,1]
    B = x[:,:,2]
    # first apply Sobel edge detector to each RGB component
    Rs = sobel(R)
    Gs = sobel(G)
    Bs = sobel(B)
    # multiply the edges detected for each channel by the channel itself
    R_edge_map = np.multiply(Rs, R)
    G_edge_map = np.multiply(Gs, G)
    B_edge_map = np.multiply(Bs, B)
    # get eme for each channel
    r_eme = eme(R_edge_map, 10)
    g_eme = eme(G_edge_map, 10)
    b_eme = eme(B_edge_map, 10)
    # coefficients
    lambda_r = 0.299
    lambda_g = 0.587
    lambda_b = 0.144
    return (lambda_r*r_eme) + (lambda_g*g_eme) + (lambda_b*b_eme)

def plip_g(x,mu=1026.0):
    return mu-x

def plip_theta(g1, g2, k):
    g1 = plip_g(g1)
    g2 = plip_g(g2)
    return k*((g1-g2)/(k-g2))

def plip_cross(g1, g2, gamma):
    g1 = plip_g(g1)
    g2 = plip_g(g2)
    return g1+g2-((g1*g2)/(gamma))

def plip_diag(c, g, gamma):
    g = plip_g(g)
    return gamma - (gamma * math.pow((1 - (g/gamma) ), c) )

def plip_multiplication(g1, g2):
    return plip_phiInverse(plip_phi(g1) * plip_phi(g2))
    #return plip_phiInverse(plip_phi(plip_g(g1)) * plip_phi(plip_g(g2)))

def plip_phiInverse(g):
    plip_lambda = 1026.0
    plip_beta   = 1.0
    return plip_lambda * (1 - math.pow(math.exp(-g / plip_lambda), 1 / plip_beta));

def plip_phi(g):
    plip_lambda = 1026.0
    plip_beta   = 1.0
    return -plip_lambda * math.pow(math.log(1 - g / plip_lambda), plip_beta)

def _uiconm(x, window_size):
    """
      Underwater image contrast measure
      https://github.com/tkrahn108/UIQM/blob/master/src/uiconm.cpp
      https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5609219
    """
    plip_lambda = 1026.0
    plip_gamma  = 1026.0
    plip_beta   = 1.0
    plip_mu     = 1026.0
    plip_k      = 1026.0
    # if 4 blocks, then 2x2...etc.
    k1 = int(x.shape[1]/window_size)
    k2 = int(x.shape[0]/window_size)
    # weight
    w = -1./(k1*k2)
    blocksize_x = window_size
    blocksize_y = window_size
    # make sure image is divisible by window_size - doesn't matter if we cut out some pixels
    x = x[:blocksize_y*k2, :blocksize_x*k1]
    # entropy scale - higher helps with randomness
    alpha = 1
    val = 0
    for l in range(k1):
        for k in range(k2):
            block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1), :]
            max_ = np.max(block)
            min_ = np.min(block)
            top = max_-min_
            bot = max_+min_
            if math.isnan(top) or math.isnan(bot) or bot == 0.0 or top == 0.0: val += 0.0
            else: val += alpha*math.pow((top/bot),alpha) * math.log(top/bot)
            #try: val += plip_multiplication((top/bot),math.log(top/bot))
    return w*val

def getUIQM(x):
    """
      Function to return UIQM to be called from other programs
      x: image
    """
    x = x.astype(np.float32)
    ### from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7300447
    #c1 = 0.4680; c2 = 0.2745; c3 = 0.2576
    ### from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7300447
    c1 = 0.0282; c2 = 0.2953; c3 = 3.5753
    uicm   = _uicm(x)
    uism   = _uism(x)
    uiconm = _uiconm(x, 10)
    uiqm = (c1*uicm) + (c2*uism) + (c3*uiconm)
    return uiqm

def getUCIQE(img_rgb):
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)  # Transform to Lab color space

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

import multiprocessing as mp
from niqe import niqe
from piqe import piqe

def compute_single_image_uiqm(idx, img):
    u=getUIQM(img)
    return u
def compute_single_image_uciqe(idx, img):
    u=getUCIQE(img)
    return u
def compute_single_image_en(idx, img):
    u=shannon_entropy(img,base=np.e)
    return u

def compute_single_image_piqe(idx, img):
    u,_,_,_ = piqe(img)
    return u
def parallel_compute_uiqm(batch_images):
    pool = mp.Pool(10)
    tasks = [(i, img) for i, img in enumerate(batch_images)]
    results_uiqm = pool.starmap(compute_single_image_uiqm, tasks)
    results_uciqe = pool.starmap(compute_single_image_uciqe, tasks)
    results_en = pool.starmap(compute_single_image_en, tasks)
    return np.mean(results_uiqm),np.mean(results_uciqe),np.mean(results_en)
def read_images_to_batch(image_paths, target_shape=None):
    images = []
    for path in os.listdir(image_paths):
        path0=image_paths+'//'+path
        img = cv2.imread(path0, cv2.IMREAD_COLOR)
        if target_shape is not None:
            img = cv2.resize(img, target_shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    batch_images = np.stack(images, axis=0)
    return batch_images


# 示例用法
import time

def fun(image_paths):
    print(image_paths, end=' ')
    batch_images = read_images_to_batch(image_paths,target_shape=(256,256))
    xx = parallel_compute_uiqm(batch_images)
    print(xx)
if __name__ == '__main__':
    fun('/root/autodl-tmp/uduie/ssm/c60/ttk0/val_30/images/fake_B')
    fun('/root/autodl-tmp/uduie/ssm/u45/ttk0/val_30/images/fake_B')
    fun('/root/autodl-tmp/uduie/ssm/uccs/ttk0/val_30/images/fake_B')
    fun('/root/autodl-tmp/uduie/ssm/ours/ttk0/val_30/images/fake_B')

    #/root/autodl-tmp/uduie/ssm/ufo/ttk0/val_49/images/fake_B (1108.142503950331, 18.638020647819083)
    #/root/autodl-tmp/uduie/ssm/uwi/ttk0/val_49/images/fake_B (925.7740154278897, 19.2282263279611)
    #/root/autodl-tmp/uduie/ssm/c60/ttk0/val_49/images/fake_B (3.0397081076172574, 0.584583910731902, 5.066859947081135)       1.23
    #/root/autodl-tmp/uduie/ssm/ours/ttk0/val_49/images/fake_B (3.3176477361008607, 0.6027731960113292, 5.223270533753695)     1.87
    #/root/autodl-tmp/uduie/ssm/u45/ttk0/val_49/images/fake_B (3.3291373779548654, 0.6024589544115527, 5.249818218305768)      1.65
    #/root/autodl-tmp/uduie/ssm/uccs/ttk0/val_49/images/fake_B (3.3886296601035246, 0.5904492319822698, 5.303478990807412)     1.91


############################################################################################################################################################

    #/root/autodl-tmp/uduie/ssm_ttk0/ufo/ttk0/val_61/images/fake_B (1140.0872552659778, 18.681544596628903)
    #/root/autodl-tmp/uduie/ssm_ttk0/uwi/ttk0/val_61/images/fake_B (912.4186385750588, 19.223408327798424)
    #/root/autodl-tmp/uduie/ssm_ttk0/c60/ttk0/val_61/images/fake_B (3.1610322641728463, 0.6017209843964775, 5.110705096232445)  1.44
    #/root/autodl-tmp/uduie/ssm_ttk0/ours/ttk0/val_61/images/fake_B (3.3783219635987063, 0.6207245207448898, 5.2587611511500905) 2.04
    #/root/autodl-tmp/uduie/ssm_ttk0/u45/ttk0/val_61/images/fake_B (3.3977275758786556, 0.6219625876665584, 5.274829548071684)   1.84
    #/root/autodl-tmp/uduie/ssm_ttk0/uccs/ttk0/val_61/images/fake_B (3.4453170565994893, 0.6089563167544767, 5.348235445068781)  2.23

    #root/autodl-tmp/uduie/ssm/ufo/ttk0/val_74/images/fake_B (1113.9712195502389, 18.727254868205044)
    #/root/autodl-tmp/uduie/ssm/ufo/ttk0/val_77/images/fake_B (1136.1046902126736, 18.738282365424975)





    #/root/autodl-tmp/uduie/ssm/uwi/ttk0/val_30/images/fake_B (888.2521905441334, 19.480539916590644)
    #u45 2.03



    #/root/autodl-tmp/uduie/ssm_ttk1/ufo/ttk1/val_82/images/fake_B (1061.0520434485543, 19.06295964269409)
    #/root/autodl-tmp/uduie/ssm_ttk1/uwi/ttk1/val_82/images/fake_B (850.1559493520296, 19.64332041647739)

    #/root/autodl-tmp/uduie/ssm_ttk1/ufo/ttk1/val_44/images/fake_B (1032.855034467909, 18.954799503645184)
    #/root/autodl-tmp/uduie/ssm_ttk1/uwi/ttk1/val_44/images/fake_B (926.10404979297, 19.312225261271944)
    #/root/autodl-tmp/uduie/ssm_ttk1/c60/ttk1/val_44/images/fake_B (3.089089840737267, 0.57781901453579, 5.084258100308421)     1.43
    #/root/autodl-tmp/uduie/ssm_ttk1/ours/ttk1/val_44/images/fake_B (3.2830835469011674, 0.5902475966541354, 5.22542049055655)  1.97
    #/root/autodl-tmp/uduie/ssm_ttk1/u45/ttk1/val_44/images/fake_B (3.31143830017918, 0.5889709393585124, 5.240065107599779)     1.70
    #/root/autodl-tmp/uduie/ssm_ttk1/uccs/ttk1/val_44/images/fake_B (3.284482685057949, 0.5945214126012107, 5.313256263996091)   1.97


    #/root/autodl-tmp/uduie/ssm_ttk2/ufo/ttk2/val_45/images/fake_B (1185.0536880493164, 18.146462499960407)
    #root/autodl-tmp/uduie/ssm_ttk2/c60/ttk2/val_45/images/fake_B (3.0103511295833183, 0.5582663444538112, 4.989895785940507) 1.28
    #/root/autodl-tmp/uduie/ssm_ttk2/ours/ttk2/val_45/images/fake_B (3.2884821406268214, 0.5817649984000072, 5.146259811732239)2.01
    #/root/autodl-tmp/uduie/ssm2/u45/ttk2/val_45/images/fake_B (3.319871219593107, 0.583281213847434, 5.187857031820542)     1.94
    #/root/autodl-tmp/uduie/ssm2/uwi/ttk2/val_45/images/fake_B (1412.1228067438617, 17.738085730640357)




