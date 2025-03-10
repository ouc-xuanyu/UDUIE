import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import pyiqa
import torch
mymodel=pyiqa.create_metric('uranker',device=torch.device("cuda:2"))
def compute_from_paths(image_path):
    return mymodel(image_path)

def get_images_in_folder(folder_path):
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

def fun(folder_path):
    print(folder_path)
    images = get_images_in_folder(folder_path)
    scores=[]
    s=[]
    for image in images:
        a=compute_from_paths(image)
        scores.append(a)
    print(torch.stack(scores).mean(dim=0))

if __name__ == '__main__':
    fun('/root/autodl-tmp/uduie/ssm/c60/ttk0/val_30/images/fake_B')
    fun('/root/autodl-tmp/uduie/ssm/u45/ttk0/val_30/images/fake_B')
    fun('/root/autodl-tmp/uduie/ssm/uccs/ttk0/val_30/images/fake_B')
    fun('/root/autodl-tmp/uduie/ssm/ours/ttk0/val_30/images/fake_B')