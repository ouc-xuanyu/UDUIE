import os
import cv2
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

def calculate_psnr(image_path1, image_path2, target_size=(256, 256)):
    """计算两张图片的PSNR，先将图像缩放至目标尺寸"""
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    
    # 缩放图像到256x256
    img1_resized = cv2.resize(img1, target_size)
    img2_resized = cv2.resize(img2, target_size)
    
    psnr = compare_psnr(img1_resized, img2_resized)
    return psnr


def find_corresponding_image(file_base_name, folder_b):
    """在文件夹B中找到与基本文件名匹配的图像"""
    for filename in os.listdir(folder_b):
        if file_base_name == os.path.splitext(filename)[0]:
            return os.path.join(folder_b, filename)
    return None

def main(folder_a, folder_b, output_file):
    """遍历文件夹A中的图像，与文件夹B中对应的图像计算PSNR，并保存结果到文件"""
    with open(output_file, 'w') as f:
        for filename in os.listdir(folder_a):
            if filename.endswith((".jpg", ".png")):  # 确保只处理图像文件
                path_a = os.path.join(folder_a, filename)
                file_base_name = os.path.splitext(filename)[0]
                path_b = find_corresponding_image(file_base_name, folder_b)

                if path_b is not None and os.path.exists(path_b):  # 找到对应文件并确保其存在
                    psnr = calculate_psnr(path_a, path_b)
                    f.write(f"{filename}, {psnr}\n")
                    print(f"File: {filename}, PSNR: {psnr}")
                else:
                    print(f"No corresponding file found for {filename} in folder B.")

if __name__ == "__main__":
    #folder_a = '/data/luohan/query-selected-attention-main/results/lambda10+ours/val_60/images/fake_B'
    folder_a = '/data/luohan/PDDNet-main/PDDNet/results/PDDNet-Re/test_UIEB/test_results'
    folder_b = '/data/luohan/query-selected-attention-main/dataset_426/valB'
    output_file = "pdd_psnr_results.txt"
    main(folder_a, folder_b, output_file)