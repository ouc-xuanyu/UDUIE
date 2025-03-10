import os
import shutil

def copy_files_across_folders(folder_a, folder_b, folder_c, folder_d, folder_e):
    """遍历folder_A和folder_D，根据folder_B和folder_C进行文件操作"""
    # 遍历folder_A
    # for filename in os.listdir(folder_a):
    #     path_a = os.path.join(folder_a, filename)
        
    #     # 检查文件是否存在，同时确保是文件而不是目录
    #     if os.path.isfile(path_a):
    #         path_b = os.path.join(folder_b, filename)
    #         if not os.path.exists(path_b):
    #             # 文件在B中不存在，复制到C
    #             path_c = os.path.join(folder_c, filename)
    #             shutil.copy2(path_a, path_c)  # 使用copy2保留原文件的元数据
    #             print(f"File {filename} copied from A to C because it's missing in B")

    # 遍历folder_D
    for filename in os.listdir(folder_d):
        path_d = os.path.join(folder_d, filename)
        
        # 检查文件是否存在，同时确保是文件而不是目录
        if os.path.isfile(path_d):
            path_c = os.path.join(folder_c, filename)
            if os.path.exists(path_c):
                # 文件在C中存在，复制到E
                path_e = os.path.join(folder_e, filename)
                shutil.copy2(path_d, path_e)  # 使用copy2保留原文件的元数据
                print(f"File {filename} copied from D to E because it exists in C")

if __name__ == "__main__":
    folder_a = "/data/luohan/underwater_imagenet/trainB"
    folder_b = "/data/luohan/trainB"
    folder_c = "/data/luohan/query-selected-attention-main/dataset_426/valBB"
    folder_d = "/data/luohan/underwater_imagenet/trainA"
    folder_e = "/data/luohan/query-selected-attention-main/dataset_426/valAA"
    copy_files_across_folders(folder_a, folder_b, folder_c, folder_d, folder_e)
