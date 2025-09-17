import nibabel as nib
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import os
import re

# 你可以替换成任何下载下来的文件名
#filename = '../data/ct_rate_data_volumes/train_1_a_1.nii.gz'
#filename = "data/ct_rate_data_volumes/dataset/train/train_55/train_55_b/train_55_b_5.nii.gz"
filename = "ct_rate_data_volumes/dataset/train/train_406/train_406_a/train_406_a_1.nii.gz"

def view(filename):
    # 加载 .nii.gz 文件
    img = nib.load(filename)
    data = img.get_fdata()  # 获取三维图像数据 (shape like 512 x 512 x 200)
    data = np.transpose(data, (2, 0, 1))
    fig = px.imshow(data, animation_frame=0, binary_string=True)
    fig.update_layout(title_text=filename.split('/')[-1])
    fig.show()

view(filename)

# def natural_sort_key(s):
#     """用正则表达式提取字符串中的数字，支持 train_2 在 train_10 前"""
#     return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

# base = "data/ct_rate_data_volumes/dataset/train"
# # i = 1

# all_files = []
# for dirpath, _, filenames in os.walk(base):
#     for file in filenames:
#         if file.endswith('.nii.gz'):
#             full_path = os.path.join(dirpath, file)
#             all_files.append(full_path)

# # 自然排序
# all_files = sorted(all_files, key=natural_sort_key)

# # 只看前几个
# for path in all_files[0:0]:
#     print(f"Viewing: {path}")
#     view(path)


# for dirpath, _, filenames in os.walk(base):
#     for file in filenames:
#         if not file.endswith('.nii.gz'):
#             continue
#         if i > 1:
#             break
#         view(os.path.join(dirpath, file))
#         i += 1