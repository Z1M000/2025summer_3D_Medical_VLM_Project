import os
import numpy as np
import nibabel as nib
import monai.transforms as mtf
import pandas as pd
from tqdm import tqdm
import time

startTime = time.time()

transform = mtf.Compose([
    mtf.CropForeground(),
    mtf.Resize(spatial_size=[32, 256, 256], mode="bilinear")
])

#1 Download ct_rate_download in batches
#2 update output_base_dir
nii_base_dir = "ct_rate_data_volumes/dataset/train"
csv_path = "data/ct_rate_raw/chopped_train_reports.csv"
output_base_dir = "data/converted/ct_rate_to_m3d_cap_5k/train_4501_5000"
count = 1

print("Converting niigz to npy...")
for dirpath, _, filenames in os.walk(nii_base_dir):
    for file in filenames:
        if not file.endswith(".nii.gz"):
            continue
        
        # print(f"Processing: {file}")
        input_path = os.path.join(dirpath, file)
        volume_name = file.replace(".nii.gz", "")
        parent_name = "_".join(volume_name.split("_")[:2]) 
        output_dir = os.path.join(output_base_dir, parent_name, volume_name)
        os.makedirs(output_dir, exist_ok=True)
        print(count, volume_name)
        count += 1

        try:
            img = nib.load(input_path)
            data = img.get_fdata()  # 获取三维图像数据
            data = np.transpose(data, (2, 0, 1))  # 转换为 (depth, height, width) 格式

            data = data[np.newaxis, ...]  # 加通道维度 → (1, D, H, W)
            data = data - data.min()
            data = data / np.clip(data.max(), a_min=1e-8, a_max=None)  # 归一化

            data_trans = transform(data) # resize to (1, 32, 256, 256)
            np.save(os.path.join(output_dir, f"{volume_name}.npy"), data_trans)
        
        except Exception as e:
            print(f"❌ Failed: {file} | {e}")


endTime = time.time()
print(f"⏱️ Total time: {endTime - startTime:.2f} seconds")