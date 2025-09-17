import os
import numpy as np
import nibabel as nib
import monai.transforms as mtf
import pandas as pd
from tqdm import tqdm
import time

startTime = time.time()

#1 make sure chopped_train_reports.csv is updated
#2 update output_base_dir
nii_base_dir = "ct_rate_data_volumes/dataset/train"
csv_path = "data/ct_rate_raw/chopped_train_reports.csv"
output_base_dir = "data/converted/ct_rate_to_m3d_cap_5k/train_4501_5000"

print("Adding text files...")
df = pd.read_csv(csv_path)

for row in df.itertuples():
    volume_name = row.VolumeName.replace(".nii.gz", "")
    technique = row.Technique_EN
    findings = row.Findings_EN
    impressions = row.Impressions_EN
    full_text = f"Technique: {technique}\nFindings: {findings}\nImpressions: {impressions}"
   
    parent_name = "_".join(volume_name.split("_")[:2])
    output_dir = os.path.join(output_base_dir, parent_name, volume_name)
    os.makedirs(output_dir, exist_ok=True)
    text_path = os.path.join(output_dir, "text.txt")
    
    with open(text_path, "w") as f:
        f.write(full_text)

endTime = time.time()
print(f"⏱️ Total time: {endTime - startTime:.2f} seconds")