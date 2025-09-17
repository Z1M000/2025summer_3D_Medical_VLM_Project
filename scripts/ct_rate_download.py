import shutil

import pandas as pd
import os

from huggingface_hub import hf_hub_download
from tqdm import tqdm


split = 'train'
batch_size = 0
start_at = 10672 #下次从train5001 是11805开始
count = 1

repo_id = 'ibrahimhamamci/CT-RATE'
directory_name = f'dataset/{split}/'
hf_token = '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'

# data = pd.read_csv('../data/ct_rate_raw/train_predicted_labels.csv')
current_dir = os.path.dirname(os.path.abspath(__file__))
label_path = os.path.join(current_dir, '..', 'data', 'ct_rate_raw', 'train_predicted_labels.csv')
data = pd.read_csv(label_path)

for i in tqdm(range(start_at, len(data), batch_size)):

    data_batched = data[i:i+batch_size]

    for name in data_batched['VolumeName']:
        print(count)
        count += 1
        folder1 = name.split('_')[0]
        folder2 = name.split('_')[1]
        folder = folder1 + '_' + folder2
        folder3 = name.split('_')[2]
        subfolder = folder + '_' + folder3
        subfolder = directory_name + folder + '/' + subfolder

        hf_hub_download(repo_id=repo_id,
            repo_type='dataset',
            token=hf_token,
            subfolder=subfolder,
            filename=name,
            cache_dir='./',
            local_dir='ct_rate_data_volumes',
            local_dir_use_symlinks=False,
            resume_download=True,
            )

    shutil.rmtree('./datasets--ibrahimhamamci--CT-RATE')