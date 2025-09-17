import numpy as np
import plotly.express as px

# 加载 .npy 文件
image = np.load("data/converted/nii2npy/train_1_a.npy")

# 展开为 (slice, H, W)
slices = image[0]  # 去掉 batch 维度

# 创建交互式切片视图
fig = px.imshow(slices, animation_frame=0, binary_string=True)
fig.update_layout(title='3D Image Slices', dragmode=False)
fig.show()
