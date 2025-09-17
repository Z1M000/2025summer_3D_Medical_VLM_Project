from huggingface_hub import HfApi, upload_folder

# 替换成你的 repo 路径（用户名/数据集名）
REPO_ID = "Z1M000/ct_rate_to_m3d_5k"

# 本地路径（500个train数据在这个文件夹里）
LOCAL_FOLDER = "/Users/zimoli/Desktop/m3d/M3D/data/converted/ct_rate_to_m3d_cap_5k"


# 上传整个文件夹内容，替换旧的
# 1 log in hf
# huggingface-cli login
# 2 改allow_patterns
api = HfApi()

api.upload_large_folder(
    repo_id=REPO_ID,
    repo_type="dataset",
    folder_path=LOCAL_FOLDER,
    allow_patterns="train_4501_5000/**",  # 下次要改！！
    ignore_patterns=["*.DS_Store", "*.gitattributes"],
    print_report=True,
    print_report_every=60  # 每 60 秒打印一次上传状态
)



# api.delete_file(
#     path_in_repo="train_2000_2500/train_2000",  # 仓库里文件的相对路径
#     repo_id="Z1M000/ct_rate_to_m3d_5k",  # 改成你的 repo
#     repo_type="dataset"
# )
