import json
import csv
from tqdm import tqdm
import os
import sys

# input_path = "data/ct_rate_raw/chopped_valid_vqa.json" 
# output_path = "data/converted/chopped_valid_vqa.csv"

input_path = "data/ct_rate_raw/train_vqa.json" 
output_path = "data/converted/converted_vqa_frq_5k_short_answer.csv"

# define CSV fields
fieldnames = [
    "Image Path", "Text", "Question", "Answer"
]

count = 0

with open(input_path, "r") as f_json, open(output_path, "w", newline='', encoding="utf-8") as f_csv:
    data = json.load(f_json)

    writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
    writer.writeheader()

    for item in tqdm(data, desc="Processing FRQs"):
        count += 1
        if count % 100000 == 0:
            print(f"已遇见 {count} 个样本")
        
        if not item["id"].startswith("short_answer_"):
            continue

        # image_path = item.get("image", "")
        raw_image_name = item.get("image", "").replace(".nii.gz", "")

        if raw_image_name == "train_5001_a_1":
            sys.exit("遇到 train_5001_a_1，程序已退出 ✅")
        
        # raw_image_name = raw_image_name.replace("valid", "train") #转真正的train.json时候不用这句！
        parent = "_".join(raw_image_name.split("_")[:2])
        image_path = f"converted/ct_rate_to_m3d_cap_5k/{parent}/{raw_image_name}/{raw_image_name}.npy"
        
        text_path = f"data/converted/ct_rate_to_m3d_cap_5k/text_paths/{parent}/{raw_image_name}/text.txt"
        if os.path.exists(text_path):
            with open(text_path, "r") as f:
                text = f.read().strip()
        else:
            text = ""

        conversations = item.get("conversations", [])

        # 按每两个对话成对处理
        for i in range(0, len(conversations) - 1, 2):
            human = conversations[i]
            gpt = conversations[i + 1]

            question = human["value"].replace('\n', ' ').replace("<short_answer>", "")
            if question.startswith("<image> "):
                question = question.replace("<image> ", "")

            answer = gpt["value"].strip()

            writer.writerow({
                "Image Path": image_path,
                # "Text Path": text_path,
                "Text": text, 
                "Question": question,
                "Answer": answer,
            })

print(f"✅ Done! Output saved to {output_path}\n\n\n")
