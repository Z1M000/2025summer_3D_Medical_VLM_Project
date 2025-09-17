import json
import csv
import re
from tqdm import tqdm
import os
import sys

input_path = "data/ct_rate_raw/train_vqa.json" 
output_path = "data/converted/converted_vqa_mcq_5k.csv"

# define CSV fields
fieldnames = [
    "Image Path", "Text", "Question Type", "Question",
    "Choice A", "Choice B", "Choice C", "Choice D",
    "Answer", "Answer Choice"
]

# 正则表达式匹配 pattern
pattern = re.compile(r"(?:<image>\n)?(.*?)\(a\)(.*?)\(b\)(.*?)\(c\)(.*?)\(d\)(.*?)<multiple_choice>")
count = 0

with open(input_path, "r") as f_json, open(output_path, "w", newline='', encoding="utf-8") as f_csv:
    data = json.load(f_json)

    writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
    writer.writeheader()

    for item in tqdm(data, desc="Processing MCQs"):
        count += 1
        if count % 100000 == 0:
            print(f"已见 {count} 个样本")
        
        if not item["id"].startswith("multiple_choice_"):
            continue  # 跳过非 MCQ 的 sample

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

            if human.get("type") != "multiple_choice":
                continue

            question_raw = human["value"].replace('\n', ' ')

            match = pattern.match(question_raw)
            if not match:
                print("⚠️ 跳过格式错误的问题：", question_raw[:100])
                continue

            question = match.group(1).strip()
            if question.startswith("<image> "):
                question = question.replace("<image> ", "")
            choices = [match.group(j).strip() for j in range(2, 6)]

            # 提取答案字母和文本
            answer_raw = gpt["value"].strip()
            answer_letter = ""
            answer_text = answer_raw
            m = re.match(r"\(([a-dA-D])\)", answer_raw)
            if m:
                answer_letter = m.group(1).upper()
                answer_text = answer_raw.split(")", 1)[1].strip()

            writer.writerow({
                "Image Path": image_path,
                # "Text Path": text_path,
                "Text": text,  # 你要求保留但空着
                "Question Type": "",  # 你要求保留但空着
                "Question": question,
                "Choice A": choices[0],
                "Choice B": choices[1],
                "Choice C": choices[2],
                "Choice D": choices[3],
                "Answer": answer_text,
                "Answer Choice": answer_letter
            })

print(f"✅ Done! Output saved to {output_path}\n\n\n")
