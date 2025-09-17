import json

with open('data/ct_rate_raw/train_vqa.json', 'r') as f:
    for i, line in enumerate(f):
        print(line.strip())
        if i >= 1000:
            break


# with open('data/ct_rate_raw/train_vqa.json', 'r') as f:
#     for i, line in enumerate(f):
#         print(line.strip())
#         if i >= 10:  # Only show first 10 lines
#             break
