import os
import json

dataset_path = "/root/Bound57"
output_json_path = "/root/PcBD/data_utils/Bound57.json"

result = []

# 获取所有目录
taxonomy_ids = sorted(os.listdir(os.path.join(dataset_path, "train")))

# 遍历每个taxonomy_id
for taxonomy_id in taxonomy_ids:
    taxonomy_data = {"taxonomy_id": taxonomy_id}

    # 遍历train、test和val目录
    for split in ["train", "test", "val"]:
        split_path = os.path.join(dataset_path, split, taxonomy_id)
        models = sorted(os.listdir(split_path))
        model_ids = [model for model in models if os.path.isdir(os.path.join(split_path, model))]
        taxonomy_data[split] = model_ids

    result.append(taxonomy_data)

# 将结果写入JSON文件
with open(output_json_path, "w") as json_file:
    json.dump(result, json_file, indent=2)

print(f"JSON file has been generated at {output_json_path}")
