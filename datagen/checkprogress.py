import os

def count_train_folders(root_folder):
    count = 0

    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)

        if os.path.isdir(folder_path):
            #if (len(os.listdir(folder_path)) != 8):
               # print(f"error at{folder_path}")
            subfolders = [f"{i:02d}" for i in range(8)]
            if all(os.path.isdir(os.path.join(folder_path, subfolder)) for subfolder in subfolders):
                if all(
                    all(
                        os.path.isfile(os.path.join(folder_path, subfolder, file))
                        for file in ["input.pcd", "label.npy", "gt.pcd"]
                    )
                    for subfolder in subfolders
                ):
                    count += 1
                #"""
                else:
                    print(root_folder)
                    print(folder_name)
            else:
                print(folder_path)#"""
    return count

def count_test_folders(root_folder):
    count = 0

    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)

        if os.path.isdir(folder_path):
            #if (len(os.listdir(folder_path)) != 1):
                #print(f"error at{folder_path}")
            subfolders = [f"{i:02d}" for i in range(1)]
            if all(os.path.isdir(os.path.join(folder_path, subfolder)) for subfolder in subfolders):
                if all(
                    all(
                        os.path.isfile(os.path.join(folder_path, subfolder, file))
                        for file in ["input.pcd", "label.npy", "gt.pcd"]
                    )
                    for subfolder in subfolders
                ):
                    count += 1
                #"""
                else:
                    print(root_folder)
                    print(folder_name)
            else:
                print(folder_path)#"""
    return count

def count_models(folder_path):

    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    output = []
    for subfolder in subfolders:
        category_folder = os.path.join(folder_path, subfolder)
        object_subfolders = [f for f in os.listdir(category_folder) if os.path.isdir(os.path.join(category_folder, f))]
        count = 0
        for object_id in object_subfolders:
            object_path =  os.path.join(category_folder, object_id)
            if (os.path.isfile(os.path.join(object_path, file))
                        for file in ["model.obj", "model.stl"]):
                count += 1
        expect_model = [os.path.basename(subfolder), count]
        output.append(expect_model)
    return output



base_dir = '/root/ShapeNettest'
root_dir = "/root/Bound57"  
"""base_count = count_models(base_dir)
model_count_dict = {model_id: count for model_id, count in base_count}
print(model_count_dict)
total_count = 0
for model_id, count in base_count:
    total_count += count
print(f"total count: {total_count}")"""

train_dir = os.path.join(root_dir, "train")
test_dir = os.path.join(root_dir, "test")
val_dir = os.path.join(root_dir, "val")
train_folders = set(os.listdir(train_dir))
test_folders = set(os.listdir(test_dir))
val_folders = set(os.listdir(val_dir))

common_folders = train_folders.intersection(test_folders, val_folders)
common_train_folders_list = [os.path.join(train_dir, folder) for folder in common_folders]
common_test_folders_list = [os.path.join(test_dir, folder) for folder in common_folders]
common_val_folders_list = [os.path.join(val_dir, folder) for folder in common_folders]
common_folders_list = list(common_folders)
base_count = count_models(base_dir)
model_count_dict = {model_id: count for model_id, count in base_count}
total_count = 0
total_expect = 0
count = 0
for i in range(len(common_train_folders_list)):
    result_count = count_train_folders(common_train_folders_list[i])
    result_count += count_test_folders(common_test_folders_list[i])
    result_count += count_train_folders(common_val_folders_list[i])
    total_count += result_count
    expect_count = model_count_dict.get(common_folders_list[i], result_count)
    total_expect += (expect_count - result_count)
    print(f"Number of {common_folders_list[i]}: {result_count}, expecting: {expect_count-result_count}")
    if (expect_count-result_count) > 0:
        count += 1
print(f"Total categories: {len(common_train_folders_list)}, expecting categories: {count}, total models: {total_count}, total expecting: {total_expect}")