import os
from tqdm import tqdm

def process_images_folder(images_folder_path):
    # 遍历"images"文件夹中的所有文件
    for filename in os.listdir(images_folder_path):
        file_path = os.path.join(images_folder_path, filename)

        # 如果是jpg文件，则将后缀替换为png
        if filename.lower().endswith(".jpg"):
            new_file_path = os.path.splitext(file_path)[0] + ".png"
            os.rename(file_path, new_file_path)
        
        elif filename.lower().endswith(".jpeg"):
            new_file_path = os.path.splitext(file_path)[0] + ".png"
            os.rename(file_path, new_file_path)

def update_mtl_file(mtl_file_path):
    # 读取model.mtl文件的内容
    with open(mtl_file_path, 'r') as file:
        mtl_content = file.read()

    # 将导入jpg的行修改为导入相应的png
    mtl_content = mtl_content.replace('.jpg', '.png')
    mtl_content = mtl_content.replace('.jpeg', '.png')
    # 写回model.mtl文件
    with open(mtl_file_path, 'w') as file:
        file.write(mtl_content)

def process_folders(root_folder):
    # 遍历根文件夹及其子文件夹
    total_folders = sum([len(dirs) for _, dirs, _ in os.walk(root_folder)])
    for foldername, subfolders, filenames in tqdm(os.walk(root_folder), total=total_folders, desc="总体进度", unit="文件夹"):
        # 检查当前文件夹是否包含"images"文件夹
        if "images" in subfolders:
            images_folder_path = os.path.join(foldername, "images")
            
            # 处理"images"文件夹中的jpg文件
            process_images_folder(images_folder_path)

            # 修改"model.mtl"文件中的导入行
            mtl_file_path = os.path.join(foldername, "model.mtl")
            if os.path.isfile(mtl_file_path):
                update_mtl_file(mtl_file_path)


root_folder_path = '/root/ShapeNettest'
subfolders = [f for f in os.listdir(root_folder_path) if os.path.isdir(os.path.join(root_folder_path, f))]
with tqdm(subfolders) as t1:
    for type_folder in t1:
        type_source_folder = os.path.join(root_folder_path, type_folder)
        process_folders(type_source_folder)
print("处理完成。")
