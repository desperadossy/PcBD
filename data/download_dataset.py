import os
import tarfile
import urllib.request

url = "https://ssy-pcdatasets.oss-cn-hangzhou.aliyuncs.com/Bound57"
tar_path = "data/Bound57.tar.gz"
extract_path = "data/"

def download_and_extract():

    print(f"⬇ Downloading Bound57 dataset from:\n{url}")
    urllib.request.urlretrieve(url, tar_path)
    print("✅ Download complete.")

    print("📦 Extracting .tar.gz ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_path)

    os.remove(tar_path)
    print("✅ Extraction complete. Dataset is ready at ./data/Bound57")

if __name__ == "__main__":
    download_and_extract()
