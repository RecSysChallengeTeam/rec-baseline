import os
import zipfile
import pathlib

from urllib import request


# Link: https://grouplens.org/datasets/movielens/
DATASET_URL_DICT = {
    "movielens-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "movielens-full": "https://files.grouplens.org/datasets/movielens/ml-20m.zip",
}

ROOT_DIR = pathlib.Path.cwd().joinpath("dataset")


def download(url: str, path: str):
    """
    Download a file from a URL and save it locally under `path`.
    
    Args:
        url: URL of the file to download
        path: local path to save the file under    
    """
    
    print(f"Downloading {url} to {path}...")
    
    if not os.path.exists(path):
        request.urlretrieve(url, path)
        print("Done!")
    else:
        print("File already exists. Skipping download.")


def unzip(path_to_zip: str, target_dir: str):
    """
    Unzip a file to a target directory.
    
    Args:
        path_to_zip: path to the zip file
        target_dir: directory to unzip the file to
    """
        
    print(f"Unzipping {path_to_zip} to {target_dir}...")
    
    if not os.path.exists(target_dir):
        with zipfile.ZipFile(path_to_zip, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        print("Done!")
    else:
        print("File already exists. Skipping unzip.")
        

def download_and_unzip(url: str):
    # download the dataset
    path_to_save = ROOT_DIR.joinpath(url.split("/")[-1])
    
    if not os.path.exists(ROOT_DIR):
        os.makedirs(ROOT_DIR)
    
    download(url, path_to_save)
    
    # unzip the dataset
    output_folder = ROOT_DIR.joinpath("unzipped")
    unzip(path_to_save, output_folder)


if __name__ == "__main__":
    download_and_unzip(DATASET_URL_DICT["movielens-small"])
