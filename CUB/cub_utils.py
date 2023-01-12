import json
import os
import random
from pathlib import Path
from typing import List, Dict

from matplotlib.pyplot import figure, imshow, axis, show
from matplotlib.image import imread

import boto3
from botocore.exceptions import NoCredentialsError

BUCKET_NAME = "distilledrepr"


def get_secrets() -> Dict:
    assert "secrets.json" in os.listdir()
    with open("secrets.json", "r") as f:
        secrets = json.load(f)

    return secrets

# List all the objects in the specified folder(one layer only)
def list_files(folder_name: str):
    secrets = get_secrets()

    s3 = boto3.client(
        "s3",
        aws_access_key_id=secrets["access_key"],
        aws_secret_access_key=secrets["secret_key"],
    )
    # Add a trailing slash if missing to get folder names 
    if folder_name[-1] != "/":
        folder_name += "/"

    # Get subdiretories which are the different runs under that name and sort them by date
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=folder_name, Delimiter="/")
    folders = [content["Prefix"] for content in response["CommonPrefixes"]]
    folders.sort()
    
    return folders


def upload_to_aws(local_file_name, s3_file_name: str = "") -> bool:
    secrets = get_secrets()

    s3 = boto3.client(
        "s3",
        aws_access_key_id=secrets["access_key"],
        aws_secret_access_key=secrets["secret_key"],
    )

    if not s3_file_name:
        s3_file_name = local_file_name

    local_file_path = Path(local_file_name)
    try:
        if local_file_path.is_dir():
            _upload_directory(local_file_name, s3)
        else:
            s3.upload_file(str(local_file_name), BUCKET_NAME, str(s3_file_name))
        print(f"Upload Successful of {local_file_name}")
        return True
    except FileNotFoundError:
        print(f"Flle {local_file_name} was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False


def download_from_aws(files: List[str]) -> None:
    secrets = get_secrets()

    s3 = boto3.client(
        "s3",
        aws_access_key_id=secrets["access_key"],
        aws_secret_access_key=secrets["secret_key"],
    )
    for filename in files:
        print(f"Downloading {filename}")
        parent_dir = os.path.dirname(filename)
        if not os.path.exists(parent_dir) and parent_dir != "":
            os.makedirs(os.path.dirname(filename))
        with open(filename, "wb") as f:
            s3.download_fileobj(BUCKET_NAME, filename, f)


def _upload_directory(path, s3_client):
    for root, dirs, files in os.walk(path):
        for file_name in files:
            full_file_name = os.path.join(root, file_name)
            s3_client.upload_file(str(full_file_name), BUCKET_NAME, str(full_file_name))


def get_class_attribute_names(
    img_dir="CUB_200_2011/images/",
    feature_file="CUB_200_2011/attributes/attributes.txt",
):
    """
    Returns:
    class_to_folder: map class id (0 to 199) to the path to the corresponding image folder (containing actual class names)
    attr_id_to_name: map attribute id (0 to 311) to actual attribute name read from feature_file argument
    """
    class_to_folder = dict()
    for folder in os.listdir(img_dir):
        class_id = int(folder.split(".")[0])
        class_to_folder[class_id - 1] = os.path.join(img_dir, folder)

    attr_id_to_name = dict()
    with open(feature_file, "r") as f:
        for line in f:
            idx, name = line.strip().split(" ")
            attr_id_to_name[int(idx) - 1] = name
    return class_to_folder, attr_id_to_name


def sample_files(class_label, class_to_folder, number_of_files=10) -> List[str]:
    """
    Given a class id, extract the path to the corresponding image folder and sample number_of_files randomly from that folder
    """
    folder = class_to_folder[class_label]
    class_files = random.sample(os.listdir(folder), number_of_files)
    class_files = [os.path.join(folder, f) for f in class_files]
    return class_files


def show_img_horizontally(list_of_files) -> None:
    """
    Given a list of files, display them horizontally in the notebook output
    """
    fig = figure(figsize=(40, 40))
    number_of_files = len(list_of_files)
    for i in range(number_of_files):
        a = fig.add_subplot(1, number_of_files, i + 1)
        image = imread(list_of_files[i])
        imshow(image)
        axis("off")
    show(block=True)
