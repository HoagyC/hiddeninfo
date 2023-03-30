import json
import os
import sys
import random
from pathlib import Path
from typing import List, Dict

from matplotlib.pyplot import figure, imshow, axis, show
from matplotlib.image import imread

import boto3
from botocore.exceptions import NoCredentialsError, ClientError

BUCKET_NAME = "distilledrepr"

def get_secrets() -> Dict:
    assert "secrets.json" in os.listdir(sys.path[-1])
    with open(os.path.join(sys.path[-1], "secrets.json"), "r") as f:
        secrets = json.load(f)

    return secrets

def kill_python_processes() -> None:
    """Kill all python processes."""
    processes = os.popen("ps -Af").read() # -A and -f are for all processes and full format
    current_pid = os.getpid()
    for line in processes.splitlines():
        if "python" in line.lower() and str(current_pid) not in line:
            print("Killing process: " + line)
            pid = int(line.split()[1])
            os.kill(pid, 9)

# List all the objects in the specified folder(one layer only)
def list_aws_files(folder_name: str, get_folders: bool = True) -> List[str]:
    secrets = get_secrets()

    s3 = boto3.client(
        "s3",
        aws_access_key_id=secrets["access_key"],
        aws_secret_access_key=secrets["secret_key"],
    )
    # Add a trailing slash if missing to get folder names 
    if folder_name[-1] != "/":
        folder_name += "/"

    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=folder_name, Delimiter="/")
    if get_folders:
        # Get subdiretories which are the different runs under that name and sort them by date (no folder path)
        results = [content["Prefix"] for content in response["CommonPrefixes"]]
        results.sort()
    else:
        # Get objects within the folder (includes the rest of the folder path)
        results = [content["Key"] for content in response["Contents"]]
        results.sort()
    
    return results


def upload_to_aws(local_file_name, s3_file_name: str = "") -> bool:
    """"
    Upload a file to an S3 bucket
    :param local_file_name: File to upload
    :param s3_file_name: S3 object name. If not specified then local_file_name is used
    """
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
        print(f"File {local_file_name} was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False


def download_from_aws(files: List[str], force_redownload: bool = False) -> bool:
    """
    Download a file from an S3 bucket
    :param files: List of files to download
    :param force_redownload: If True, will download even if the file already exists
    
    Returns:
        True if all files were downloaded successfully, False otherwise
    """
    secrets = get_secrets()
    if not force_redownload:
        files = [f for f in files if not os.path.exists(f)]

    s3 = boto3.client(
        "s3",
        aws_access_key_id=secrets["access_key"],
        aws_secret_access_key=secrets["secret_key"],
    )
    all_correct = True
    for filename in files:
        try:
            parent_dir = os.path.dirname(filename)
            if not os.path.exists(parent_dir) and parent_dir != "":
                os.makedirs(os.path.dirname(filename))
            with open(filename, "wb") as f:
                s3.download_fileobj(BUCKET_NAME, filename, f)

            print(f"Successfully downloaded file: {filename}")
        except ClientError:
            print(f"File: {filename} does not exist")
            all_correct = False

    return all_correct


def download_folder_from_aws(folder_name: str, force_redownload: bool = False) -> None:
    secrets = get_secrets()
    s3_resource = boto3.resource('s3', aws_access_key_id=secrets["access_key"], aws_secret_access_key=secrets["secret_key"])
    bucket = s3_resource.Bucket(BUCKET_NAME) 
    for obj in bucket.objects.filter(Prefix = folder_name):
        if not force_redownload and os.path.exists(obj.key):
            continue
        try:
            if not os.path.exists(os.path.dirname(obj.key)):
                os.makedirs(os.path.dirname(obj.key))
            bucket.download_file(obj.key, obj.key)
            print(f"Successfully downloaded file: {obj.key}")
        except ClientError:
            print(f"File: {obj.key} does not exist")



def _upload_directory(path, s3_client):
    for root, dirs, files in os.walk(path):
        for file_name in files:
            full_file_name = os.path.join(root, file_name)
            s3_client.upload_file(str(full_file_name), BUCKET_NAME, str(full_file_name))


def make_class_id_to_folder_dict(
    img_dir="CUB_200_2011/images/",
):
    """
    Returns:
    class_to_folder: map class id (0 to 199) to the path to the corresponding image folder (containing actual class names)
    """
    class_to_folder = dict()
    for folder in os.listdir(img_dir):
        class_id = int(folder.split(".")[0])
        class_to_folder[class_id - 1] = os.path.join(img_dir, folder)
    return class_to_folder


def make_attr_id_to_name_dict(feature_file="CUB_200_2011/attributes/image_attribute_labels.txt"):
    """
    Returns:
    attr_id_to_name: map attribute id (0 to 311) to actual attribute name read from feature_file argument
    """

    attr_id_to_name = dict()
    with open(feature_file, "r") as f:
        for line in f:
            idx, name = line.strip().split(" ")
            attr_id_to_name[int(idx) - 1] = name
    return attr_id_to_name


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


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "kill":
        kill_python_processes()