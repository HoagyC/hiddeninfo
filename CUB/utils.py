import os
from pathlib import Path

import boto3
from botocore.exceptions import NoCredentialsError

BUCKET_NAME = "distilledrepr"


def upload_to_aws(local_file_name, s3_file_name: str="") -> bool:
    if "ACCESS_KEY" in os.environ:
        access_key = os.environ["ACCESS_KEY"]
    else:
        print("No AWS access key in environ")
        access_key = input("Enter AWS access key: ")

    if "SECRET_KEY" in os.environ:
        secret_key = os.environ["SECRET_KEY"]
    else:
        print("No AWS secret key in environ")
        secret_key = input("Enter AWS secret key: ")

    s3 = boto3.client('s3', aws_access_key_id=access_key,
                      aws_secret_access_key=secret_key)

    if not s3_file_name:
        s3_file_name = local_file_name
    
    local_file_path = Path(local_file_name)
    try:
        if local_file_path.is_dir():
            _upload_directory(local_file_name, s3)
        else:
            s3.upload_file(str(local_file_name), BUCKET_NAME, str(s3_file_name))
        print("Upload Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False

def _upload_directory(path, s3_client):
    for root, dirs, files in os.walk(path):
        for file_name in files:
            full_file_name = os.path.join(root, file_name)
            s3_client.upload_file(str(full_file_name), BUCKET_NAME, str(full_file_name))
