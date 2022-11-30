import json
import os
from pathlib import Path
from typing import List, Dict

import boto3
from botocore.exceptions import NoCredentialsError

BUCKET_NAME = "distilledrepr"


def get_secrets() -> Dict:
    assert "secrets.json" in os.listdir()
    with open("secrets.json", "r") as f:
        secrets = json.load(f)

    return secrets


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
        with open(filename, "wb") as f:
            s3.download_fileobj(BUCKET_NAME, filename, f)


def _upload_directory(path, s3_client):
    for root, dirs, files in os.walk(path):
        for file_name in files:
            full_file_name = os.path.join(root, file_name)
            s3_client.upload_file(str(full_file_name), BUCKET_NAME, str(full_file_name))
