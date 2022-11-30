from itertools import product
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CUB.utils import download_from_aws

if __name__ == "__main__":
    endings = ["train.pkl", "test.pkl", "val.pkl"]
    folders = ["CUB_masked_class/", "CUB_processed/"]
    files = [a + b for a, b in product(folders, endings)]
    files += ["CUB_dataset.zip"]
    download_from_aws(files)
