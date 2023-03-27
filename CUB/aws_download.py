from itertools import product
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CUB.cub_utils import download_from_aws

# This downloads the basic files used for various experiments, as part of the setup process for a new server
if __name__ == "__main__":
    endings = ["train.pkl", "test.pkl", "val.pkl"]
    folders = ["CUB_masked_class/", "CUB_processed/", "CUB_instance_masked/"]
    files = [a + b for a, b in product(folders, endings)]
    files += ["CUB_dataset.zip"]

    download_from_aws(files)
