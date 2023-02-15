from itertools import product
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CUB.cub_utils import download_from_aws

if __name__ == "__main__":
    endings = ["train.pkl", "test.pkl", "val.pkl"]
    folders = ["CUB_masked_class/", "CUB_processed/", "CUB_instance_masked/"]
    files = [a + b for a, b in product(folders, endings)]
    files += ["CUB_dataset.zip"]

    test_run = [
        "out/ind_XtoC/20221130-150657/final_model.pth",
        "out/ind_CtoY/20221130-194327/final_model.pth",
        "out/basic/20221201-041522/final_model.pth",
    ]
    files += test_run
    download_from_aws(files)
