import os
import pickle
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

def view_image(data, attr_names):
    """
    Takes an image that has been processed into a CUB dataset and displays it,
    along with its class and attribute labels.
    """

    # Get the image
    img_path = data["img_path"] # string pointing to jpg

    # Replace the first part of the path "Users/hoagycunningham" if on mac
    img_path = img_path.replace("root", "Users/hoagycunningham")
    img = Image.open(img_path).convert("RGB")
    
    # Convert to numpy array and normalize
    img = np.array(img)
    img = img / 255.0
    img = np.clip(img, 0, 1)

    # Get the class and attribute labels
    id = data["id"]
    class_label = data["class_label"]
    attribute_label = data["attribute_label"]
    attribute_certainty = data["attribute_certainty"]
    uncertain_attribute_label = data["uncertain_attribute_label"]

    
    # import pdb; pdb.set_trace()
    # Write the names and whether they are active or not to the right of the image on the plot
    
    # Display the image
    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    axs[0].imshow(img)
    axs[1].text(0.5, 0.2, '\n'.join([a for n, a in enumerate(attr_names) if attribute_label[n]]), wrap=True, horizontalalignment='center', fontsize=12, linespacing=0.4)
    axs[1].get_xaxis().set_visible(False)
    axs[1].get_yaxis().set_visible(False)
    plt.axis("off")

    # Put title above the image in axs[0]
    axs[0].set_title(f"Class: {class_label}, ID: {id}")   
    plt.show()

if __name__ == "__main__":
    # Load the dataset
    data_path = "CUB_instance_masked/train.pkl"
    with open(data_path, "rb") as train_data_file:
        dataset = pickle.load(train_data_file)

    attr_name_path = "CUB_masked_class/attributes.txt"
    with open(attr_name_path, "r") as attr_name_file:
        attr_names = attr_name_file.readlines()
    
    # View an image
    while True:
        i = np.random.randint(0, len(dataset))
        view_image(dataset[i], attr_names)
        time.sleep(2) # Use this time to quit the program after closing an image