import os 
import pickle
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from transformers import CLIPProcessor, CLIPModel

# To be run from the CUB folder
IMAGE_PATH = Path("../CUB_200_2011/images")
OUTPUT_PATH = Path("../CUB_proccesed/CLIP_embeddings.pkl")
MODEL_NAME = "openai/clip-vit-base-patch32"

processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME)

folders = sorted(os.listdir(IMAGE_PATH))

embeddings = {}

for bird_folder in tqdm(folders):
    bird_folder_path = IMAGE_PATH / bird_folder

    for image_name in os.listdir(bird_folder_path):
        image_path = bird_folder_path / image_name 
        image = Image.open(image_path)
        processed_image = processor(images=image, return_tensors="pt")
        output = model.get_image_features(**processed_image)

        embeddings[image_path] = output

with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(embeddings, f)