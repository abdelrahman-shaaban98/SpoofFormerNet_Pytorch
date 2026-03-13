import os
import glob
from tqdm import tqdm
from PIL import Image
import numpy as np

from transformers import pipeline

# Used in inference on one test image
def estimate_depth(rgb_path: str) -> np.array:
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    image = Image.open(rgb_path)
    depth = pipe(image)["depth"]
    depth_np = np.array(depth)
    # Image.fromarray(depth_np).save("images/depth_map.png")
    return depth_np


# Used to generate depth for the training data
def estimate_depth_dir(input_dir: str, output_dir:str):
    # input_dir = "data/train_img/color"
    # output_dir = "data/train_img/depth_generated"

    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", use_fast=True)
    os.makedirs(output_dir, exist_ok=True)
    image_paths = glob.glob(os.path.join(input_dir, "*"))

    for path in tqdm(image_paths, leave=False):
        image = Image.open(path)
        depth = pipe(image)["depth"]
        depth_np = np.array(depth)
        filename = os.path.basename(path)
        save_path = os.path.join(output_dir, filename)
        Image.fromarray(depth_np).save(save_path)