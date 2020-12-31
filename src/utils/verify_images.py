from pathlib import Path

import fire
from fastcore.utils import *
from PIL import Image


def verify_image(image_path: Path):
    try:
        img = Image.open(image_path)
        img.verify()  # to veify if its an img
        img.close()  # to close img and free memory space
    except (IOError, SyntaxError):
        print("Bad file:", image_path)


def verify_images_in_dir(image_dir: str):
    path = Path(image_dir)
    parallel(verify_image, path.ls())


if __name__ == "__main__":
    fire.Fire(verify_images_in_dir)
