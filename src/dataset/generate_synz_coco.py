import json
import os
from pathlib import Path

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

from datetime import datetime

import numpy as np
from fastcore.utils import parallel
from PIL import Image
from pygame import Rect
from skimage import color, filters, io
from src.utils.itemgetters import get_image_size, get_rect_data

df = None
uisketch_df = None


def get_category_dict(category, i):
    return {"id": i, "supercategory": "none", "name": category}


def get_image_dict(filename, i, width, height):
    return {
        "id": int(i),
        "width": int(width),
        "height": int(height),
        "file_name": filename,
        "license": 0,
        "flickr_url": "",
        "coco_url": "",
        "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


class GenerateSynZCoco(object):
    def __init__(self, uisketch_labels_csv_path, output_path, train_test_valid):

        self.uisketch_path = Path(uisketch_labels_csv_path).parent
        self.synz_path = Path(output_path)

        (self.synz_path / "images" / "train").mkdir(exist_ok=True, parents=True)
        (self.synz_path / "images" / "val").mkdir(exist_ok=True)
        (self.synz_path / "images" / "test").mkdir(exist_ok=True)
        (self.synz_path / "annotations").mkdir(exist_ok=True)

        self.image_map = {filename: i for i, filename in enumerate(df.filename.unique())}

        self.categories = df.category.unique()

        self.category_map = {category: i + 1 for i, category in enumerate(sorted(self.categories))}
        self.categories_data = [get_category_dict(category, i) for category, i in self.category_map.items()]

        self.train_test_valid = train_test_valid

    def generate_synz(self, filename):
        f_df = df.query("filename == @filename")

        image_id = self.image_map[filename]
        width, height = get_image_size(f_df.iloc[0])

        image_data = get_image_dict(filename, image_id, width, height)
        annotations = []

        canvas = Image.new("RGB", (width, height), (255, 255, 255))
        for i, row in f_df.iterrows():
            r = Rect(get_rect_data(row))

            category = row["category"]
            category_id = self.category_map[category]
            uisketch_id = row["uisketch_id"]
            area = r.width * r.height

            image_name = uisketch_df.loc[uisketch_id]["name"]
            image_path = self.uisketch_path / image_name

            uisketch_image = Image.open(image_path)
            uisketch_image = uisketch_image.resize(r.size)

            canvas.paste(uisketch_image, r.topleft)
            _id = int((image_id * 1e6) + i)

            annotation = {
                "id": _id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": [],
                "area": area,
                "bbox": [r.left, r.top, r.width, r.height],
                "iscrowd": 0,
            }

            annotations.append(annotation)

        output_path = self.synz_path / "images" / self.train_test_valid / filename

        image = np.array(canvas).astype(np.uint8)
        image = color.rgb2gray(image)
        threshold = filters.threshold_otsu(image)
        image = np.where(image > threshold, 0, 255)
        image = color.gray2rgb(image)
        io.imsave(output_path, image.astype(np.uint8), check_contrast=False)

        return image_data, annotations

    def generate(self):
        annotation_data = parallel(self.generate_synz, df.filename.unique())

        image_data = []
        annotations = []
        for img_data, anno in annotation_data:
            image_data.append(img_data)
            annotations.extend(anno)

        filename = f"{self.train_test_valid}.json"

        path = self.synz_path / "annotations" / filename

        now = datetime.now()

        with open(path, "w") as f:
            data = {
                "info": {
                    "year": now.year,
                    "version": "1.0.0",
                    "description": "Enhanced Synthetic Dataset for Training UI Element Detector From Lo-Fi Sketches",
                    "contributor": "Vinoth Pandian",
                    "url": "https://www.kaggle.com/vinothpandian/synz-dataset",
                    "date_created": now.strftime("%Y/%m/%d"),
                },
                "licenses": [
                    {"id": 0, "name": "CC-BY-SA 4.0", "url": "https://creativecommons.org/licenses/by-sa/4.0/"}
                ],
                "images": image_data,
                "annotations": annotations,
                "categories": self.categories_data,
            }
            json.dump(
                data,
                f,
                default=lambda o: o.__dict__,
            )

        return path
