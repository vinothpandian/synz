import json
from pathlib import Path

import fire
from fastcore.utils import parallel
from PIL import Image


def verify_image(image_path: Path):
    try:
        img = Image.open(image_path)
        img.verify()  # to veify if its an img
        img.close()  # to close img and free memory space
    except (IOError, SyntaxError):
        print("Bad file:", image_path)

def check_annotation(annotation):
    if annotation["area"] <= 0:
        print(f"Bad annotation at {annotation['id']}")


def verify_annotations(file):
    annotation_path = Path(file)
    coco_root_path = annotation_path.parent.parent
    coco_images_path = coco_root_path/"images"/annotation_path.stem

    with open(annotation_path, "r") as f:
        data = json.load(f)

    images_anno = data["images"]
    image_paths = [ coco_images_path/image_anno["file_name"] for image_anno in images_anno ]

    print("*** Verifying images **")
    parallel(verify_image,image_paths)
    print("*** Images listed in annotations exist **")


    print("*** Verifying annotations **")
    annotations = data["annotations"]
    parallel(check_annotation, annotations, threadpool=True)
    print("*** Annotations are valid **")


if __name__ == "__main__":
    fire.Fire(verify_annotations)
