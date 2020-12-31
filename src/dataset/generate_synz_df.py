import os
from functools import partial
from pathlib import Path

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"


import pandas as pd
from fastcore.foundation import L
from fastcore.utils import parallel
from pygame import Rect
from src.utils.itemgetters import get_image_size, get_rect_data, get_size

synz_columns = [
    "uisketch_id",
    "filename",
    "category",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "width",
    "height",
    "image_width",
    "image_height",
]

annotation_df: pd.DataFrame = None
uisketch_df: pd.DataFrame = None


def generate(output_csv_path: Path, version: int):

    filenames = annotation_df.filename.unique()

    # # Uncomment below for generating tiny_synz dataset for testing
    # if "train" in str(output_csv_path):
    #     filenames = filenames[:1000]
    # else:
    #     filenames = filenames[:500]

    get_sketch_data_for_version = partial(get_sketch_data, version=version)

    synz_data = parallel(get_sketch_data_for_version, filenames)
    synz_df = pd.concat(synz_data, ignore_index=True)

    synz_df.to_csv(output_csv_path, index=False)


def get_sketch_data(filename: str, version: int):
    df = annotation_df.query(f"filename == @filename")
    result = L()
    chosen_uisketch = set()
    current_filename = f"{version}_{filename}"

    for i, row in df.iterrows():
        category = row["category"]
        xmin, ymin, width, height = get_rect_data(row)
        image_width, image_height = get_image_size(row)

        exp_width, exp_height = 576, 1024

        asp = width / height
        off = 0.05

        possible_items = None
        base_query = asp_query = "label == @category & index not in @chosen_uisketch"

        retry = 10
        while retry > 0:

            if asp > 1:
                asp_query = base_query + " & aspect_ratio > 1"
            else:
                asp_query = base_query + " & aspect_ratio <= 1"

            min_asp = asp - off
            max_asp = asp + off

            query = asp_query + f" & {min_asp} <= aspect_ratio <= {max_asp}"
            possible_items = uisketch_df.query(query)
            if len(possible_items) > 0:
                break
            else:
                off *= 2
                retry -= 1

        if retry <= 0:
            possible_items = uisketch_df.query(asp_query)
            if len(possible_items) == 0:
                possible_items = uisketch_df.query(base_query)

        if possible_items is None:
            continue

        chosen_row = possible_items.sample(1, random_state=i + version).iloc[0]
        uisketch_id = chosen_row.name

        chosen_uisketch.add(uisketch_id)
        sketch_w, sketch_h = get_size(chosen_row)

        # Resize
        width_ratio = exp_width / image_width
        height_ratio = exp_height / image_height

        xmin = int(xmin * width_ratio)
        ymin = int(ymin * height_ratio)
        width = int(width * width_ratio)
        height = int(height * height_ratio)

        annotation_rect = Rect(xmin, ymin, width, height)
        uisketch_rect = Rect(xmin, ymin, sketch_w, sketch_h)

        r = uisketch_rect.fit(annotation_rect)
        r.topleft = annotation_rect.topleft

        r.normalize()

        if r.width <= 0 or r.height <= 0:
            continue

        current_result = [
            uisketch_id,
            current_filename,
            category,
            r.left,
            r.top,
            r.right,
            r.bottom,
            r.width,
            r.height,
            exp_width,
            exp_height,
        ]
        result.append(dict(zip(synz_columns, current_result)))
    return pd.DataFrame(result)


def generate_synz(
    annotation_csv_path: str,
    uisketch_labels_csv_path: str,
    output_path: str,
    version: int,
):
    global annotation_df, uisketch_df

    loaded_annotations_df = pd.read_csv(annotation_csv_path)
    loaded_uisketch_df = pd.read_csv(uisketch_labels_csv_path, index_col=0)

    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    # Train
    print("*** Generating training data ***")
    train_annotations_df = loaded_annotations_df.query("train_test_valid == 'train'")
    train_uisketch_df = loaded_uisketch_df.query("train_test_valid == 'train'")
    train_output_csv_path = output_path / f"SynZ_generated_train_v{version}.csv"

    annotation_df = train_annotations_df
    uisketch_df = train_uisketch_df

    generate(train_output_csv_path, version=version)
    print(f"*** Generated training data at {str(train_output_csv_path)} ***")

    print("\n")

    # Validation
    print("*** Generating validation data ***")
    valid_annotations_df = loaded_annotations_df.query("train_test_valid == 'valid'")
    valid_uisketch_df = loaded_uisketch_df.query("train_test_valid == 'valid'")
    valid_output_csv_path = output_path / f"SynZ_generated_valid_v{version}.csv"

    annotation_df = valid_annotations_df
    uisketch_df = valid_uisketch_df

    if len(valid_annotations_df) > 0:
        generate(valid_output_csv_path, version=version)
        print(f"*** Generated validation data at {str(valid_output_csv_path)} ***")

    print("\n")

    # Test
    print("*** Generating testing data ***")
    test_annotations_df = loaded_annotations_df.query("train_test_valid == 'test'")
    test_uisketch_df = loaded_uisketch_df.query("train_test_valid == 'test'")
    test_output_csv_path = output_path / f"SynZ_generated_test_v{version}.csv"

    annotation_df = test_annotations_df
    uisketch_df = test_uisketch_df

    generate(test_output_csv_path, version=version)
    print(f"*** Generated testing data at {str(test_output_csv_path)} ***")

    return train_output_csv_path, valid_output_csv_path, test_output_csv_path
