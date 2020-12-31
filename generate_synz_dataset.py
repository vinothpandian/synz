import os
import sys
from pathlib import Path

from src.dataset import generate_synz_coco, generate_synz_df

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import fire
import pandas as pd


class Pipeline(object):
    def __init__(
        self,
        annotation_csv_path: str,
        uisketch_labels_csv_path: str,
        output_path: str,
        number_of_versions: int = 1,
    ):
        does_annotation_file_exist = os.path.exists(annotation_csv_path) or annotation_csv_path[:-4] != ".csv"
        does_label_file_exist = os.path.exists(uisketch_labels_csv_path) or uisketch_labels_csv_path[:-4] != ".csv"

        if not does_annotation_file_exist:
            print("Annotations file is either not found or not a CSV file")
            sys.exit(0)

        if not does_label_file_exist:
            print("UISketch labels file is either not found or not a CSV file")
            sys.exit(0)

        self.annotation_csv_path = annotation_csv_path
        self.uisketch_labels_csv_path = uisketch_labels_csv_path
        self.output_path = Path(output_path)
        self.uisketch_path = Path(uisketch_labels_csv_path).parent

        self.number_of_versions = number_of_versions

        self.uisketch_df = pd.read_csv(self.uisketch_labels_csv_path, index_col=0)

        self.generated_train_csv_paths = None
        self.generated_valid_csv_paths = None
        self.generated_test_csv_paths = None

    def _load_train_valid_test_dfs(self):

        if self.generated_train_csv_paths is None:
            self.generated_train_csv_paths = self.output_path.glob("**/*_train_*.csv")

        if self.generated_valid_csv_paths is None:
            self.generated_valid_csv_paths = self.output_path.glob("**/*_valid_*.csv")

        if self.generated_test_csv_paths is None:
            self.generated_test_csv_paths = self.output_path.glob("**/*_test_*.csv")

        self.train_df = pd.DataFrame()
        self.valid_df = pd.DataFrame()
        self.test_df = pd.DataFrame()

        for csv_path in self.generated_train_csv_paths:
            df = pd.read_csv(csv_path)
            self.train_df = pd.concat([self.train_df, df], ignore_index=True)

        for csv_path in self.generated_valid_csv_paths:
            df = pd.read_csv(csv_path)
            self.valid_df = pd.concat([self.valid_df, df], ignore_index=True)

        for csv_path in self.generated_test_csv_paths:
            df = pd.read_csv(csv_path)
            self.test_df = pd.concat([self.test_df, df], ignore_index=True)

    def generate_df(self):
        print("#" * 80)
        print("*** Generating SynZ annotation DataFrames ***")

        generated_train_csv_paths = []
        generated_valid_csv_paths = []
        generated_test_csv_paths = []

        for i in range(self.number_of_versions):
            print("#" * 80)
            print(f"Version {i}")
            train_csv_path, valid_csv_path, test_csv_path = generate_synz_df.generate_synz(
                annotation_csv_path=self.annotation_csv_path,
                uisketch_labels_csv_path=self.uisketch_labels_csv_path,
                output_path=self.output_path,
                version=i,
            )

            generated_train_csv_paths.append(train_csv_path)
            generated_valid_csv_paths.append(valid_csv_path)
            generated_test_csv_paths.append(test_csv_path)
            print("#" * 80)
            print("\n")

        self.generated_train_csv_paths = generated_train_csv_paths
        self.generated_valid_csv_paths = generated_valid_csv_paths
        self.generated_test_csv_paths = generated_test_csv_paths

    def generate_coco(self):
        self._load_train_valid_test_dfs()

        print("#" * 80)
        print("*** Generating images and coco annotations ***")

        generate_synz_coco.uisketch_df = self.uisketch_df

        generate_synz_coco.df = self.train_df
        generate_train_synz_coco = generate_synz_coco.GenerateSynZCoco(
            uisketch_labels_csv_path=self.uisketch_labels_csv_path,
            output_path=self.output_path,
            train_test_valid="train",
        )

        train_coco_csv_path = generate_train_synz_coco.generate()
        print(f"*** Generated COCO training annotations at {train_coco_csv_path} ***")

        generate_synz_coco.df = self.valid_df
        generate_valid_synz_coco = generate_synz_coco.GenerateSynZCoco(
            uisketch_labels_csv_path=self.uisketch_labels_csv_path,
            output_path=self.output_path,
            train_test_valid="val",
        )
        valid_coco_csv_path = generate_valid_synz_coco.generate()
        print(f"*** Generated COCO valid annotations at {valid_coco_csv_path} ***")

        generate_synz_coco.df = self.test_df
        generate_test_synz_coco = generate_synz_coco.GenerateSynZCoco(
            uisketch_labels_csv_path=self.uisketch_labels_csv_path,
            output_path=self.output_path,
            train_test_valid="test",
        )
        test_coco_csv_path = generate_test_synz_coco.generate()

        print(f"*** Generated COCO test annotations at {test_coco_csv_path} ***")
        print(f"*** Generated images at {self.output_path/'images'} ***")

    def run(self):
        self.generate_df()
        self.generate_coco()


if __name__ == "__main__":
    fire.Fire(Pipeline)
