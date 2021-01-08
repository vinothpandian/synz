<h1 align="center">SynZ: Enhanced Synthetic Dataset for Training UI Element Detector From Lo-Fi Sketches</h1>
<p align="center">
  <img alt="Version" src="https://img.shields.io/badge/version-1.0.0-blue.svg?cacheSeconds=2592000" />
  <a href="https://creativecommons.org/licenses/by-sa/4.0/" target="_blank">
    <img alt="License: CC BY-SA 4.0" src="https://img.shields.io/badge/License-CC_BY_SA_4.0-yellow.svg" />
  </a>
  <br/>
  <a href="#" target="_blank">
    <img alt="Python: 3.8.6" src="https://img.shields.io/badge/Python-3.8.6-important" />
  </a>
  <a href="#" target="_blank">
    <img alt="Dependency: PyTorch 1.7.1" src="https://img.shields.io/badge/PyTorch-1.7.1-important" />
  </a>
  <a href="#" target="_blank">
    <img alt="Dependency: Detectron2" src="https://img.shields.io/badge/Detectron-2-important" />
  </a>
  <br/>
  <br/>
  <span>💾 </span>
  <a href="https://www.kaggle.com/vinothpandian/synz-dataset" target="_blank">
    SynZ Dataset in Kaggle
  </a>
  <span>&nbsp;&nbsp;&nbsp;&nbsp;</span>
  <span>💾 </span>
  <a href="https://www.kaggle.com/vinothpandian/uisketch" target="_blank">
    UISketch Dataset in Kaggle
  </a>
  <br/>
</p>

---

## Script to extract SynZ annotations from RICO

- Follow the steps and run the script from the [`notebooks`](./notebooks) folder
  - Notebook [`RICO_data_extraction.ipynb`](./notebooks/01_RICO_data_extraction.ipynb) converts RICO annotations to a CSV file
  - Notebook [`RICO_to_SynZ_annotations.ipynb`](./notebooks/02_RICO_to_SynZ_annotations.ipynb) converts RICO annotations to SynZ annotations for further steps

---

## Generating SynZ sketches

To generate SynZ sketches from scratch,

- Download all the UISketch dataset to the [`data`](./data) folder

  - UISketch dataset from [kaggle](https://www.kaggle.com/vinothpandian/uisketch)

- Either run the notebooks described in previous section to generate `SynZ_ready_annotations.csv` file or download the SynZ annotations from [here](https://blackbox-toolkit.com/datasets/SynZ_ready_annotations.csv)

- Install the necessary requirements from `requirements.txt` file

  ```sh
  pip install -r requirements.txt
  ```

- Run the following command to generate SynZ dataset

  ```sh
  python generate_synz_dataset.py \
    --annotation_csv_path=./data/SynZ_ready_annotations.csv \
    --uisketch_labels_csv_path=./data/uisketch/labels.csv \
    --output_path=./data/SynZ \
    --number_of_versions=1 \
    run

  ```

---

## Downlading MetaMorph model trained weights

- Download the trained SynZ checkpoint from [here](https://blackbox-toolkit.com/models/synz_checkpoint.tar.gz) to [`models`](./models) folder
- Extract the `synz_checkpoint.tar.gz` folder to acquire the MetaMorph model's trained weights from `model_final.pth`

---

## Authors

👤 **Vinoth Pandian**

- Website: [vinoth.info](https://vinoth.info)
- Github: [@vinothpandian](https://github.com/vinothpandian)
- LinkedIn: [@vinothpandian](https://linkedin.com/in/vinothpandian)

👤 **Sarah Suleri**

- Website: [sarahsuleri.info](https://sarahsuleri.info)
- LinkedIn: [@sarahsuleri](https://linkedin.com/in/sarahsuleri)
