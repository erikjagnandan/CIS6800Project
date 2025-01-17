# SmoothDINOv2

SmoothDINOv2 is a method for reducing the error in the metric depth estimates produced by DINOv2 on video stream data. This repository provides the code and tools to train and evaluate SmoothDINOv2, along with pre-trained models and example visualizations.

---

## Features
- **Error reduction for DINOv2 depth estimates**: SmoothDINOv2 uses regularized CNN models to improve depth estimation accuracy.
- **Train and evaluate models**: Includes scripts for training, validation, and visualization.
- **Pre-trained models**: Includes checkpoints for regularized and unregularized CNN models.
- **Data handling utilities**: Tools for creating train-validation splits and working with the NYU Depth v2 dataset.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/erikjagnandan/SmoothDINOv2.git
   cd SmoothDINOv2
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

---

## Dataset

Due to memory limitations on GitHub, only a small subset of the NYU Depth v2 dataset is included. The full dataset is available for download from [Kaggle](https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2).

---

## Data Preparation

### Using the provided subset
Run `train_val_split.py` to create a valid train-validation split:
   ```bash
   python train_val_split.py
   ```

### Using the full dataset
Place the full dataset in the datasets/nyu_data/data directory. Do not run train_val_split.py, as the provided train_list.json and val_list.json files correspond to the original train-validation split used in all experiments.

---

## Usage

### Running the Code
To train or evaluate the model, run the following command:

```bash
python main.py [arguments]
```

Due to memory limitations on GitHub, we cannot upload all of our model files saved throughout the training process for each approach. However, we have included the model files at the end of training (end of epoch 9) for the unregularized and regularized CNN. Also due to memory limitations on GitHub, only a small subset of the NYU Depth v2 dataset is included. The full dataset is available for download from https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2

Note that we perform a custom train-validation split of the data, so if you try to use the small dataset of a few scenes provided here, you should run train_val_split.py so as to get a valid train-validation split on the given data before running main.py. If you run main.py without first running train_val_split.py, you will get an error like: FileNotFoundError: [Errno 2] No such file or directory: '/home/SmoothDINOv2/datasets/nyu_data/data/nyu2_train/<dataset_dir>/<image_index>.png' because the training code is trying to retrieve an image from our original train-validation split, as opposed to one for the small dataset provided here. If you instead download the full dataset and want to evaluate our models on our original train-validation split, you should not run train_val_split.py, as the train_list.json and val_list.json files in the train_val_split directory specify the train-validation split we used for all of our experiments.

To run our code, simply run python main.py

We also include 2 Python Notebooks:

NCUT_Visualizations.ipynb - used to generate our NCUT visualizations

SmoothDINOv2_Demo.ipynb - demo for visualizing the output depth maps produced by our approach

Each notebook contains instructions on how to run it.

Note that the dinov2 folder has been copied from the DINOv2 GitHub repository, available at https://github.com/facebookresearch/dinov2
