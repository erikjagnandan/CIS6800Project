# SmoothDINOv2

SmoothDINOv2 is a method for reducing the error in the metric depth estimates produced by DINOv2 on video stream data. This repository provides the code and tools to train and evaluate SmoothDINOv2, along with pre-trained models and example visualizations.

---

## Features
- **Error reduction for DINOv2 depth estimates**: SmoothDINOv2 uses a CNN-based adapter model placed between the DINO encoder and depth decoder to combine DINO features from consecutive frames and improve depth estimation accuracy.
- **Train and evaluate models**: Includes scripts for training and validation.
- **Pre-trained models**: Includes checkpoints for regularized and unregularized CNN models.
- **Data handling utilities**: Contains files specifying our train-validation split and a script for creating custom train-validation splits.

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
Run `train_val_split.py` to create a valid train-validation split on the provided subset:
   ```bash
   python train_val_split.py
   ```

If you run `main.py` with the provided subset without first running `train_val_split.py`, you will get an error similar to `FileNotFoundError: [Errno 2] No such file or directory: '/home/SmoothDINOv2/datasets/nyu_data/data/nyu2_train/<dataset_dir>/<image_index>.png'` because the training code is trying to retrieve an image from the full dataset using our original train-validation split, as opposed to a valid train-validation split for the small subset provided here.


### Using the full dataset
Place the full dataset in the `datasets/nyu_data/data` directory. Do not run `train_val_split.py`, as the provided `train_list.json` and `val_list.json` files correspond to the original train-validation split used in all experiments.

---

## Usage

### Running the Code
To train or evaluate the model, run the following command:

```bash
python main.py [arguments]
```

### Command Line Arguments

| Argument               | Description                                       | Default Value          | Possible Values                          |
|------------------------|---------------------------------------------------|------------------------|------------------------------------------|
| `--model_string`       | Model type to use.                                | `cnn`                  | `cnn`, `cnn_regularized`                |
| `--batch_size`         | Number of samples per batch.                      | `10`                   | Any positive integer                     |
| `--batches_per_backprop` | Number of batches before backpropagation.        | `1`                    | Any positive integer                     |
| `--train`              | Whether to run training (`True`) or validation (`False`). | `False`          | `True`, `False`                          |
| `--load_model`         | Whether to load a saved model (`True`) or start fresh (`False`). | `False`   | `True`, `False`                          |
| `--epoch_to_load`      | Epoch number to load if loading a model.          | `9`                    | Any non-negative integer                 |
| `--segment_to_load`    | Segment of the epoch to load for validation.      | `None`                 | `None`, or integer in `[1, N]`           |
| `--val_checkpoint_ratios` | Ratios within an epoch to save model and validate. | `[0.25, 0.5, 0.75]` | List of floats in `[0, 1]`               |
| `--backbone_size`      | Backbone size of the model.                       | `small`                | `small`, `base`, `large`, `giant`        |

---

## Pre-trained Models
The repository includes model files for the unregularized and regularized CNN at the end of training (epoch 9). These can be loaded for validation or further training by setting the `--load_model` argument to `True` (no need to specify `--epoch_to_load` and `--segment_to_load`, as their default values load the model at the end of epoch 9).

---

## Jupyter Notebooks

The repository includes two Jupyter notebooks for demonstration and visualization:
1. **NCUT_Visualizations.ipynb**: Generates NCUT visualizations.
2. **SmoothDINOv2_Demo.ipynb**: Visualizes depth maps produced by SmoothDINOv2.

Each notebook contains step-by-step instructions for running it.

---

## Acknowledgments
The dinov2 folder is copied from the DINOv2 GitHub repository, available at [facebookresearch/dinov2.](https://github.com/facebookresearch/dinov2)



Due to memory limitations on GitHub, we cannot upload all of our model files saved throughout the training process for each approach. However, we have included the model files at the end of training (end of epoch 9) for the unregularized and regularized CNN. Also due to memory limitations on GitHub, only a small subset of the NYU Depth v2 dataset is included. The full dataset is available for download from https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2

Note that we perform a custom train-validation split of the data, so if you try to use the small dataset of a few scenes provided here, you should run train_val_split.py so as to get a valid train-validation split on the given data before running main.py. If you run main.py without first running train_val_split.py, you will get an error like: FileNotFoundError: [Errno 2] No such file or directory: '/home/SmoothDINOv2/datasets/nyu_data/data/nyu2_train/<dataset_dir>/<image_index>.png' because the training code is trying to retrieve an image from our original train-validation split, as opposed to one for the small dataset provided here. If you instead download the full dataset and want to evaluate our models on our original train-validation split, you should not run train_val_split.py, as the train_list.json and val_list.json files in the train_val_split directory specify the train-validation split we used for all of our experiments.

To run our code, simply run python main.py

We also include 2 Python Notebooks:

NCUT_Visualizations.ipynb - used to generate our NCUT visualizations

SmoothDINOv2_Demo.ipynb - demo for visualizing the output depth maps produced by our approach

Each notebook contains instructions on how to run it.

Note that the dinov2 folder has been copied from the DINOv2 GitHub repository, available at https://github.com/facebookresearch/dinov2
