# SmoothDINOv2

SmoothDINOv2 is a method for reducing the error in the metric depth estimates produced by DINOv2 on video stream data. SmoothDINOv2 utilizes a zero-initialized, two layer Convolutional Neural Network (CNN) placed between the DINO encoder and the downstream depth decoder to combine the DINO features (produced by the DINO encoder) from consecutive video frames. This CNN receives the DINO features from the previous and current frames as input, along with the horizontal and vertical pixel shift between the two frames, as computed by phase correlation, to make the task of aligning and subsequently combining the features easier. The resulting set of DINO features are input to the downstream depth decoder, which produces the output depth map. We find that our SmoothDINOv2, as applied to DINOv2 with the ”small” encoder size, achieves a 23.8 percent reduction in MSE on the NYUv2 Depth dataset, resulting in a model which is both faster and more accurate than DINOv2 with the ”base” encoder size. Additionally, through visualizations of the DINO features both before and after our two-layer CNN, as well as comparisons of the output depth maps with and without applying our approach, we observe that SmoothDINOv2 is not altering the fundamental structure of the DINO features or of the output depth map. However, we do observe that with SmoothDINOv2, the output depth map is more accurately scaled to the ground truth. We believe that this occurs because the CNN-based adapter effectively smooths over the errors in depth estimation made across consecutive frames. We attribute the observed reductions in MSE primarily to this improved scaling

This repository provides the code and tools to train and evaluate SmoothDINOv2, along with pre-trained models and example visualizations. A conference paper detailing our findings is currently in progress.

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
3. Clone the DINOv2 repository and copy the `dinov2` folder into your cloned SmoothDINOv2 repository:
   ```bash
   git clone https://github.com/facebookresearch/dinov2.git
   cp -r dinov2/dinov2 ./SmoothDINOv2/

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
1. `NCUT_Visualizations.ipynb`: Generates NCUT visualizations.
2. `SmoothDINOv2_Demo.ipynb`: Visualizes depth maps produced by SmoothDINOv2.

Each notebook contains step-by-step instructions for running it.
