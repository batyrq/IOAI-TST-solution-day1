Here's a README for your IOAI TST in Kazakhstan solution, based on the provided code:

-----

# Image Restoration for IOAI TST in Kazakhstan

This repository contains the solution for the Image Restoration task presented at the IOAI TST in Kazakhstan. The goal is to restore original images from their filtered versions, where a specific 2x2 pixel-wise filter has been applied. The solution utilizes a custom U-Net architecture with residual blocks and Squeeze-and-Excitation (SE) modules, and incorporates information about the applied filter directly into the network.

## Table of Contents

  - [Problem Description](https://www.google.com/search?q=%23problem-description)
  - [Solution Overview](https://www.google.com/search?q=%23solution-overview)
  - [Dataset](https://www.google.com/search?q=%23dataset)
  - [Model Architecture](https://www.google.com/search?q=%23model-architecture)
  - [Loss Function](https://www.google.com/search?q=%23loss-function)
  - [Training](https://www.google.com/search?q=%23training)
  - [Inference](https://www.google.com/search?q=%23inference)
  - [Results](https://www.google.com/search?q=%23results)
  - [Dependencies](https://www.google.com/search?q=%23dependencies)
  - [Usage](https://www.google.com/search?q=%23usage)

## Problem Description

The challenge involves restoring images that have been corrupted by one of several predefined 2x2 pixel-wise filters. Each filter selectively retains a specific color channel (Red, Green, or Blue) for each of the four pixels within a 2x2 block. The task is to predict the original, unfiltered image.

## Solution Overview

The solution consists of the following key components:

1.  **Custom 2x2 Filter Application (`apply_fast_filter`):** A function to simulate the filtering process based on a given pattern.
2.  **Filter Detection (`detect_filter`):** A method to identify which of the 12 possible filters was applied to a given image. This is crucial for guiding the restoration process.
3.  **Custom Dataset (`FilteredRestoreDataset`):** A PyTorch `Dataset` that generates pairs of filtered and original images for training, along with a one-hot encoded representation of the applied filter.
4.  **Custom U-Net Model (`CustomUNet`):** A U-Net based convolutional neural network designed for image-to-image translation tasks. This model is augmented to take the filter information as an additional input, allowing it to adapt its restoration based on the specific degradation.
5.  **Combined Loss Function (`CombinedLossNoPretrained`):** A custom loss function that combines Mean Squared Error (MSE) and Structural Similarity Index Measure (SSIM) to optimize for both pixel-wise accuracy and perceptual quality.
6.  **Training Loop:** Standard PyTorch training loop for optimizing the model.
7.  **Inference Pipeline:** A process for loading filtered test images, detecting the applied filter, and using the trained model to restore them.
8.  **Submission Generation:** Formatting the restored images into the required submission format.

## Dataset

The dataset consists of `real_images` (original images) and `filtered_images` (images processed by one of the 12 filters). The `FilteredRestoreDataset` dynamically applies each of the 12 filters to the original images to create a diverse training set.

**Dataset Structure:**

```
/kaggle/input/tst-day-1/
├── train/
│   └── train/
│       └── real_images/
│           ├── img_0001.png
│           ├── ...
│       └── filtered_images/ (Not directly used in training, filters are applied on the fly)
├── test/
│   └── test/
│       └── filtered_images/
│           ├── img_test_0001.png
│           ├── ...
└── sample_submission.csv
```

## Model Architecture

The core of the solution is a `CustomUNet` model.

  * **Encoder-Decoder Structure:** Follows the typical U-Net architecture with downsampling (encoder) and upsampling (decoder) paths, connected by skip connections.
  * **Residual Blocks:** Each convolutional block in the encoder and decoder is a `ResidualBlock` which helps in training deeper networks by alleviating the vanishing gradient problem.
  * **Squeeze-and-Excitation (SE) Blocks:** Each `ResidualBlock` incorporates an `SEBlock`, which adaptively recalibrates channel-wise feature responses by explicitly modeling interdependencies between channels.
  * **Filter Integration:** A crucial aspect is the integration of the filter information. The `filter_id` (representing which of the 12 filters was applied) is one-hot encoded and passed through a linear layer (`filter_fc`). The resulting feature vector is then expanded and concatenated with the feature maps at each skip connection in the decoder, allowing the model to condition its restoration on the specific filter.

## Loss Function

The `CombinedLossNoPretrained` is used for training:

$$L = \alpha \cdot L_{MSE} + \beta \cdot (1 - L_{SSIM})$$

Where:

  * $L\_{MSE}$ is the Mean Squared Error between the predicted and original images.
  * $L\_{SSIM}$ is the Structural Similarity Index Measure.
  * $\\alpha$ and $\\beta$ are hyperparameters (weights) for MSE and SSIM components, respectively (set to 0.7 and 0.3 in this solution).

This combined loss encourages both pixel-accurate restoration and perceptually pleasing results.

## Training

The model is trained using the Adam optimizer with a learning rate of $1e-3$ for 20 epochs. The dataset is split into training and validation sets based on original image indices to ensure that a filtered version of an image does not appear in both train and validation sets.

**Training Details:**

  * **Epochs:** 20
  * **Optimizer:** Adam
  * **Learning Rate:** $1e-3$
  * **Batch Size:** 32
  * **Image Size:** Resized to $128 \\times 128$ for training and inference.

## Inference

The inference process involves:

1.  Loading a filtered image from the test set.
2.  Converting the image to a NumPy array to facilitate filter detection.
3.  **Detecting the filter:** The `detect_filter` function attempts to identify the specific 2x2 filter applied to the image by applying each known pattern and checking for equality with the input.
4.  Converting the filtered image to a PyTorch tensor and creating a one-hot encoded vector for the detected filter.
5.  Passing the filtered image tensor and the one-hot filter vector to the trained `CustomUNet` model.
6.  The model outputs the restored image.
7.  The restored image is then converted back to a PIL Image and saved.
8.  Finally, the pixel values of the restored image are flattened and inserted into the `sample_submission.csv` template.
9.  A final step swaps the first and last $128 \\times 128$ pixel blocks in the flattened output, which was a specific requirement for the competition's submission format.

## Results

The solution achieved a PSNR (Peak Signal-to-Noise Ratio) score of **24.3** on the evaluation metric.

## Dependencies

The following Python libraries are required:

  * `os`
  * `cv2` (OpenCV Python)
  * `numpy`
  * `pandas`
  * `pathlib`
  * `PIL` (Pillow)
  * `matplotlib`
  * `torch`
  * `torchvision`
  * `scikit-learn` (for `train_test_split`)
  * `torchmetrics` (for SSIM calculation)

These can be installed via pip:

```bash
pip install opencv-python numpy pandas Pillow matplotlib torch torchvision scikit-learn torchmetrics
```

## Usage

1.  **Clone the repository (if applicable):**

    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Place the dataset:** Ensure your dataset is structured as described in the [Dataset](https://www.google.com/search?q=%23dataset) section and the paths in the code (`/kaggle/input/tst-day-1/train/train/real_images` and `/kaggle/input/tst-day-1/test/test/filtered_images`) are updated to your local paths if running outside a Kaggle environment.

3.  **Run the script:**

    ```bash
    python your_solution_script.py
    ```

    This will train the model and generate `submission.csv` (and `submission_swapped.csv`) in the current directory.

-----
