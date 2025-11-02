# Traffic Sign Classification Using ResNet-18 on the GTSRB

## Requirements
Install dependencies:

```pip install -r requirements.txt```
## Quickstart
- If you already trained your model with checkpoints saved in ```checkpoint/``` directory, run ```python3 run_eval.py```

- Else if it is empty,You may train your model, run ```python3 run_train.py```
## Datasets

The dataset used for model training is the German
Traffic Sign Recognition Benchmark (GTSRB)

Dataset Composition

- Total Images: Over 50,000 color images.

- Number of Classes: 43 traffic sign categories (e.g., speed limits, prohibitions, warnings).

- Image Format: RGB, variable resolutions.

- Image Size (for training): Resized to 32×32 pixels for consistency with ResNet input requirements.

- Color Space: Normalized RGB.

- Dataset Split:

  - Training Set: 80% of the images.

  - Testing Set: 20% of the images.

## ResNet model

ResNet-18 Structure:

- Input Layer: 3×48×48 RGB images
- Convolutional Stem: 7×7 conv, BatchNorm, ReLU, MaxPool
- Residual Blocks: 4 stages with [2,2,2,2] blocks each
- Global Average Pooling: Reduces spatial dimensions
- Classification Head: Modified from 1000 to 43 classes for GTSRB



## Evaluation Logs
First evaluation, no prior trainning:
```
Each class accuracy:
 [  0.           0.           0.           0.           0.
   0.           0.           0.           0.           0.
   0.           0.           0.           0.           0.
   0.           0.           0.           0.           0.
   0.           0.           0.           0.66666667   0.
   0.           0.           0.           0.           0.
 100.           0.           0.           0.           9.16666667
   0.           0.           0.           0.           0.
   0.           0.           0.        ]
Testing finished, accuracy: 0.013
```

Accuracy of ResNet model is 1.3% which low but expected


After train for 50 epochs:
Using device: cuda
```Each class accuracy:
 [ 96.66666667  98.75        98.53333333  98.88888889  99.24242424
  94.28571429  89.33333333  86.66666667  87.77777778  98.33333333
  97.87878788  94.52380952  97.68115942  99.72222222  98.88888889
  99.52380952 100.          92.22222222  95.64102564 100.
 100.          66.66666667  96.66666667  82.         100.
  95.625       98.88888889  86.66666667  99.33333333 100.
  62.66666667  98.51851852 100.         100.         100.
  94.61538462  92.5         98.33333333  99.42028986  71.11111111
  94.44444444  98.33333333 100.        ]
Testing finished, accuracy: 0.958```

Accuracy of ResNet model is 95.8% which low but expected


