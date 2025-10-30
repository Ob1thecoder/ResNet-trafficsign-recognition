Install dependencies:

```pip install torch torchvision matplotlib seaborn scikit-learn matplotlib tqdm pandas pillow numpy```


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
 [100.          99.72222222  99.73333333  96.22222222  99.24242424
  90.31746032  85.33333333  87.55555556  98.88888889  98.95833333
  92.12121212  86.9047619   98.84057971  99.72222222  98.51851852
 100.         100.          93.05555556  87.94871795  76.66666667
 100.          66.66666667  91.66666667  96.          94.44444444
  97.29166667 100.         100.          99.33333333 100.
  62.66666667 100.         100.          99.52380952 100.
  88.20512821  98.33333333  98.33333333  97.24637681  66.66666667
  94.44444444  93.33333333  96.66666667]
Testing finished, accuracy: 0.951```