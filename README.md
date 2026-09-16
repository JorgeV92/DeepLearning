# Deep Learning 

A list of programs in python for machine learning and deep learning.

Run both linear classifiers on the full CIFAR-10 training set:

```bash
conda activate dl
python lc.py
```

This trains separate SVM and softmax models for 10 epochs on all 50,000 training
images, then prints accuracy and sample predictions on the separate 10,000-image
test set. Every training image is used once per epoch. Data is downloaded to the
ignored `data/` directory if needed.

To customize training and access all test predictions:

```python
from lc import train_full_cifar10

results = train_full_cifar10(num_epochs=10, batch_size=256)
svm_predictions = results["svm"]["predictions"]
softmax_predictions = results["softmax"]["predictions"]
```

Each result also contains `model`, `train_accuracy`, and `test_accuracy`.
The individual `train_svm_cifar10()` and `train_softmax_cifar10()` functions
still use the default 49,000/1,000 training/validation split.
