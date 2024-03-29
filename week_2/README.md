# FCN
I have implemented a package [VOCLoader.py](./VOCLoader.py) to load the dataset Pascal VOC 2012.

However, the FCN trained on it is not satisfactory enough probably due to my inappropriate processing of the dataset.

Therefore, I highly recommend `d2l` package to load the dataset and train neural network using its high-level APIs. With the help of this package, the only thing you need to take care of is the design of networks, which is also more important for learning and understanding deep learning.

Moreover, I have written a [template](./VOC_train_template.ipynb) for network training and prediction with respect to semantic segmentation tasks on the Pascal VOC 2012 dataset. This template is modified based on the teaching materials from [Fully Convolutional Networks](https://d2l.ai/chapter_computer-vision/fcn.html).