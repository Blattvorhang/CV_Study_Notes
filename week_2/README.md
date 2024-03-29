# FCN and UNet
## Project Structure
- [VOCLoader.py](./VOCLoader.py): Used to load the Pascal VOC 2012 dataset. It includes functions for data preprocessing and data augmentation.
- [VOC_train_template.ipynb](./VOC_train_template.ipynb): A template for training and predicting with FCN on the Pascal VOC 2012 dataset.
- [FCN.ipynb](./FCN.ipynb): FCN I have implemented.
- [FCN-d2l.ipynb](./FCN-d2l.ipynb): FCN copied from [Dive into Deep Learning](https://d2l.ai/chapter_computer-vision/fcn.html)
- [UNet.ipynb](./UNet.ipynb): UNet

## FCN
I have implemented a package [VOCLoader.py](./VOCLoader.py) to load the dataset Pascal VOC 2012.

However, the FCN trained on it is not satisfactory enough probably due to my inappropriate processing of the dataset.

Therefore, I highly recommend `d2l` package to load the dataset and train neural network using its high-level APIs. With the help of this package, the only thing you need to take care of is the design of networks, which is also more important for learning and understanding deep learning.

Moreover, I have written a [template](./VOC_train_template.ipynb) for network training and prediction with respect to semantic segmentation tasks on the Pascal VOC 2012 dataset. This template is modified based on the teaching materials from [Fully Convolutional Networks](https://d2l.ai/chapter_computer-vision/fcn.html).

## UNet
This net is initially designed for binary classification, for twenty-one-class classification on the Pascal VOC dataset, I modified the corresponding number of channels of UNet.