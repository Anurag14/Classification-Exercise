# Classification-Exercise

* Given a labelled dataset of images and their corresponding labels, implement a classification network (CNN based, any existing CNN architecture can be used) which takes new images and classifies them into 10 classes 
  1. What changes are required to make the network eligbile for classifying images from another dataset which has 20 classes?
  2. What changes are required to make the network work for the data, which has mulltiple labels instead of one?

* Custom losses:
  1. MSE loss (x-y)^2
  2. (x-y)^3
  3. Cosine similarity loss (1 - (x^Ty)/(||X||.||y||)) where x and y are two vectors and ||x|| and ||y|| are the norm values and ^T denotes the transpose operations
  4. (x/||x|| - y/||y||)^3


References:
[1](https://github.com/kuangliu/pytorch-cifar) [2](https://modelzoo.co/model/cifar-10-on-pytorch-with-vgg-resnet) [3](https://github.com/keras-team/keras/issues/10371) 
