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
ResNet v1:
[Deep Residual Learning for Image Recognition
](https://arxiv.org/pdf/1512.03385.pdf)
ResNet v2:
[Identity Mappings in Deep Residual Networks
](https://arxiv.org/pdf/1603.05027.pdf)
Model|n|200-epoch accuracy|Original paper accuracy |sec/epoch GTX1080Ti
:------------|--:|-------:|-----------------------:|---:
ResNet20   v1|  3| 92.16 %|                 91.25 %|35
ResNet32   v1|  5| 92.46 %|                 92.49 %|50
ResNet44   v1|  7| 92.50 %|                 92.83 %|70
ResNet56   v1|  9| 92.71 %|                 93.03 %|90
ResNet110  v1| 18| 92.65 %|            93.39+-.16 %|165
ResNet164  v1| 27|     - %|                 94.07 %|  -
ResNet1001 v1|N/A|     - %|                 92.39 %|  -
&nbsp;
Model|n|200-epoch accuracy|Original paper accuracy |sec/epoch GTX1080Ti
:------------|--:|-------:|-----------------------:|---:
ResNet20   v2|  2|     - %|                     - %|---
ResNet32   v2|N/A| NA    %|            NA         %| NA
ResNet44   v2|N/A| NA    %|            NA         %| NA
ResNet56   v2|  6| 93.01 %|            NA         %|100
ResNet110  v2| 12| 93.15 %|            93.63      %|180
ResNet164  v2| 18|     - %|            94.54      %|  -
ResNet1001 v2|111|     - %|            95.08+-.14 %|  -
