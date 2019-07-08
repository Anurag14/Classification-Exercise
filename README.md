# Classification-Exercise

Given a labelled dataset of images and their corresponding labels, implement a classification network (CNN based, any existing CNN architecture can be used) which takes new images and classifies them into 10 classes 
  a. What changes are required to make the network eligbile for classifying images from another dataset which has 20 classes?
  b. What changes are required to make the network work for the data, which has mulltiple labels instead of one?

Custom losses:
  a. MSE loss (x-y)^2
  b. (x-y)^3
  c. Cosine similarity loss (1 - (x^Ty)/(||X||.||y||)) where x and y are two vectors and ||x|| and ||y|| are the norm values and ^T denotes the transpose operations
  d. (x/||x|| - y/||y||)^3
