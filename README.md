# RetinaNET Object Detector

Implementation of RetinaNET object detector using keras-retinanet package and SVHN dataset.

### 1. Loading Dataset and Creating Annotations

The dataset provided for this homework is stored in a .mat file. At first, I used a pandas dataframe to extract and store the features of each image (i.e., height, image name, label, left, top, width, cut image, image, image height, image width). Then, I created an annotation file for each image using the Pascal Voc format. This annotation format standardises image data sets for object class recognition and offers many advantages, such as it provides a common set of tools for accessing the data sets and annotations.

Load the dataset and create the Pascal VOC annotations through the following command:

<p align="center">
  project/root> python load_data.py
</p>
