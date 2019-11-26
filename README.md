# RetinaNET Object Detector

Implementation of RetinaNET object detector using keras-retinanet package and SVHN dataset. Due to lack of computational power, the presented model is trained by first loading a pre-trained model and then training it for few epochs. Moreover, due to this problem,
ResNet-50 model is used as feature extractor. Since the evaluation metric is mAP, the hyper parameters of the model are tuned in order to obtain the highest mAP as possible.

### Project files

The dataset provided for this homework is stored in a .mat file. At first, I used a pandas dataframe to extract and store the features of each image (i.e., height, image name, label, left, top, width, cut image, image, image height, image width). Then, I created an annotation file for each image using the Pascal Voc format. This annotation format standardises image data sets for object class recognition and offers many advantages, such as it provides a common set of tools for accessing the data sets and annotations.

Load the dataset and create the Pascal VOC annotations through the following command:
- project/root> python load_data.py

### Credits

The following GitHub Repositories have helped the development of this project:
- Pavitrakumar78 Repository, the python file "construct datasets.py" has been used to parse the input file containing the dataset and to create the annotations
- Fizyr Repository, the entire repository has been imported and widely used to create and train the RetinaNET model
- Penny4860 Repository, the pre-trained model has been taken from this repository
