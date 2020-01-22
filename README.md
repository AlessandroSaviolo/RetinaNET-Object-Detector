# RetinaNET Object Detector for SVHN Dataset

This project is part of a series of projects for the course _Selected Topics in Visual Recognition using Deep Learning_ that I attended during my exchange program at National Chiao Tung University (Taiwan). See `task.pdf` for the details of the assignment. See `report.pdf` for the report containing the representation and the analysis of the produced results.

The purpose of this project is to implement an object detector for the [SVHN Dataset](http://ufldl.stanford.edu/housenumbers/). The implementation consists of a RetinaNET object detector which uses the keras-retinanet package. Due to the lack of computational power, the presented model is trained by first loading a pre-trained model and then training it for few epochs. Moreover, due to this problem, ResNet-50 model is used as feature extractor. Since the evaluation metric is mAP, the hyper parameters of the model are tuned in order to obtain the highest mAP as possible.

## 1. Dataset

- [SVHN Training set](https://drive.google.com/open?id=1Yu290bIuW-n3v3FM7ziYTUj0wIKSJFkh)

- [SVHN Test set](https://drive.google.com/open?id=1STaRswNKSkvyzlUtFlteDHBan2tlydK8)

## 2. Project Structure

- `load_data.py` : load the dataset and create the Pascal VOC annotations

- `train.py` : train a new model from scratch, or load the [model](https://drive.google.com/open?id=1a1sfy6x5UcCNcg8xYS7zgdrr-H3xrDve) and keep training it

- `infer.py` : make predictions and output a JSON file containing the generated bounding box coordinates, labels and scores

- `RetinaNET.ipynb` : test inference speed of the model

## 3. Credits

The following GitHub Repositories have helped the development of this project:

- [Pavitrakumar78 Repository](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN): the python file "construct datasets.py" has been used to parse the input file containing the dataset and to create the annotations

- [Fizyr Repository](https://github.com/fizyr/keras-retinanet): the entire repository has been imported and widely used to create and train the RetinaNET model

- [Penny4860 Repository](https://github.com/penny4860/retinanet-digit-detector): the pre-trained model has been taken from this repository
