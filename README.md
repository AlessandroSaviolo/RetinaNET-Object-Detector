# RetinaNET Object Detector for SVHN Dataset

This project is part of a series of projects for the course _Selected Topics in Visual Recognition using Deep Learning_ that I attended during my exchange program at National Chiao Tung University (Taiwan). See `task.pdf` for the details of the assignment. See `report.pdf` for the report containing the representation and the analysis of the produced results.

The purpose of this project is to implement an object detector for the [SVHN Dataset](http://ufldl.stanford.edu/housenumbers/). The implementation consists of a RetinaNET object detector which uses the keras-retinanet package. Due to the lack of computational power, the presented model is trained by first loading a pre-trained model and then training it for few epochs. Moreover, due to this problem, ResNet-50 model is used as feature extractor. Since the evaluation metric is mAP, the hyper parameters of the model are tuned in order to obtain the highest mAP as possible.

## 1. Dataset

- [SVHN Training set](http://ufldl.stanford.edu/housenumbers/train.tar.gz)

- [SVHN Test set](http://ufldl.stanford.edu/housenumbers/test.tar.gz)

## 2. Project Structure

- `load_data.py` : load the dataset and create the Pascal VOC annotations

- `train.py` : train a new model from scratch, or load the [model](https://drive.google.com/open?id=1-MqGpht6UnGzX3Ps_8-IIJ8hAz24pHoN) and keep training it

- `infer.py` : make predictions and output a JSON file containing the generated bounding box coordinates, labels and scores

- `RetinaNET.ipynb` : test inference speed of the model

## 3. Credits

The following GitHub Repositories have helped the development of this project:

- [Pavitrakumar78 Repository](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN): the python file "construct datasets.py" has been used to parse the input file containing the dataset and to create the annotations

- [Fizyr Repository](https://github.com/fizyr/keras-retinanet): the entire repository has been imported and widely used to create and train the RetinaNET model

- [Penny4860 Repository](https://github.com/penny4860/retinanet-digit-detector): the pre-trained model has been taken from this repository

## 4. License

Copyright (C) 2021 Alessandro Saviolo, [FlexSight SRL](http://www.flexsight.eu/), Padova, Italy
```
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
```
