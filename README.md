# RetinaNET Object Detector

Implementation of RetinaNET object detector using keras-retinanet package and SVHN dataset. Due to lack of computational power, the presented model is trained by first loading a pre-trained model and then training it for few epochs. Moreover, due to this problem,
ResNet-50 model is used as feature extractor. Since the evaluation metric is mAP, the hyper parameters of the model are tuned in order to obtain the highest mAP as possible.

### Project files

- project/root> python load_data.py
Load the dataset and create the Pascal VOC annotations.

- project/root> python train.py
Train a new model from scratch, or load the model from "project-root/snapshots" folder and keep training it.

- project/root> python infer.py
Make predictions and output a JSON file containing the generated bounding box coordinates, labels and scores.

### Credits

The following GitHub Repositories have helped the development of this project:
- Pavitrakumar78 Repository, the python file "construct datasets.py" has been used to parse the input file containing the dataset and to create the annotations
- Fizyr Repository, the entire repository has been imported and widely used to create and train the RetinaNET model
- Penny4860 Repository, the pre-trained model has been taken from this repository
