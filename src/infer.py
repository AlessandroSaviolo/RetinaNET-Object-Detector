"""
Created on Tue 27 Mar 2018 by PavitrakumarPC
Modified on Wed 17 Nov 2019 by AlessandroSaviolo
"""

from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image
from retina.utils import decode_output
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import pandas as pd
import time
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def load_inference_model(model_path=os.path.join('snapshots', 'resnet.h5')):
    model = models.load_model(model_path, backbone_name='resnet50')
    model = models.convert_model(model)
    # model.summary()
    return model


def post_process(boxes, original_img, preprocessed_img):
    # post-processing
    h, w, _ = preprocessed_img.shape
    h2, w2, _ = original_img.shape
    boxes[:, :, 0] = boxes[:, :, 0] / w * w2
    boxes[:, :, 2] = boxes[:, :, 2] / w * w2
    boxes[:, :, 1] = boxes[:, :, 1] / h * h2
    boxes[:, :, 3] = boxes[:, :, 3] / h * h2
    return boxes


if __name__ == '__main__':

    # load model
    print('Loading model')
    model = load_inference_model('snapshots/resnet152_pascal_03.h5')

    # load test images
    print('Loading test images')
    num_images = len([img for img in os.listdir('test') if img.endswith('.png')])
    images = [cv2.imread('test/{0}.png'.format(i)) for i in range(1, num_images + 1)]

    # store resulting predictions
    predictions = []

    i = 1
    for image in images:
        print('Predicting {0}.png'.format(i))

        # copy to draw on
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # preprocess image for network
        image = preprocess_image(image)
        image, _ = resize_image(image, 416, 448)

        # process image
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

        boxes = post_process(boxes, draw, image)
        labels = labels[0]
        scores = scores[0]
        boxes = boxes[0]

        # compute boxes and return image prediction
        image_prediction = decode_output(
            draw,
            boxes,
            labels,
            scores,
            class_labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
            min_score_thresh=.1
        )
        # print(image_prediction)
        predictions.append(image_prediction)

        # plot
        # plt.imshow(draw)
        # plt.axis('off')
        # plt.show()
        # plt.savefig('{0}.png'.format(i))

        i += 1

    # save resulting predictions in JSON format
    print('Creating JSON file')
    with open('0845086_4.json', 'w') as f:
        f.write(pd.Series(predictions).to_json(orient='values'))
