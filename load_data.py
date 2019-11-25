"""
Created on Tue 27 Mar 2018 by PavitrakumarPC
Modified on Wed 13 Nov 2019 by AlessandroSaviolo
"""

import cv2
import os
import pandas as pd
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from random import randint


def images_constructor(img_folder):
    images = []
    for img_file in os.listdir(img_folder):
        if img_file.endswith('.png'):
            images.append([img_file, cv2.imread(os.path.join(img_folder, img_file))])
    img_data = pd.DataFrame([], columns=['image_name', 'image_height', 'image_width', 'image'])
    for img_info in images:
        full_img = img_info[1]
        row_dict = {
            'image_name': [img_info[0]], 
            'image_height': [img_info[1].shape[0]], 
            'image_width': [img_info[1].shape[1]], 
            'image': [full_img]
        }
        img_data = pd.concat([img_data, pd.DataFrame.from_dict(row_dict, orient='columns')], sort=True)
    return img_data


def get_name(index, hdf5_data):
    name = hdf5_data['/digitStruct/name']
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]].value])


def get_bbox(index, hdf5_data):
    attrs = {}
    item = hdf5_data['digitStruct']['bbox'][index].item()
    for key in ['label', 'left', 'top', 'width', 'height']:
        attr = hdf5_data[item][key]
        values = [hdf5_data[attr.value[i].item()].value[0][0] for i in range(len(attr))] if len(attr) > 1 else [attr.value[0][0]]
        attrs[key] = values
    return attrs


def boundingboxes_constructor(mat_file):
    f = h5py.File(mat_file, 'r')
    bbox_df = pd.DataFrame([], columns=['height', 'image_name', 'label', 'left', 'top', 'width'])
    for j in range(f['/digitStruct/bbox'].shape[0]):
        image_name = get_name(j, f)
        row_dict = get_bbox(j, f)
        row_dict['image_name'] = image_name
        bbox_df = pd.concat([bbox_df, pd.DataFrame.from_dict(row_dict, orient='columns')], sort=True)
    return bbox_df


def build_dataset(img_folder, mat_file_name, h5_name):
    boundingboxes = boundingboxes_constructor(os.path.join(img_folder, mat_file_name))
    images = images_constructor(img_folder)
    dataframe = boundingboxes.merge(images, on='image_name', how='left')
    dataframe.to_hdf(os.path.join(img_folder, h5_name), 'table')


def show_images(dataframe, num_images):
    for _ in range(num_images):
        idx = randint(0, len(dataframe.index))
        selected_rows = dataframe.loc[dataframe['image_name'] == dataframe.iloc[idx]['image_name']]
        image = Image.fromarray(selected_rows['image'].iloc[0], 'RGB')
        plt.imshow(image)
        for index, row in selected_rows.iterrows():
            patch = Rectangle(
                (row['left'], row['top']),
                row['width'],
                row['height'],
                linewidth=1,
                edgecolor='lime',
                facecolor='none'
            )
            pred = 0 if row['label'] == 10.0 else int(row['label'])
            plt.text(row['left'], row['top'], pred, fontdict={'color': 'lime'})
            plt.gca().add_patch(patch)
        plt.show()


def to_xml(rows):
    xml = [
        '<annotation>',
        '   <filename>{0}</filename>'.format(rows.iloc[0]['image_name']),
        '   <size>',
        '      <width>{0}</width>'.format(rows.iloc[0]['image_width']),
        '      <height>{0}</height>'.format(rows.iloc[0]['image_height']),
        '      <depth>3</depth>',
        '   </size>',
        '   <object>',
        '      <name>{0}</name>'.format(rows.iloc[0]['label']),
        '      <bndbox>',
        '         <xmin>{0}</xmin>'.format(rows.iloc[0]['left']),
        '         <ymin>{0}</ymin>'.format(rows.iloc[0]['top']),
        '         <xmax>{0}</xmax>'.format(rows.iloc[0]['left'] + rows.iloc[0]['width']),
        '         <ymax>{0}</ymax>'.format(rows.iloc[0]['top'] + rows.iloc[0]['height']),
        '      </bndbox>',
        '   </object>'
    ]

    for i in range(1, len(rows.index)):
        xml += [
            '   <object>',
            '      <name>{0}</name>'.format(rows.iloc[i]['label']),
            '      <bndbox>',
            '         <xmin>{0}</xmin>'.format(rows.iloc[i]['left']),
            '         <ymin>{0}</ymin>'.format(rows.iloc[i]['top']),
            '         <xmax>{0}</xmax>'.format(rows.iloc[i]['left'] + rows.iloc[i]['width']),
            '         <ymax>{0}</ymax>'.format(rows.iloc[i]['top'] + rows.iloc[i]['height']),
            '      </bndbox>',
            '   </object>',
        ]

    xml.append('</annotation>')

    with open('annotation/train/' + rows.iloc[0]['image_name'].strip('.png') + '.xml', 'w') as f:
        f.write('\n'.join(xml))


if __name__ == '__main__':
    train_folder = 'train'
    h5_file_name = 'train_data_processed.h5'
    h5_file_path = os.path.join(train_folder, h5_file_name)
    mat_file_name = 'digitStruct.mat'
    mat_file_path = os.path.join(train_folder, mat_file_name)

    # load or build dataframe with attributes:
    # ['height', 'image_name', 'label', 'left', 'top', 'width', 'cut_image', 'image', 'image_height', 'image_width']
    dataframe = pd.read_hdf(h5_file_path, 'table') if os.path.exists(h5_file_path) \
        else build_dataset(train_folder, mat_file_name, h5_file_name)

    # convert some columns to integer type
    dataframe[['height', 'label', 'left', 'top', 'width', 'image_height', 'image_width']] = \
        dataframe[['height', 'label', 'left', 'top', 'width', 'image_height', 'image_width']].astype('int')

    # show random images from the dataframe drawing the bounding boxes
    # show_images(dataframe, num_images=2)

    # number of images
    num_images = int(dataframe['image_name'].iloc[-1].strip('.png'))

    # generate annotations in xml format
    for i in np.arange(1, num_images + 1):
        rows = dataframe.loc[dataframe['image_name'] == '{0}.png'.format(i)]
        to_xml(rows)
