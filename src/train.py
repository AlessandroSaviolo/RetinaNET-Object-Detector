"""
Copyright 2017-2018 Fizyr (https://fizyr.com)
 
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
 
    http://www.apache.org/licenses/LICENSE-2.0
 
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Modified on Wed 17 Nov 2019 by AlessandroSaviolo
"""
 
import argparse
import os
import sys
import keras
import keras.preprocessing.image
import tensorflow as tf
from keras_retinanet import losses
from keras_retinanet import models
from keras_retinanet.callbacks import RedirectModel
from keras_retinanet.models.retinanet import retinanet_bbox
from keras_retinanet.utils.anchors import make_shapes_callback
from keras_retinanet.utils.config import read_config_file, parse_anchor_parameters
from keras_retinanet.utils.model import freeze as freeze_model
from retina.pascal import PascalVocGenerator
import warnings
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise
 
 
def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)
 
 
def model_with_weights(model, weights, skip_mismatch):
    """ Load weights for model.
 
    Args
        model         : The model to load weights for.
        weights       : The weights to load.
        skip_mismatch : If True, skips layers whose shape of weights doesn't match with the model.
    """
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model
 
 
def create_models(backbone_retinanet, num_classes, weights, multi_gpu=0, freeze_backbone=False, lr=1e-4, config=None):
    """ Creates three models (model, training_model, prediction_model).
 
    Args
        backbone_retinanet : A function to call to create a retinanet model with a given backbone.
        num_classes        : The number of classes to train.
        weights            : The weights to load into the model.
        multi_gpu          : The number of GPUs to use for training.
        freeze_backbone    : If True, disables learning for the backbone.
        config             : Config parameters, None indicates the default configuration.
 
    Returns
        model            : The base model. This is also the model that is saved in snapshots.
        training_model   : The training model. If multi_gpu=0, this is identical to model.
        prediction_model : The model wrapped with utility functions to perform object detection (applies regression values and performs NMS).
    """
 
    modifier = freeze_model if freeze_backbone else None
 
    # load anchor parameters, or pass None (so that defaults will be used)
    anchor_params = None
    num_anchors = None
    if config and 'anchor_parameters' in config:
        anchor_params = parse_anchor_parameters(config)
        num_anchors = anchor_params.num_anchors()
 
    # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
    # optionally wrap in a parallel model
    if multi_gpu > 1:
        from keras.utils import multi_gpu_model
        with tf.device('/cpu:0'):
            model = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier), weights=weights, skip_mismatch=True)
        training_model = multi_gpu_model(model, gpus=multi_gpu)
    else:
        model = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier), weights=weights, skip_mismatch=True)
        training_model = model
 
    # make prediction model
    prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)
 
    # compile model
    training_model.compile(
        loss={'regression': losses.smooth_l1(), 'classification': losses.focal()},
        optimizer=keras.optimizers.adam(lr=lr, clipnorm=0.001)
    )
 
    return model, training_model, prediction_model
 
 
def create_callbacks(model, training_model, prediction_model, args):
    """ Creates the callbacks to use during training.
 
    Args
        model: The base model.
        training_model: The model that is used for training.
        args: parseargs args object.
 
    Returns:
        A list of callbacks used for training.
    """
    callbacks = []
    tensorboard_callback = None
 
    if args.tensorboard_dir:
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=args.tensorboard_dir,
            histogram_freq=0,
            batch_size=args.batch_size,
            write_graph=True,
            write_grads=False,
            write_images=False,
            embeddings_freq=0,
            embeddings_layer_names=None,
            embeddings_metadata=None
        )
        callbacks.append(tensorboard_callback)
 
    # save the model
    if args.snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        makedirs(args.snapshot_path)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path,
                '{backbone}_{dataset_type}_{{epoch:02d}}.h5'.format(backbone=args.backbone, dataset_type=args.dataset_type)
            ),
            verbose=1,
            save_best_only=True,
            monitor="mAP",
            mode='max'
        )
        checkpoint = RedirectModel(checkpoint, model)
        callbacks.append(checkpoint)
 
    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.1,
        patience=2,
        verbose=1,
        mode='auto',
        min_delta=0.0001,
        cooldown=0,
        min_lr=0
    ))
 
    return callbacks
 
 
def create_generators(args, preprocess_image):
    """ Create generator for training.
 
    Args
        args             : parseargs object containing configuration for generators.
        preprocess_image : Function that preprocesses an image for the network.
    """
    common_args = {
        'batch_size': args.batch_size,
        'config': args.config,
        'image_min_side': args.image_min_side,
        'image_max_side': args.image_max_side,
        'preprocess_image': preprocess_image,
        'group_method': 'ratio',
    }

    train_generator = PascalVocGenerator(
        args.train_imgs_dir,
        args.train_anns_dir,
        **common_args
    )

    return train_generator
 
 
def check_args(parsed_args):
    """ Function to check for inherent contradictions within parsed arguments.
    For example, batch_size < num_gpus
    Intended to raise errors prior to backend initialisation.
 
    Args
        parsed_args: parser.parse_args()
 
    Returns
        parsed_args
    """
 
    if parsed_args.multi_gpu > 1 and parsed_args.batch_size < parsed_args.multi_gpu:
        raise ValueError("Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(
            parsed_args.batch_size, parsed_args.multi_gpu))
 
    if parsed_args.multi_gpu > 1 and parsed_args.snapshot:
        raise ValueError("Multi GPU training ({}) and resuming from snapshots ({}) is not supported.".format(
            parsed_args.multi_gpu, parsed_args.snapshot))
 
    if parsed_args.multi_gpu > 1 and not parsed_args.multi_gpu_force:
        raise ValueError("Multi-GPU support is experimental, use at own risk! Run with --multi-gpu-force if "
                         "you wish to continue.")
 
    if 'resnet' not in parsed_args.backbone:
        warnings.warn('Using experimental backbone {}'.format(
            parsed_args.backbone))
 
    return parsed_args
 
 
def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True
 
    pascal_parser = subparsers.add_parser('pascal')
    pascal_parser.add_argument('train_imgs_dir', default="train")
    pascal_parser.add_argument('train_anns_dir', default="annotation/train")
 
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--snapshot',          help='Resume training from a snapshot.', default='snapshots/resnet152_pascal_03.h5')
    group.add_argument('--imagenet-weights',  help='Initialize the model with pretrained imagenet weights. This is the default behaviour.', action='store_const', const=True, default=True)
    group.add_argument('--weights',           help='Initialize the model with weights from a file.')
    group.add_argument('--no-weights',        help='Don\'t initialize the model with any weights.', dest='imagenet_weights', action='store_const', const=False)
    parser.add_argument('--backbone',         help='Backbone model used by retinanet.', default='resnet50', type=str)
    parser.add_argument('--batch-size',       help='Size of the batches.', default=64, type=int)
    parser.add_argument('--gpu',              help='Id of the GPU to use.')
    parser.add_argument('--multi-gpu',        help='Number of GPUs to use for parallel processing.', type=int, default=0)
    parser.add_argument('--multi-gpu-force',  help='Extra flag needed to enable (experimental) multi-gpu support.', action='store_true')
    parser.add_argument('--epochs',           help='Number of epochs to train.', type=int, default=5)
    parser.add_argument('--steps',            help='Number of steps per epoch.', type=int, default=1000)
    parser.add_argument('--lr',               help='Learning rate.', type=float, default=1e-4)
    parser.add_argument('--snapshot-path',    help='Path to store snapshots of models during training (defaults to \'./snapshots\')', default='./snapshots')
    parser.add_argument('--tensorboard-dir',  help='Log directory for Tensorboard output', default='./logs')
    parser.add_argument('--no-snapshots',     help='Disable saving snapshots.', dest='snapshots', action='store_false')
    parser.add_argument('--no-evaluation',    help='Disable per epoch evaluation.', dest='evaluation', action='store_false')
    parser.add_argument('--freeze-backbone',  help='Freeze training of backbone layers.', action='store_true')
    parser.add_argument('--random-transform', help='Randomly transform image and annotations.', action='store_true')
    parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=416)
    parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=448)
    parser.add_argument('--config',           help='Path to a configuration parameters .ini file.')
    parser.add_argument('--weighted-average', help='Compute the mAP using the weighted average of precisions among classes.', action='store_true')
    parser.add_argument('--compute-val-loss', help='Compute validation loss during training', dest='compute_val_loss', action='store_true')
 
    # Fit generator arguments
    parser.add_argument('--workers', help='Number of multiprocessing workers. To disable multiprocessing, set workers to 0', type=int, default=0)
    parser.add_argument('--max-queue-size', help='Queue length for multiprocessing workers in fit generator.', type=int, default=10)
 
    return check_args(parser.parse_args(args))


def main(args=None):

    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
 
    # create object that stores backbone information
    backbone = models.backbone(args.backbone)
 
    # optionally load config parameters
    if args.config:
        args.config = read_config_file(args.config)

    # create the generators
    train_generator = create_generators(args, backbone.preprocess_image)

    # create the model
    if args.snapshot is not None:
        print('Loading model')
        model = models.load_model(args.snapshot, backbone_name=args.backbone)
        training_model = model
        anchor_params = None
        if args.config and 'anchor_parameters' in args.config:
            anchor_params = parse_anchor_parameters(args.config)
        prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)
    else:
        weights = args.weights
        if weights is None and args.imagenet_weights:
            weights = backbone.download_imagenet()
        print('Creating model')
        model, training_model, prediction_model = create_models(
            backbone_retinanet=backbone.retinanet,
            num_classes=train_generator.num_classes(),
            weights=weights,
            multi_gpu=args.multi_gpu,
            freeze_backbone=args.freeze_backbone,
            lr=args.lr,
            config=args.config
        )
 
    # this lets the generator compute backbone layer shapes using the actual backbone model
    if 'densenet' in args.backbone:
        train_generator.compute_shapes = make_shapes_callback(model)
 
    # create the callbacks
    callbacks = create_callbacks(
        model,
        training_model,
        prediction_model,
        args
    )
 
    # start training
    return training_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=train_generator.size() // args.batch_size,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks
    )


if __name__ == '__main__':
    main(["pascal",
          "train",
          "annotation/train"])
