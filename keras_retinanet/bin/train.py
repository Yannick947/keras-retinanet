#!/usr/bin/env python

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
"""

import argparse
import os
import sys
import warnings
import json

import keras
import keras.preprocessing.image
import tensorflow as tf
# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from .. import layers  # noqa: F401
from .. import losses
from .. import models
from ..callbacks import RedirectModel
from ..callbacks.eval import Evaluate
from ..models.retinanet import retinanet_bbox
from ..preprocessing.csv_generator import CSVGenerator
from ..preprocessing.kitti import KittiGenerator
from ..preprocessing.open_images import OpenImagesGenerator
from ..preprocessing.pascal_voc import PascalVocGenerator
from ..utils.anchors import make_shapes_callback
from ..utils.config import read_config_file, parse_anchor_parameters
from ..utils.gpu import setup_gpu
from ..utils.image import random_visual_effect_generator
from ..utils.keras_version import check_keras_version
from ..utils.model import freeze as freeze_model
from ..utils.tf_version import check_tf_version
from ..utils.transform import random_transform_generator
from tensorboard.plugins.hparams import api as hp
from datetime import datetime

def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


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


def create_models(backbone_retinanet, num_classes, weights, multi_gpu=0,
                  freeze_backbone=False, lr=1e-5, config=None):
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
    num_anchors   = None
    if config and 'anchor_parameters' in config:
        anchor_params = parse_anchor_parameters(config)
        num_anchors   = anchor_params.num_anchors()


    # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
    # optionally wrap in a parallel model
    if multi_gpu > 1:
        from keras.utils import multi_gpu_model
        with tf.device('/cpu:0'):
            model = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier), weights=weights, skip_mismatch=True)
        training_model = multi_gpu_model(model, gpus=multi_gpu)
    else:
        model          = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier), weights=weights, skip_mismatch=True)
        training_model = model

    # make prediction model
    prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)

    # compile model
    training_model.compile(
        loss={
            'regression'    : losses.smooth_l1(),
            'classification': losses.focal()
        },
        optimizer=keras.optimizers.adam(lr=lr, clipnorm=0.001)
    )

    return model, training_model, prediction_model


def create_callbacks(model, training_model, prediction_model, validation_generator, args):
    """ Creates the callbacks to use during training.

    Args
        model: The base model.
        training_model: The model that is used for training.
        prediction_model: The model that should be used for validation.
        validation_generator: The generator for creating validation data.
        args: parseargs args object.

    Returns:
        A list of callbacks used for training.
    """
    callbacks = []   

    tensorboard_callback = None

    if args.tensorboard_dir:
        makedirs(args.tensorboard_dir)
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir                = args.tensorboard_dir,
            update_freq            = 'epoch',
            histogram_freq         = 0,
            batch_size             = args.batch_size,
            write_graph            = False,
            write_grads            = False,
            write_images           = False,
            embeddings_freq        = 0,
            embeddings_layer_names = None,
            embeddings_metadata    = None
        )
    
    
    HP_MIN_SIDE = hp.HParam('image_max_side', hp.IntInterval(1500, 1800))
    HP_MAX_SIDE = hp.HParam('image_min_side', hp.IntInterval(100, 800))
    HP_BACKBONE = hp.HParam('backbone', hp.Discrete(['resnet50', 'resnet152',
                                                     'mobilenet128_0.75', 'mobilenet224_1.0',
                                                     'retinanet','densenet', 'vgg16',
                                                     'vgg19', 'EfficientNetB7', 'EfficientNetB6',
                                                     'EfficientNet5']))
    HP_AUGMENTATION = hp.HParam('augmentation', hp.Discrete(['true', 'false']))
    HP_FREEZEBACKBONE = hp.HParam('backbone_frozen', hp.Discrete(['true', 'false']))
    HP_ANCHOROPTI = hp.HParam('anchors_optimized', hp.Discrete(['true', 'false']))
    HP_DATEINT = hp.HParam('date_asint', hp.IntInterval(1, 30000000000000))
    HP_NORESIZE = hp.HParam('no_resize', hp.Discrete(['true', 'false']))
    HP_AUGMENTATION_FACTOR = hp.HParam('augmentation_factor', hp.RealInterval(0.0, 10.0))
    HP_VISUAL_AUG_FACTOR = hp.HParam('visual_aug_factor', hp.RealInterval(0.0,10.0))
    HP_NUM_TRAIN_IMAGES = hp.HParam('num_train_imgs', hp.IntInterval(1, 10000))
    HP_PRETRAINED_ON = hp.HParam('pretrained_on', hp.Discrete(['Wider Person', 'ImageNet', 'Coco']))
    HP_LR = hp.HParam('learning_rate', hp.RealInterval(1e-6, 1e-3))

    if args.no_resize: 
        no_resize_flag = 'true'
    else: 
        no_resize_flag = 'false'
    
    if args.config: 
        anchors_opti = 'true'
    else: 
        anchors_opti = 'false'

    if args.freeze_backbone: 
        frozen = 'true'
    else: 
        frozen = 'false'

    if args.random_transform: 
        aug = 'true'
    else: 
        aug = 'false'

    if args.weights is not None: 
        pretrained = 'coco'

    elif args.snapshot is not None: 
        pretrained = 'Wider Person'

    else: 
        pretrained = 'Imagenet'

    hparams = {
        HP_MIN_SIDE             : args.image_min_side,
        HP_MAX_SIDE             : args.image_max_side,
        HP_BACKBONE             : args.backbone,
        HP_AUGMENTATION         : aug,
        HP_FREEZEBACKBONE       : frozen,
        HP_ANCHOROPTI           : anchors_opti,
        HP_DATEINT              : int(datetime.now().strftime("%m%d%H")),
        HP_VISUAL_AUG_FACTOR    : args.visual_aug_factor, 
        HP_AUGMENTATION_FACTOR  : args.augmentation_factor,
        HP_NORESIZE             : no_resize_flag,
        HP_NUM_TRAIN_IMAGES     : args.num_train_imgs,
        HP_PRETRAINED_ON        : pretrained,
        HP_LR                   : args.lr,
    }

    callbacks.append(hp.KerasCallback(args.tensorboard_dir, hparams))

    if args.evaluation and validation_generator:

        evaluation = Evaluate(validation_generator, tensorboard=tensorboard_callback, weighted_average=args.weighted_average)
        evaluation = RedirectModel(evaluation, prediction_model)
        callbacks.append(evaluation)

    # save the model
    if args.snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        makedirs(args.snapshot_path)

        time = datetime.now().strftime("%d-%m-%Y_%H%-M%-S")
        print('Time now and foldername in Tensorboard: ', time)

        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path,
                '{daytime}_{backbone}_{{epoch:02d}}_{num_trainer}.h5'.format(daytime=time, backbone=args.backbone, num_trainer=args.num_trainer)
            ),
            verbose=1,
            save_best_only=True,
            monitor="mAP",
            mode='max'
        )
        checkpoint = RedirectModel(checkpoint, model)
        callbacks.append(checkpoint)

    if args.tensorboard_dir:
        callbacks.append(tensorboard_callback)

    return callbacks


def create_generators(args, preprocess_image):
    """ Create generators for training and validation.

    Args
        args             : parseargs object containing configuration for generators.
        preprocess_image : Function that preprocesses an image for the network.
    """
    common_args = {
        'batch_size'       : args.batch_size,
        'config'           : args.config,
        'image_min_side'   : args.image_min_side,
        'image_max_side'   : args.image_max_side,
        'no_resize'        : args.no_resize,
        'preprocess_image' : preprocess_image,
    }

    # create random transform generator for augmenting training data
    if args.random_transform:
        transform_generator = random_transform_generator(
            min_rotation=-0.1 * args.augmentation_factor,
            max_rotation=0.1 * args.augmentation_factor,
            min_translation=(-0.1 * args.augmentation_factor, -0.1 * args.augmentation_factor),
            max_translation=(0.1 * args.augmentation_factor, 0.1 * args.augmentation_factor),
            min_shear=-0.1 * args.augmentation_factor,
            max_shear=0.1 * args.augmentation_factor,
            min_scaling=(1 - args.augmentation_factor / 10, 1 - args.augmentation_factor / 10),
            max_scaling=(1 + args.augmentation_factor / 10, 1 + args.augmentation_factor / 10),
            flip_x_chance=0.5,
            #Does make not that much sense for humans
            # flip_y_chance=0.5,
        )
        visual_effect_generator = random_visual_effect_generator(
            contrast_range=(1 - args.visual_aug_factor / 10, 1 + args.visual_aug_factor / 10),
            brightness_range=(-.1 * args.visual_aug_factor, .1 * args.visual_aug_factor),
            hue_range=(-0.05 * args.visual_aug_factor, 0.05 * args.visual_aug_factor),
            saturation_range=(1 - args.visual_aug_factor / 20, 1 + args.visual_aug_factor / 20)
        )
    else:
        transform_generator = random_transform_generator(flip_x_chance=0.5)
        visual_effect_generator = None

    train_generator = CSVGenerator(
        args.annotations,
        args.classes,
        transform_generator=transform_generator,
        visual_effect_generator=visual_effect_generator,
        **common_args
    )

    if args.val_annotations:
        validation_generator = CSVGenerator(
            args.val_annotations,
            args.classes,
            shuffle_groups=False,
            **common_args
        )
        print(args.val_annotations)
    else:
        validation_generator = None

    return train_generator, validation_generator


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
        raise ValueError(
            "Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(parsed_args.batch_size,
                                                                                             parsed_args.multi_gpu))

    if parsed_args.multi_gpu > 1 and parsed_args.snapshot:
        raise ValueError(
            "Multi GPU training ({}) and resuming from snapshots ({}) is not supported.".format(parsed_args.multi_gpu,
                                                                                                parsed_args.snapshot))

    if parsed_args.multi_gpu > 1 and not parsed_args.multi_gpu_force:
        raise ValueError("Multi-GPU support is experimental, use at own risk! Run with --multi-gpu-force if you wish to continue.")

    if 'resnet' not in parsed_args.backbone:
        warnings.warn('Using experimental backbone {}. Only resnet50 has been properly tested.'.format(parsed_args.backbone))

    return parsed_args


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    coco_parser = subparsers.add_parser('coco')
    coco_parser.add_argument('coco_path', help='Path to dataset directory (ie. /tmp/COCO).')

    pascal_parser = subparsers.add_parser('pascal')
    pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')
    pascal_parser.add_argument('--image-extension',   help='Declares the dataset images\' extension.', default='.jpg')

    kitti_parser = subparsers.add_parser('kitti')
    kitti_parser.add_argument('kitti_path', help='Path to dataset directory (ie. /tmp/kitti).')

    def csv_list(string):
        return string.split(',')

    oid_parser = subparsers.add_parser('oid')
    oid_parser.add_argument('main_dir', help='Path to dataset directory.')
    oid_parser.add_argument('--version',  help='The current dataset version is v4.', default='v4')
    oid_parser.add_argument('--labels-filter',  help='A list of labels to filter.', type=csv_list, default=None)
    oid_parser.add_argument('--annotation-cache-dir', help='Path to store annotation cache.', default='.')
    oid_parser.add_argument('--parent-label', help='Use the hierarchy children of this label.', default=None)

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for training.')
    csv_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')
    csv_parser.add_argument('--val-annotations', help='Path to CSV file containing annotations for validation (optional).')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--snapshot',          help='Resume training from a snapshot.')
    group.add_argument('--imagenet-weights',  help='Initialize the model with pretrained imagenet weights. This is the default behaviour.', action='store_const', const=True, default=True)
    group.add_argument('--weights',           help='Initialize the model with weights from a file.')
    group.add_argument('--no-weights',        help='Don\'t initialize the model with any weights.', dest='imagenet_weights', action='store_const', const=False)
    parser.add_argument('--backbone',         help='Backbone model used by retinanet.', default='resnet50', type=str)
    parser.add_argument('--batch-size',       help='Size of the batches.', default=1, type=int)
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).', type=int)
    parser.add_argument('--multi-gpu',        help='Number of GPUs to use for parallel processing.', type=int, default=0)
    parser.add_argument('--multi-gpu-force',  help='Extra flag needed to enable (experimental) multi-gpu support.', action='store_true')
    parser.add_argument('--initial-epoch',    help='Epoch from which to begin the train, useful if resuming from snapshot.', type=int, default=0)
    parser.add_argument('--epochs',           help='Number of epochs to train.', type=int, default=50)
    parser.add_argument('--steps',            help='Number of steps per epoch.', type=int, default=10000)
    parser.add_argument('--lr',               help='Learning rate.', type=float, default=1e-5)
    parser.add_argument('--snapshot-path',    help='Path to store snapshots of models during training (defaults to \'./snapshots\')', default='./snapshots')
    parser.add_argument('--tensorboard-dir',  help='Log directory for Tensorboard output', default='')  # default='./logs') => https://github.com/tensorflow/tensorflow/pull/34870
    parser.add_argument('--no-snapshots',     help='Disable saving snapshots.', dest='snapshots', action='store_false')
    parser.add_argument('--no-evaluation',    help='Disable per epoch evaluation.', dest='evaluation', action='store_false')
    parser.add_argument('--freeze-backbone',  help='Freeze training of backbone layers.', action='store_true')
    parser.add_argument('--random-transform', help='Randomly transform image and annotations.', action='store_true')
    parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
    parser.add_argument('--no-resize',        help='Don''t rescale the image.', action='store_true')
    parser.add_argument('--config',           help='Path to a configuration parameters .ini file.')
    parser.add_argument('--weighted-average', help='Compute the mAP using the weighted average of precisions among classes.', action='store_true')
    parser.add_argument('--compute-val-loss', help='Compute validation loss during training', dest='compute_val_loss', action='store_true')
    parser.add_argument('--num-trainer',      help='The number of the trainer notebook in colab', type=int, default=0)
    parser.add_argument('--augmentation-factor',help='The factor how much image augmentation will be done,big values mean more augmentation', type=float, default=1.0)
    parser.add_argument('--visual-aug-factor', help='Visual augmentation factor, high values mean more augmentation, 0 means 0 visual augmentation', type=float, default=1.0)
    parser.add_argument('--num-train-imgs',   help='The number of training images which were used', type=int, default=100)
    
    # Fit generator arguments
    parser.add_argument('--multiprocessing',  help='Use multiprocessing in fit_generator.', action='store_true')
    parser.add_argument('--workers',          help='Number of generator workers.', type=int, default=1)
    parser.add_argument('--max-queue-size',   help='Queue length for multiprocessing workers in fit_generator.', type=int, default=10)

    return check_args(parser.parse_args(args))

def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # create object that stores backbone information
    backbone = models.backbone(args.backbone)

    # make sure keras and tensorflow are the minimum required version
    check_keras_version()
    check_tf_version()

    # optionally choose specific GPU
    if args.gpu is not None:
        setup_gpu(args.gpu)

    # optionally load config parameters
    if args.config:
        args.config = read_config_file(args.config)

    # create the generators
    train_generator, validation_generator = create_generators(args, backbone.preprocess_image)

    # create the model
    if args.snapshot is not None:
        print('Loading model, this may take a second...')
        model            = models.load_model(args.snapshot, backbone_name=args.backbone)
        training_model   = model
        anchor_params    = None
        if args.config and 'anchor_parameters' in args.config:
            anchor_params = parse_anchor_parameters(args.config)
        prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)
    else:
        weights = args.weights
        # default to imagenet if nothing else is specified
        if weights is None and args.imagenet_weights:
            weights = backbone.download_imagenet()

        print('Creating model, this may take a second...')
        model, training_model, prediction_model = create_models(
            backbone_retinanet=backbone.retinanet,
            num_classes=train_generator.num_classes(),
            weights=weights,
            multi_gpu=args.multi_gpu,
            freeze_backbone=args.freeze_backbone,
            lr=args.lr,
            config=args.config
        )

    # print model summary
    print('\nNumber of params for this model: ', model.count_params())

    # this lets the generator compute backbone layer shapes using the actual backbone model
    if 'vgg' in args.backbone or 'densenet' in args.backbone:
        train_generator.compute_shapes = make_shapes_callback(model)
        if validation_generator:
            validation_generator.compute_shapes = train_generator.compute_shapes

    # create the callbacks
    callbacks = create_callbacks(
        model,
        training_model,
        prediction_model,
        validation_generator,
        args,
    )

    if not args.compute_val_loss:
        validation_generator = None

    # start training
    training_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=len(train_generator),
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
        workers=args.workers,
        use_multiprocessing=args.multiprocessing,
        max_queue_size=args.max_queue_size,
        validation_data=validation_generator,
        initial_epoch=args.initial_epoch
    )


if __name__ == '__main__':
     main()
