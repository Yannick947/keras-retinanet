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

import keras
from keras.utils import get_file

from . import retinanet
from . import Backbone
import efficientnet.keras as efn


class EfficientNetBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    def __init__(self, backbone):
        super(EfficientNetBackbone, self).__init__(backbone)
        self.preprocess_image_func = None

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return effnet_retinanet(*args, backbone=self.backbone, **kwargs)

    def download_imagenet(self):
        """ Downloads ImageNet weights and returns path to weights file.
        """
        BASE_WEIGHTS_PATH = (
            'https://github.com/Callidior/keras-applications/'
            'releases/download/efficientnet/')
        WEIGHTS_HASHES = {
            'efficientnet-b0': ('e9e877068bd0af75e0a36691e03c072c',
                                '345255ed8048c2f22c793070a9c1a130'),
            'efficientnet-b1': ('8f83b9aecab222a9a2480219843049a1',
                                'b20160ab7b79b7a92897fcb33d52cc61'),
            'efficientnet-b2': ('b6185fdcd190285d516936c09dceeaa4',
                                'c6e46333e8cddfa702f4d8b8b6340d70'),
            'efficientnet-b3': ('b2db0f8aac7c553657abb2cb46dcbfbb',
                                'e0cf8654fad9d3625190e30d70d0c17d'),
            'efficientnet-b4': ('ab314d28135fe552e2f9312b31da6926',
                                'b46702e4754d2022d62897e0618edc7b'),
            'efficientnet-b5': ('8d60b903aff50b09c6acf8eaba098e09',
                                '0a839ac36e46552a881f2975aaab442f'),
            'efficientnet-b6': ('a967457886eac4f5ab44139bdd827920',
                                '375a35c17ef70d46f9c664b03b4437f2'),
            'efficientnet-b7': ('e964fd6e26e9a4c144bcb811f2a10f20',
                                'd55674cc46b805f4382d18bc08ed43c1')
        }

        model_name = 'efficientnet-b' + self.backbone[-1]
        file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
        file_hash = WEIGHTS_HASHES[model_name][1]
        weights_path = get_file(file_name, BASE_WEIGHTS_PATH + file_name, cache_subdir='models', file_hash=file_hash)
        return weights_path

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4',
                             'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7']
        backbone = self.backbone.split('_')[0]

        if backbone not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(backbone, allowed_backbones))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return efn.preprocess_input(inputs)


def effnet_retinanet(num_classes, backbone='EfficientNetB0', inputs=None, modifier=None, **kwargs):
    """ Constructs a retinanet model using a resnet backbone.

    Args
        num_classes: Number of classes to predict.
        backbone: Which backbone to use (one of ('resnet50', 'resnet101', 'resnet152')).
        inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
        modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).

    Returns
        RetinaNet model with a ResNet backbone.
    """
    # choose default input
    if inputs is None:
        if keras.backend.image_data_format() == 'channels_first':
            inputs = keras.layers.Input(shape=(3, None, None))
        else:
            # inputs = keras.layers.Input(shape=(224, 224, 3))
            inputs = keras.layers.Input(shape=(None, None, 3))

    # get last conv layer from the end of each block [28x28, 14x14, 7x7]
    if backbone == 'EfficientNetB0':
        model = efn.EfficientNetB0(input_tensor=inputs, include_top=False, weights=None)
    elif backbone == 'EfficientNetB1':
        model = efn.EfficientNetB1(input_tensor=inputs, include_top=False, weights=None)
    elif backbone == 'EfficientNetB2':
        model = efn.EfficientNetB2(input_tensor=inputs, include_top=False, weights=None)
    elif backbone == 'EfficientNetB3':
        model = efn.EfficientNetB3(input_tensor=inputs, include_top=False, weights=None)
    elif backbone == 'EfficientNetB4':
        model = efn.EfficientNetB4(input_tensor=inputs, include_top=False, weights=None)
    elif backbone == 'EfficientNetB5':
        model = efn.EfficientNetB5(input_tensor=inputs, include_top=False, weights=None)
    elif backbone == 'EfficientNetB6':
        model = efn.EfficientNetB6(input_tensor=inputs, include_top=False, weights=None)
    elif backbone == 'EfficientNetB7':
        model = efn.EfficientNetB7(input_tensor=inputs, include_top=False, weights=None)
    else:
        raise ValueError('Backbone (\'{}\') is invalid.'.format(backbone))

    layer_outputs = ['block4a_expand_activation', 'block6a_expand_activation', 'top_activation']

    layer_outputs = [
        model.get_layer(name=layer_outputs[0]).output,  # 28x28
        model.get_layer(name=layer_outputs[1]).output,  # 14x14
        model.get_layer(name=layer_outputs[2]).output,  # 7x7
    ]
    # create the densenet backbone
    model = keras.models.Model(inputs=inputs, outputs=layer_outputs, name=model.name)

    # invoke modifier if given
    if modifier:
        model = modifier(model)

    # create the full model
    return retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=model.outputs, **kwargs)


def EfficientNetB0_retinanet(num_classes, inputs=None, **kwargs):
    return effnet_retinanet(num_classes=num_classes, backbone='EfficientNetB0', inputs=inputs, **kwargs)


def EfficientNetB1_retinanet(num_classes, inputs=None, **kwargs):
    return effnet_retinanet(num_classes=num_classes, backbone='EfficientNetB1', inputs=inputs, **kwargs)


def EfficientNetB2_retinanet(num_classes, inputs=None, **kwargs):
    return effnet_retinanet(num_classes=num_classes, backbone='EfficientNetB2', inputs=inputs, **kwargs)


def EfficientNetB3_retinanet(num_classes, inputs=None, **kwargs):
    return effnet_retinanet(num_classes=num_classes, backbone='EfficientNetB3', inputs=inputs, **kwargs)


def EfficientNetB4_retinanet(num_classes, inputs=None, **kwargs):
    return effnet_retinanet(num_classes=num_classes, backbone='EfficientNetB4', inputs=inputs, **kwargs)


def EfficientNetB5_retinanet(num_classes, inputs=None, **kwargs):
    return effnet_retinanet(num_classes=num_classes, backbone='EfficientNetB5', inputs=inputs, **kwargs)


def EfficientNetB6_retinanet(num_classes, inputs=None, **kwargs):
    return effnet_retinanet(num_classes=num_classes, backbone='EfficientNetB6', inputs=inputs, **kwargs)


def EfficientNetB7_retinanet(num_classes, inputs=None, **kwargs):
    return effnet_retinanet(num_classes=num_classes, backbone='EfficientNetB7', inputs=inputs, **kwargs)
