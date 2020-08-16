import argparse
import pandas as pd
import csv
import os 
from timeit import default_timer as timer
import numpy as np 
import sys
import random

import keras
import tensorflow as tf
import cv2

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

models_path = '/content/drive/My Drive/person_detection/keras-retinanet/snapshots_pcds/runtime_analysis/'
images_path = '/content/drive/My Drive/person_detection/video_labeling/video_images'
RESOLUTIONS = [240, 480, 800]
excludes = ['vgg16', 'vgg19', 'resnet50', 'resnet152', 'mobilenet224_1.0', 'EfficientNetB3', 'EfficientNetB7', 'densenet121']
def main():
    models_stats_list = list()
    model_attributes = parse_models(models_path)

    images = get_images(images_path)

    for model_name in model_attributes.keys():
        print('\n Evaluating Retinanet with backbone {}'.format(model_attributes[model_name]['backbone']))

        model = models.load_model(os.path.join(models_path, model_name),
                                  backbone_name=model_attributes[model_name]['backbone'])
        number_params = model.count_params()
        model = models.convert_model(model, custom_nms_theshold = 0.5)
        
        for resolution in RESOLUTIONS:
            inference_time = predict(images, model, resolution=resolution)
            models_stats_list.append(['RetinaNet', 
                                    model_attributes[model_name]['backbone'], 
                                    number_params, 
                                    str(resolution)+ 'x' + str(int(resolution * 320 / 240)),
                                    resolution, 
                                    inference_time])
            print('Average inference time for backbone {} and image resolution {} is {} s'.format(model_attributes[model_name]['backbone'],
                                                                                                  resolution,
                                                                                                  inference_time))
    models_stats = pd.DataFrame(models_stats_list, columns=['model', 'backbone', 'number_parameters', 'image_resolution', 'image_min_side', 'inference_time'])
    models_stats.to_csv('runtime_analysis.csv', index=None)

def get_images(images_path, sample_size=500):
    """ Get .jpg images in given path
    """
    images = list()
    print('Loading images .. ')
    for image_name in os.listdir(images_path):
        if image_name[-4:] == '.jpg':
            img_path = os.path.join(images_path, image_name)
            images.append(read_image_bgr(img_path))
        if len(images) == sample_size: 
            return images
    return images

def parse_models(models_path): 
    """Get atributes of existing models
    """
    models_attributes = dict()
    for model_name in os.listdir(models_path):
        if (not '.h5' in model_name) or (model_name[:-3] in excludes): 
            continue
        backbone = model_name[:model_name.find('.h5')]
        models_attributes[model_name] = {'backbone': backbone}
    return models_attributes
        

def predict(images, model, resolution):
    """Predict for given images and return average inference time
    """
    start_time = timer()
    for image in images:
        image = preprocess_image(image.copy())
        
        if resolution != 240:
            image, scale = resize_image(image, min_side=resolution)
        else: 
            scale = 1.0

        boxes, scores, labels = model.predict_on_batch(
            np.expand_dims(image, axis=0)
        )
        boxes /= scale
    end_time = timer()
    return (end_time - start_time) / len(images)

if __name__ == '__main__':
    main()