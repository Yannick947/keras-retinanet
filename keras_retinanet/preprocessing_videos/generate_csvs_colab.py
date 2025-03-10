import argparse
import pandas as pd
import csv
import os 
import time
import numpy as np 
import sys
import random

import keras
import tensorflow as tf
import cv2

from .. import models
from ..utils.image import read_image_bgr, preprocess_image, resize_image

TOP_PATH = '../pcds_dataset_detections/pcds_dataset'
BACKBONE = 'resnet50'

def main(args=None):

    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    if args.static_filter:
        lower_video_length = 0
        upper_video_length = 700 
    else: 
        lower_video_length, upper_video_length = get_video_stats(TOP_PATH,
                                                                 lower_quantile=0.0,
                                                                 upper_quantile=0.8,
                                                                 print_stats=True)

    keras.backend.tensorflow_backend.set_session(get_session())
    model = models.load_model(args.model_path, backbone_name=BACKBONE)
    
    if not 'vanilla.h5' in args.model_path:
        model = models.convert_model(model, custom_nms_theshold = args.nms_threshold)

    csv_counter = generate_csvs(TOP_PATH, model, args,
                                filter_lower_frames=lower_video_length,
                                filter_upper_frames=upper_video_length)
    print_csv_stats(csv_counter, lower_video_length, upper_video_length) 


def print_csv_stats(csv_counter, lower_quantile, upper_quantile):
    """Print some stats about the csv files
    """
    video_number = 0
    for _, _, files in os.walk(TOP_PATH):
        video_number += len([i for i in files if i[-4:] == '.avi'])
    
    filter_factor = lower_quantile + (1 - upper_quantile)
    video_number_filtered = video_number / filter_factor

    print('Video number which should have been processed: ', video_number_filtered)
    print('Videos which were processed ', csv_counter)


def get_session():
    """ Get a tf1 session
    """
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.compat.v1.Session(config=config)


def generate_csvs(TOP_PATH, model, args, **kwargs):
    """ Generate csv files from videos by predcitions of a model
    """
    csv_counter = 0
    for root, _, files in os.walk(TOP_PATH):
        for file_name in files: 
            if file_name[-4:] == '.avi':
                csv_counter += generate_csv(root, file_name, model, args, **kwargs)
    return csv_counter


def generate_csv(root, file_name, model, args, filter_lower_frames=0, filter_upper_frames=1000):
    """ Generate the csv files by passing videos to a model
    """
    video_path = os.path.join(root, file_name)
    if args.skip_existing and csv_exists(video_path, file_name): 
        print('Skip: ', file_name, ' because csv output already exists.')
        return 0

    vcapture = cv2.VideoCapture(video_path)
    num_frames = int(vcapture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if num_frames < filter_lower_frames or num_frames > filter_upper_frames:
        print('Skip: ', file_name, ' because its too long or short.')
        return 0

    #create empty arrs for predictions.
    detections_y = create_zeroed_array(args, num_frames, vcapture, filter_upper_frames)
    detections_x = create_zeroed_array(args, num_frames, vcapture, filter_upper_frames)

    #Fill arrs with predictions and save to csv
    detections_x, detections_y = fill_pred_image(model, detections_x, detections_y, vcapture, args)
    
    stacked_arr = np.stack([detections_x, detections_y], axis=-1, out=None)
    np.save(video_path[0:-4], stacked_arr)
    
    print('Finished video and saved detections to: ', video_path [0:-4]  + '.npy')
    vcapture.release()
    return 1


def fill_pred_image(model, detections_x, detections_y, vcapture, args):
    """ Fill an array with predictions for time and place of a model 
    """
    success = True
    frame_index = 0
    frame_rate = int(vcapture.get(cv2.CAP_PROP_FPS))
    height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))

    if args.fps: 
        time_scale_factor = int(frame_rate / args.fps)
    else: 
        time_scale_factor = 1

    while success:
        frame_index += 1
        success, image = vcapture.read()

        if success and ((frame_index % time_scale_factor) == 0):
            image = preprocess_image(image)

            #800 and 1300 are values usally used during training, adjust if used differently
            image, scale = resize_image(image, min_side=800, max_side=1333)
            boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
            boxes /= scale

            for box, score, label in zip(boxes[0], scores[0], labels[0]):

                # scores are sorted so we can break
                if score < args.predict_threshold:
                    break
                
                #If model predicts more classes than we want to have, filter here
                if (label != 0): 
                    continue

                b = box.astype(int)

                #fill arr with probability at the center of y axis, consider rezizing with args_downscale_factor_y
                try: 
                    t = int(frame_index / time_scale_factor)
                    x = int((b[0] + b[2]) * (height / width) / args.downscale_factor_y / 2 )
                    y = int((b[3] + b[1]) / args.downscale_factor_y / 2)

                    #If position already contains a pixel due to downscaling, search new position along movement axis
                    if detections_y[t, y] == 0: 
                        detections_y[t, y] = score
                        detections_x[t, x] = score

                    else: 
                        t, y = get_destination(detections_y, (t, y), axis='y')
                        t, x = get_destination(detections_x, (t, x), axis='x')
                        detections_y[t, y] = score
                        detections_x[t, x] = score
                except: 
                    print('Detection out of bounds, t: {}, y: {}, x: {}'.format(t, y, x))

    return detections_y, detections_x

def create_zeroed_array(args, num_frames, vcapture, filter_upper_frames):
    frame_rate = int(vcapture.get(cv2.CAP_PROP_FPS))
    height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.fps == None:
        detections_y_length = int(filter_upper_frames)

    else: 
        print('Care about using fps argument, not tested yet -> trial mode')
        detections_y_length = int(filter_upper_frames * args.fps / frame_rate)

    return np.zeros(shape=(detections_y_length, int(height / args.downscale_factor_y)))

def get_video_stats(TOP_PATH, lower_quantile, upper_quantile, print_stats=False, sample_size=100): 
    video_length = pd.Series()
    file_list = list()

    for root, _, files in os.walk(TOP_PATH):
        for file_name in files: 
            if file_name[-4:] == '.avi':
                file_list.append(os.path.join(root, file_name))

    random.shuffle(file_list)
    sampled_list = random.sample(file_list, sample_size)

    for video_path in sampled_list:
        vcapture = cv2.VideoCapture(video_path)
        video_length = video_length.append(pd.Series(int(vcapture.get(cv2.CAP_PROP_FRAME_COUNT))))
        vcapture.release()

    if print_stats == True: 
        print('Std of num video frames: ', video_length.std())
        print('Avg of num video frames: ', video_length.mean())

        for quantile in range(0, 10, 1):
            print('{} quantile of num video frames '.format(quantile / 10),
                video_length.quantile(quantile / 10))
    
    return video_length.quantile(lower_quantile), video_length.quantile(upper_quantile)

def csv_exists(video_path, file_name):
    video_folder = video_path[:video_path.find(file_name)]
    existing_files = os.listdir(video_folder)
    return file_name[0: -4] + '.npy' in existing_files

def get_destination(arr, old_position, axis): 
    '''Get new destination for pixel around the old position

    Arguments: 
        arr: Dataframe with current pixels
        old_position: position where pixel would be placed initially

    returns t and y coordinate for new position of pixel
    '''
    #Directions in which pixel can be moved
    if axis == 'y':
        directions = [(-1,-1), (1, 1)] 
    else: 
        directions = [(1, 0), (-1, 0)]

    for _ in range(len(directions)):
        direction = random.sample(directions, 1)[0]

        x = old_position[0] + direction[0]
        y = old_position[1] + direction[1]
        new_pos = (x, y)

        if (new_pos[0] < 0) or (new_pos[0] >= arr.shape[0])\
        or (new_pos[1] < 0) or (new_pos[1] >= arr.shape[1]):
            continue

        if arr[new_pos[0], new_pos[1]] == 0:
            return new_pos[0], new_pos[1]

    #if all position are already a detection pixel, return the old value
    return old_position


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Parsing Arguments for generating csv files from video db')
    parser.add_argument('--static-filter',      help='If static filter, videos which are below and above static filter number are not considered', action='store_true')
    parser.add_argument('--nms-threshold',      help='Non maximum suppression threshold for bounding boxes, low values supress more boxes', default=0.3, type=float)
    parser.add_argument('--predict-threshold',  help='If prediction is below this value, bounding box is filtered', default=0.4, type=float)
    parser.add_argument('--fps',                help='Frames per second in every video', default=None, type=int)
    parser.add_argument('--skip-existing',      help='Flag if the existing csv files shall be skipped or calculated once again', action='store_true')
    parser.add_argument('--downscale-factor-y', help='Factor which is used to scale down y axis', default=1, type=int)
    parser.add_argument('--model-path',         help='Path to the model which shall be used for inference')
    return parser.parse_args(args)

if __name__ == '__main__':
    main()