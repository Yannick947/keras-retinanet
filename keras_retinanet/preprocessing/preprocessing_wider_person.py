import os 
import csv

import pandas as pd
import numpy as np
import shutil
from sklearn.model_selection import train_test_split


annot_path = './content/drive/My Drive/person_detection/WiderPerson/Annotations'
images_path = ',/content/drive/My Drive/person_detection/WiderPerson/Images'
images_path_sample = '/content/sample_data/WiderPerson/Images'

annot_csv_path = '/content/drive/My Drive/person_detection/keras-retinanet/annotations_sampledata.csv'
keras_path = '/content/drive/My Drive/person_detection/keras-retinanet'
#Remove classes which shall be left out in the csv
classes_ids = {1:'pedestrian',
               3:'partially-visible'}


def filter_images_by_size(df, max_size=4000, min_size=440):
    start_size = len(os.listdir(images_path_sample))
    for image_name in os.listdir(images_path_sample):
        try: 
            image = read_image_bgr(images_path_sample + '/' + image_name)
            if (min(image.shape[0:2]) < min_size) or (max(image.shape[0:2]) > max_size):
                df = df[~df.image_name.str.contains(image_name)]
        except: 
            print('Image not in dataset. Name of file: ', image_name)
        print('final df shape: ', df.shape)
        print('Removed ',start_size - df.image_name.nunique() , 'images')
        return df


def train_test_split_files(annot_file_path, lower_filter, upper_filter):
    '''create train test split based on image names, not on annotations'''
    annotations = pd.read_csv('/content/drive/My Drive/person_detection/keras-retinanet/{}.csv'.format(annot_file_path),
                header=None,
                names=['image_name', 'x1', 'y1', 'x2', 'y2', 'label'])
    image_names = pd.Series(os.listdir(images_path), name='image_names')
    image_names = '/content/sample_data/WiderPerson/Images' + '/' + image_names
    train_names, test_names = train_test_split(image_names, test_size=0.15)
    train_df = annotations[annotations.image_name.isin(train_names)]
    test_df = annotations[annotations.image_name.isin(test_names)]
    print(train_df.shape, train_df.image_name.nunique(), test_df.image_name.nunique(), test_df.shape)

    train_df.to_csv(keras_path + '/annot_train_filtered_{}_{}.csv'.format(lower_filter, upper_filter), header=None, index=None)
    test_df.to_csv(keras_path + '/annot_test_filtered_{}_{}.csv'.format(lower_filter, upper_filter), header=None, index=None)


def generate_annotations():
  # Generate the classes csv file
    annot = os.listdir(annot_path)
    with open(keras_path + '/annotations.csv', newline='', mode='x') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=',')
        for filename in os.listdir(images_path_sample)[0:10]:
            if str(filename + '.txt') in annot:
                f = open(annot_path + '/' +  filename + '.txt', 'r')
        
            for index, line in enumerate(f): 
                if index == 0: 
                    if line.strip() == '0':
                        print('Not any  object in the image!')
                    continue
                
            else: 
                split_line = line[:line.find('/')].split(' ')
                first_char = split_line.pop(0)
                split_line.insert(len(split_line), first_char)
                split_line.insert(0, images_path + '/' + filename)
                #convert from index to class label 
                try:
                split_line[-1] = classes_ids[int(split_line[-1])]
                except: 
                continue
                split_line[0] = split_line[0]

                csv_writer.writerow(split_line)
        
        f.close()
  return

def check_bb(path):
    colnames = ['filename', 'x1', 'y1', 'x2', 'y2', 'class_label']
    df = pd.read_csv(path, names=colnames)
    df_new = df.loc[(df.x1 < df.x2) & (df.y1 < df.y2), : ]
    print ('Reduces shape from ', df.shape, 'to ', df_new.shape)
    return df_new