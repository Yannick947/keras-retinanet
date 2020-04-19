import os 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

TOP_PATH = 'C:/Users/Yannick/Google Drive/person_detection/bus_videos/pcds_dataset_detected/'
LABEL_FILE = 'pcds_dataset_labels_united.csv'
np.random.seed(42)

def main():
    
    length_t, length_y = get_lengths(TOP_PATH)
    train_file_names, _ = split_files()
    
    datagen_iter = datagen(length_t, length_y, train_file_names, batch_size=16)
    datagen_iter.next()

def split_files():
    df_names = pd.read_csv(TOP_PATH + LABEL_FILE).iloc[:,0]

    #replace .avi with .csv
    df_names = df_names.apply(lambda row: row[:-4] + '.csv')
    return train_test_split(df_names, test_size=0.2, random_state=42)
    

def datagen(length_t, length_y, file_names, batch_size=16):
    batch_index = 0
    train_x_batch = np.zeros(shape=(batch_size, length_t, length_y))
    train_y_batch = np.zeros(shape=(batch_size, 1))

    while True:
        df_y = pd.read_csv(TOP_PATH + LABEL_FILE)
        #TODO shuffle file_names
        for file_name in file_names: 
            batch_index += 1
            full_path = os.path.join(root, file_name)
        
            batch_x = pd.read_csv(full_path, header=None)
            batch_y = get_label(df_x)
            assert df_x.shape[0] == length_y and df_x.shape[1] == length_y

            # Shape for x must be 3D [samples, timesteps, features] and numpy arrays

            if batch_index == batch_size:
                batch_index = 0
                yield (train_x_batch, train_y_batch)
            

def get_label(df_x, df_y): 
    pass


def get_features():
    pass


def get_lengths(TOP_PATH):
    for root, dirs, files in os.walk(TOP_PATH): 
        for file_name in files:
            if file_name[-4:] == '.csv' and not ('label' in file_name):
                full_path = os.path.join(root, file_name)
                df = pd.read_csv(full_path, header=None)
                return df.shape[0], df.shape[1]

if __name__ == '__main__':
    main()