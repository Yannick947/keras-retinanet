import os 
import pandas as pd
import numpy as np
from random import shuffle
import math

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from process_labels import correct_path_csv, load_labels
from scaler import CSVScaler



TOP_PATH = 'C:/Users/Yannick/Google Drive/person_detection/bus_videos/pcds_dataset_detected/'

LABEL_FILE = 'pcds_dataset_labels_united.csv'
LABEL_HEADER = ['file_name', 'entering', 'exiting', 'video_type']

#Factor which indicates how many of the timesteps shall be skipped
# FILTER_ROWS = 5 means every fifth timestep will be used
FILTER_ROWS_FACTOR = 2
FILTER_COLS = 30

np.random.seed(42)


def main():

    length_t, length_y = get_filtered_lengths(TOP_PATH)
    train_file_names, _ = split_files()

    filter_cols, filter_rows = get_filters(train_file_names)

    gen = Generator_CSVS(length_t, length_y,
                         train_file_names, filter_cols,
                         filter_rows, batch_size=16)

    for _ in range(5):
        print(next(gen.datagen))


class Generator_CSVS(keras.utils.Sequence):

    def __init__(self,
                 length_t,
                 length_y,
                 file_names,
                 filter_cols,
                 filter_rows,
                 batch_size, 
                 top_path=TOP_PATH,
                 label_file=LABEL_FILE): 

        self.top_path       = top_path
        self.label_file     = label_file
        self.length_t       = length_t
        self.length_y       = length_y
        self.file_names     = file_names 
        self.filter_cols    = filter_cols
        self.filter_rows    = filter_rows
        self.batch_size     = batch_size
        self.labels         = list()
        self.scaler         = CSVScaler(top_path, label_file, file_names)
        self.df_y           = pd.read_csv(self.top_path + self.label_file, header=None, names=LABEL_HEADER)


    def __len__(self):
        return int(np.ceil(len(self.file_names) / float(self.batch_size)))    

    
    def __getitem__(self, file_name):

        df_x = self.get_features(file_name)
        #TODO: Only remove when index is saved in csv
        if df_x is not None:
            df_x.drop(df_x.columns[0], axis=1, inplace=True)
            df_x = self.scaler.transform_features(df_x)

        label = get_entering(file_name, self.df_y)
        label = self.scaler.transform_labels(label)

        if (df_x is None) or (label is None): 
            raise FileNotFoundError('No matching csv for existing label, or scaling went wrong')

        df_x = clean_ends(df_x, del_leading=self.filter_cols, del_trailing=self.filter_cols)
        df_x = filter_rows(df_x, self.filter_rows)
        assert df_x.shape[0] == (self.length_t)\
           and df_x.shape[1] == (self.length_y)

        return df_x, label


    def get_labels(self): 
        return np.array(self.labels)

    def reset_label_states(self): 
        self.labels = list()
        
    def datagen(self):

        '''
        Datagenerator for bus video csv

        yields: Batch of samples
        '''

        batch_index = 0

        x_batch = np.zeros(shape=(self.batch_size,
                                self.length_t,
                                self.length_y))
        y_batch = np.zeros(shape=(self.batch_size, 1))

        while True:

            for file_name in self.file_names.sample(frac=1): 
                try: 
                    df_x, label = self.__getitem__(file_name)
                except FileNotFoundError as e: 
                    continue

                x_batch[batch_index,:,:] = df_x
                y_batch[batch_index] = label
                batch_index += 1

                # Shape for x must be 3D [samples, timesteps, features] and numpy arrays
                if batch_index == self.batch_size:
                    batch_index = 0
                    self.labels.extend(list(y_batch))
                    yield (x_batch, y_batch)


    def get_features(self, file_name): 
        '''
        Get sample of features for given filename. 

        Arguments: 
            file_name: Name of given training sample

            returns: Features for given file_name
        '''

        full_path = os.path.join(TOP_PATH, file_name)

        try:
            df_x = pd.read_csv(full_path, header=None)
            return df_x

        except Exception as e:
            # print('No matching file for label found, skip')
            return None

def check_for_index_col(top_path): 
    '''
    returns: True if disturbing index column in sample csv file exists. Doesnt hold for all.
    '''

    for root, _, files in os.walk(top_path): 
        for file_name in files:
            if file_name[-4:] == '.csv' and not ('label' in file_name):
                full_path = os.path.join(root, file_name)
                df = pd.read_csv(full_path, header=None)
                for i in range(df.shape[0]):
                    if df.iloc[i, 0] != i:
                        return False
                return True

def create_datagen(top_path=TOP_PATH): 
    '''
    '''
    length_t, length_y = get_filtered_lengths(top_path)

    train_file_names, test_file_names = split_files()

    filter_cols, filter_rows_factor = get_filters(train_file_names)

    gen_train = Generator_CSVS(length_t, length_y,
                               train_file_names, filter_cols, 
                               filter_rows_factor, batch_size=16)

    gen_test = Generator_CSVS(length_t, length_y,
                              test_file_names, filter_cols, 
                              filter_rows_factor, batch_size=16)

    return gen_train, gen_test


def split_files():
    df_names = pd.read_csv(TOP_PATH + LABEL_FILE).iloc[:,0]

    #replace .avi with .csv
    df_names = df_names.apply(lambda row: row[:-4] + '.csv')
    return train_test_split(df_names, test_size=0.2, random_state=42)
            

def get_filters(file_names): 
    #TODO: Implement. Now dummy function
    return FILTER_COLS, FILTER_ROWS_FACTOR


def get_entering(file_name, df_y): 
    '''
    Get number of entering persons to existing training sample. 

    Arguments: 
        file_name: Name of given training sample
        df_y: Dataframe with all labels for all samples

        returns: Label for given features
    '''
    try: 
        entering = df_y.loc[df_y.file_name == file_name].entering
        return entering 

    except Exception as e:
        # print('No matching label found for existing csv file')
        return None

def get_exiting(file_name, df_y): 
    '''
    Get number of exiting persons to existing training sample. 

    Arguments: 
        file_name: Name of given training sample
        df_y: Dataframe with all labels for all samples

        returns: Exiting persons for given file
    '''
    try: 
        exiting = df_y.loc[df_y.file_name == file_name].exiting
        return exiting 

    except Exception as e:
        # print(e, ', no matching label found for existing csv file')
        return None
    


def clean_ends(df, del_leading=5, del_trailing=5):
    ''' Delete leading and trailing columns due to sparsity. 

    Arguments: 
        df: Dataframe to adjust
        del_leading: Number of leading columns to delete
        del_trailing: Number of trailing columns to delete
        
    returns: Dataframe with cleaned columns
    '''

    for i in range(del_leading):
        df.drop(df.columns[i], axis=1, inplace=True)

    col_length = df.shape[1]

    for i in range(del_trailing):
        df.drop(df.columns[col_length - i - 1], axis=1, inplace=True)
    
    return df

def filter_rows(df, filter_rows): 
    '''
    '''
    return df.iloc[::filter_rows, :]


    

def get_lengths(top_path=TOP_PATH):
    '''
    returns: Number of timesteps, number of features (columns)
    '''

    for root, _, files in os.walk(top_path): 
        for file_name in files:
            if file_name[-4:] == '.csv' and not ('label' in file_name):
                full_path = os.path.join(root, file_name)
                df = pd.read_csv(full_path, header=None)
                if check_for_index_col(top_path):
                    print('Warning: Index column existing, make sure to drop it!')
                    return df.shape[0], df.shape[1] - 1
                else: 
                    return df.shape[0], df.shape[1]

def get_filtered_lengths(top_path=TOP_PATH,
                         filter_cols=FILTER_COLS,
                         filter_rows=FILTER_ROWS_FACTOR):
    '''
    
    '''

    timestep_num, feature_num = get_lengths(top_path)
    return math.ceil(timestep_num / filter_rows), feature_num - (2 * filter_cols)

if __name__ == '__main__':
    main()