import os 
import pandas as pd
import numpy as np

from process_labels import correct_path_csv, load_labels

from sklearn.model_selection import train_test_split



TOP_PATH = 'C:/Users/Yannick/Google Drive/person_detection/bus_videos/pcds_dataset_detected/'

LABEL_FILE = 'pcds_dataset_labels_united.csv'
LABEL_HEADER = ['file_name', 'entering', 'exiting', 'video_type']

FILTER_ROWS = 0 
FILTER_COLS = 5

np.random.seed(42)


def main():

    length_t, length_y = get_lengths(TOP_PATH)
    train_file_names, _ = split_files()

    filter_cols, filter_rows = get_filters(train_file_names)

    gen = datagen(length_t, length_y,
                  train_file_names, filter_cols, 
                  filter_rows, batch_size=16)

    for _ in range(5):
        print(next(gen))

def create_datagen(TOP_PATH=TOP_PATH): 
    '''
    '''
    length_t, length_y = get_lengths(TOP_PATH)
    train_file_names, test_file_names = split_files()

    filter_cols, filter_rows = get_filters(train_file_names)

    gen_train = datagen(length_t, length_y,
                        train_file_names, filter_cols, 
                        filter_rows, batch_size=16)

    gen_test = datagen(length_t, length_y,
                       test_file_names, filter_cols, 
                       filter_rows, batch_size=16)

    return gen_train, gen_test

def split_files():
    df_names = pd.read_csv(TOP_PATH + LABEL_FILE).iloc[:,0]

    #replace .avi with .csv
    df_names = df_names.apply(lambda row: row[:-4] + '.csv')
    return train_test_split(df_names, test_size=0.2, random_state=42)
    

def datagen(length_t, 
            length_y,
            file_names,
            filter_cols  = 5 , 
            filter_rows  = 0 , 
            batch_size   = 16 
            ):

    '''
    Datagenerator for bus video csv files
    Arguments: 
        length_t: 
        length_y: 
        file_names: 
        batch_size:

    yields: Batch of samples
    '''

    batch_index = 0

    x_batch = np.zeros(shape=(batch_size,
                              length_t - filter_rows * 2,
                              length_y - filter_cols * 2 ))
    y_batch = np.zeros(shape=(batch_size, 1))

    while True:
        df_y = pd.read_csv(TOP_PATH + LABEL_FILE, header=None, names=LABEL_HEADER)
        #TODO shuffle file_names
        for file_name in file_names: 

            df_x = get_features(file_name)
            label = get_entering(file_name, df_y)

            if (df_x is None) or (label is None): 
                continue

            df_x = clean_ends(df_x, del_leading=filter_cols, del_trailing=filter_cols)

            assert df_x.shape[0] == (length_t - 2 * filter_rows)\
               and df_x.shape[1] == (length_y - 2 * filter_cols)

            x_batch[batch_index,:,:] = df_x
            y_batch[batch_index] = label
            batch_index += 1

            # Shape for x must be 3D [samples, timesteps, features] and numpy arrays
            if batch_index == batch_size:
                batch_index = 0
                yield (x_batch, y_batch)
            
def get_filters(file_names): 
    #TODO: Implement. Now dummy function 
    return FILTER_COLS, FILTER_ROWS


def get_features(file_name): 
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


def get_lengths(TOP_PATH=TOP_PATH):
    '''
    returns: Number of timesteps, number of features (columns)
    '''

    for root, _, files in os.walk(TOP_PATH): 
        for file_name in files:
            if file_name[-4:] == '.csv' and not ('label' in file_name):
                full_path = os.path.join(root, file_name)
                df = pd.read_csv(full_path, header=None)
                return df.shape[0], df.shape[1]

def get_cleaned_lengths(filter_cols=FILTER_COLS, filter_rows=FILTER_ROWS, **kwargs):
    '''
    '''

    timestep_num, feature_num = get_lengths(**kwargs)
    return timestep_num - (2 * filter_rows), feature_num - (2 * filter_cols)

if __name__ == '__main__':
    main()