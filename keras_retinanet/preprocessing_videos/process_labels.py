import pandas as pd
import os 

'''
Label files from original dataset have following structure: 
DepthVideoName, EnteringNumber, ExitingNumber, VideoType

DepthVideoName: the depth video name
EnteringNumber: the number of people entering the bus
ExitingNumber: the number of people exiting the bus
VideoType: the video type. There are 4 video types represented by the index (0: N-C-, 1: N-C+, 2: N+C-, 3: N+C+)

'''

HEADER = ['file_name', 'entering', 'exiting', 'video_type']
TOP_PATH = '/content/drive/My Drive/person_detection/bus_videos/pcds_dataset/'

def main():

    print('Has to be tested again when new data is available')
    for file_name in os.listdir(TOP_PATH): 
        if 'label' in file_name and not ('crowd' in file_name) and not (file_name == 'labels_united.csv'):
            df_labels = load_labels(TOP_PATH)

            labels_singlefile = get_labels(TOP_PATH, file_name)
            labels_singlefile = correct_path(labels_singlefile, file_name)
            
            unite_labels(TOP_PATH, df_labels, labels_singlefile)

def unite_labels(top_path, df_labels, labels_singlefile):
    '''
    Add content of all existing label.txt files to the labels_united.csv file

    Arguments: 
        top_path: Path where labels_united.csv must be placed and where shall 
        be searched for label.txt files 
    '''

    df_labels = pd.concat([df_labels, labels_singlefile],
                            axis=0).drop_duplicates(subset='file_name')

    df_labels.to_csv(top_path + '/labels_united.csv', header = None, index=None)
                

def correct_path(df_labels, file_name): 
    ''' 
    Due to the change of the folder structure it has to be added the 'front_in' or 'back_out' dir

    Arguments: 
        df_labels: The dataframe with the labels
        file_name: The filename of the label file 

        returns: Corrected df of labels 

        raises: ValueError if there is no back or front in the label path
    '''
    
    if 'front' in file_name: 
        df_labels['file_name'] = df_labels['file_name'].apply(lambda row: 'front_in' + row[1:-4] + '.csv')
        return df_labels

    elif 'back' in file_name: 
        df_labels['file_name'] = df_labels['file_name'].apply(lambda row: 'back_out' + row[1:-4] + '.csv')
        return df_labels
    
    else: 
        raise ValueError('File {} not a valid label file'.format(file_name))


def load_labels(top_path):
    '''
    Checks in top path if there is an already existing 'labels_united.csv', 
    otherwise return an empty df with correct header names

    Arguments: 
        top_path: Path where shall be searched for the labels_united.csv file

        returns: The previously stored labels_united file as pandas Dataframe, 
                 and an empty Dataframe if no such file exists
    '''

    files = os.listdir(top_path)

    if 'labels_united.csv' in files: 
        return pd.read_csv(top_path + '/labels_united.csv', names=HEADER)

    else: 
        return pd.DataFrame(columns=HEADER) 


def get_labels(root, file_name):
    '''
    Gets the labels of a single txt file

    Arguments: 
        file_name: The name of the file which shall be returned
    '''

    full_path = os.path.join(root, file_name)
    with open(full_path, mode='r') as label:
        lines_after_header = label.readlines()[4:]
        lines_splitted = [i.split() for i in lines_after_header]
        assert len(lines_splitted[1]) == 4
        return pd.DataFrame(lines_splitted, columns=HEADER)

if __name__ == '__main__':
    main()