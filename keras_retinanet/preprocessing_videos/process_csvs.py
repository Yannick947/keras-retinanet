import os
import csv
from keras_retinanet.preprocessing_videos.process_download import move_files

PCDS_DIR = 'C:/Users/Yannick/OneDrive/Dokumente/Python/PCDS_videos/downloads/'
DESTINATION_DIR = 'C:/Users/Yannick/OneDrive/Dokumente/Python/PCDS_videos/pcds_dataset/'

def main(): 
    move_videos(PCDS_DIR + file_name, DESTINATION_DIR)


def get_destination(full_path_file, destination_dir):
    '''
    Returns the exact destination of the file where it shall be placed. 

    Arguments: 
        full_path_file: Full absolute path of the fall which shall be moved
        destination_dir: Top level folder where files are placed in

        returns: The exact absolute path to the directory within the destination_dir
        where the file shall be placed
    '''
    
    if 'label' in full_path_file: 
        return destination_dir + full_path_file[full_path_file.find('\\'):].replace('\\', '_')

    elif 'back' in full_path_file: 
        return destination_dir + 'back_out' + full_path_file[find_nth(full_path_file, '\\', 2):]

    elif 'front' in full_path_file: 
        return destination_dir + 'front_in' + full_path_file[find_nth(full_path_file, '\\', 2):]


def find_nth(string, search, n):
    '''
    Find the start poisiton of the nth character in a given string

    Arguments: 
        string: String in which will be searched
        search: charcater which will be searched in string
        n: The nth number of the string which shall be found

        returns: start position of nth char in given string, -1 if this 
        amount of strings doesnt exist in string
    '''

    start = string.find(search)
    while start >= 0 and n > 1:
        start = string.find(search, start+len(search))
        n -= 1
    return start

if __name__ == "__main__":
 main()


