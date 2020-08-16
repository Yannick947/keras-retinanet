import numpy as np
import urllib
import os
import csv
import time
from zipfile import ZipFile

PCDS_DIR = 'E:/pcds_downloads/'
DESTINATION_DIR = 'E:/pcds_dataset/'

def main(): 
    for file_name in os.listdir(PCDS_DIR):
        extract_zip(file_name, PCDS_DIR)

    for file_name in os.listdir(PCDS_DIR):
        remove_depth_videos(PCDS_DIR + file_name)
        for root, _, files in os.walk(PCDS_DIR + file_name):
            for file_name in files: 
                full_path_file = os.path.join(root, file_name)

                #Get destination path for file and move it
                destination_path = get_destination(full_path_file, DESTINATION_DIR)
                move_files(full_path_file, DESTINATION_DIR, destination_path)


def extract_zip(file_name, file_dir):
    '''
    Extract given zip folder if it is 
    '''

    if file_name[-4:] == '.zip':
        with ZipFile(os.path.join(file_dir, file_name), 'r') as zipObj:
            zipObj.extractall(os.path.join(file_dir, file_name[0:-4]))


def remove_depth_videos(video_dir):
    '''
    Romve all depth videos from directory and subdirectories

    Arguments: 
        video_dir: Directory which shall be searched for depth videos
    '''

    for root, _, files in os.walk(video_dir):
        for name in files: 
            if 'Depth' in name: 
                os.remove(root + '/' + name.replace('\\','/'))
                print('Removed: ', name)
            
def move_files(full_path_file, destination_dir, destination_path, filter_datatypes=None):
    '''
    Move files to another folder. Folder structure
    remains the same and videos stay at their place, label files are
    placed at the destination directory. 

    Arguements: 
        full_path_file: Path to the file which shall be moved
        destination_dir: Directory where files shall be placed in 
        destination_path: Exact path where file shall be placed including the name of the file
        filter_datatypes: !!List!! of datatypes which shall be considered (e.g '.csv', '.npy')

    '''

    existing_files = []
    for _, _, files in os.walk(destination_dir):
        existing_files.extend(files)

    if (filter_datatypes == None) or (any(filter_d in full_path_file for filter_d in filter_datatypes)):
        os.replace(full_path_file, destination_path)


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


