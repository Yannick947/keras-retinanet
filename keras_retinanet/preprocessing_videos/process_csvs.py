import os
from keras_retinanet.preprocessing_videos.process_download import move_files

PCDS_DIR = '/content/drive/My Drive/person_detection/pcds_dataset_detections/pcds_dataset/'
DESTINATION_DIR = '/content/drive/My Drive/person_detection/pcds_dataset_detections/pcds_dataset_detected/'

def main(): 
    for root, _, files in os.walk(PCDS_DIR):
        for file_name in files: 
            full_path_file = os.path.join(root, file_name)
            destination_path = get_destination(full_path_file, DESTINATION_DIR, file_name)
            move_files(full_path_file, DESTINATION_DIR, destination_path, filter_datatypes=['.npy', '.csv'])


def get_destination(full_path_file, destination_dir, file_name):
    '''
    Returns the exact destination of the file where it shall be placed. 

    Arguments: 
        full_path_file: Full absolute path of the fall which shall be moved
        destination_dir: Top level folder where files are placed in
        file_name: Name of the file

        returns: The exact absolute path to the directory within the destination_dir
        where the file shall be placed
    '''
            
    if 'label' in full_path_file: 
        return destination_dir + 'pcds_dataset' + \
               full_path_file[full_path_file.find('pcds_dataset') + 12:].replace('\\', '_').replace('/', '_')

    elif 'back' in full_path_file: 
        return destination_dir + full_path_file[full_path_file.find('back_out'):]

    elif 'front' in full_path_file: 
        return destination_dir + full_path_file[full_path_file.find('front_in'):]

if __name__ == "__main__":
 main()


