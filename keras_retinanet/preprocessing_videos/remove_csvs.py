import os 
import sys 

def main():
    if len(sys.argv) != 2: 
        print('One argument must be given, which indicates the parent directory where\
               all csv files inside shall be deleted.')
               
    top_dir = sys.argv[1]
    for root, _, files in os.walk(top_dir): 
        for file_name in files: 
            if file_name[-4:] == '.csv':
                full_file_path = os.path.join(root, file_name)
                os.remove(full_file_path)

if __name__ == '__main__':
    main()