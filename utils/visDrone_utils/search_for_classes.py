from os.path import join
import numpy as np
import os
import cv2
import sys




if __name__ == '__main__':
    """
    This script helps you to find videos and frames which contain shown-up of requested class.
    
    data path: /home/general_vol/visDrone/VisDrone2019-VID-train:
    1. sequences/vid_name --> all frames files (named by frame-num) .jpg
    2. annotations/ --> all videos tags file .txt (every txt file contains tags of all frames) 
            
    """

    data_path = r'/home/general_vol/visDrone/VisDrone2019-VID-train/annotations'
    requested_class = 'ignored regions'

    classes = {'ignored regions': 0,
               'pedestrian': 1,
               'people': 2,
               'bicycle': 3,
               'car': 4,
               'van': 5,
               'truck': 6,
               'tricycle': 7,
               'awning - tricycle': 8,
               'bus': 9,
               'motor': 10,
               'others': 11}

    requested_class = classes[requested_class]
    classes = {val: key for key, val in classes.items()}

    filelist = [f for f in os.listdir(data_path)]
    satisfied_video = list()
    total_num_of_frames = 0
    num_of_appearance_frames = 0
    for idx, file in enumerate(filelist):
        print(fr'({idx}|{len(filelist)})')
        p_file = join(data_path, file)
        file_content = open(p_file, "r+")
        lines = file_content.readlines()
        lines = [[int(i) for i in line.split(',')] for line in lines]
        total_num_of_frames += len(set([line[0] for line in lines]))
        satisfied_frames = set([line[0] for line in lines if line[-3] == requested_class])
        num_of_appearance_frames += len(satisfied_frames)
        if satisfied_frames:
            satisfied_video.append((file, satisfied_frames))
            # print(fr'{file}: {len(satisfied_frames)} frames')
            print(fr'{file}: {satisfied_frames}')

    print(fr'\nTotal frames: {total_num_of_frames}\nframes with appearance: {num_of_appearance_frames} ')
    # for v in satisfied_video:
    #     print(v)
    x = 0
