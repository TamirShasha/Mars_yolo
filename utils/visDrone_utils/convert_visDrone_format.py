import numpy as np
import os
import cv2
import sys
from os.path import join

CLASSES_KEY_BY_NAME = {'ignored regions': 0,
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
                       'others': 11,
                       'remove': -1}

CLASSES_NAME_BY_KEY = {val: key for key, val in CLASSES_KEY_BY_NAME.items()}

CLASSES_TRANSLATOR = {'ignored regions': 'drop',
                      'pedestrian': 'person',
                      'people': 'person',
                      'bicycle': 'drop',
                      'car': 'vehicle',
                      'van': 'vehicle',
                      'truck': 'vehicle',
                      'tricycle': 'vehicle',
                      'awning - tricycle': 'vehicle',
                      'bus': 'vehicle',
                      'motor': 'vehicle',
                      'others': 'vehicle'}

CLASSES_MARS_KEYS = {'person': 0,
                     'vehicle': 1,
                     'drop': 2}


def sub_classes_translator(cls):
    """
    reformat visdom class (class id) to MARS class (class id)
    :param cls: visDrone format
    :return: [person, vehicle, drop]
    """

    class_name = CLASSES_NAME_BY_KEY[cls]
    translation_name = CLASSES_TRANSLATOR[class_name]
    translation_key = CLASSES_MARS_KEYS[translation_name]
    return translation_key


def absolute_to_relative(x, w, h):
    x[:, 0] = x[:, 0] / w
    x[:, 1] = x[:, 1] / h
    x[:, 2] = x[:, 2] / w
    x[:, 3] = x[:, 3] / h
    return x


def xywh_to_xcycwchc(tag):
    """
    h = half h, and w = half w
    """
    new_format = tag.copy()

    new_format[:, 2:4] = new_format[:, 2:4] / 2
    new_format[:, 0] = new_format[:, 0] + new_format[:, 2]
    new_format[:, 1] = new_format[:, 1] + new_format[:, 3]

    return new_format


def reformat_video_annotations(path, video_width, video_height):
    # Take only tags of frame 0!!!
    file = open(path, "r+")
    lines = file.readlines()
    lines = [[float(i) for i in line.split(',')] for line in lines]

    frames_num = set([str(int(line[0])) for line in lines])
    frame_tags_map = []

    for frame in frames_num:
        # Filter tags by current frame.
        frame_tags = [line for line in lines if line[0] == float(frame)]

        # Retrieve classes from tags and translate sub-classes value.
        classes_tags = np.array([sub_classes_translator(tag[-3]) for tag in frame_tags])

        # Retrieve only bbox from tags.
        bboxes_array = np.array([tag[2:6] for tag in frame_tags])

        # YOLO bboxes relative.
        relative_bboxes = absolute_to_relative(bboxes_array, video_width, video_height)

        # YOLO bboxes are xcycwchc.
        reformat_bboxes = xywh_to_xcycwchc(relative_bboxes)

        # Round coordinates of bbox.
        reformat_bboxes = np.round_(reformat_bboxes, decimals=6, out=None)

        # Connect bboxes to corresponding class for every tag.
        reformat_tags = np.concatenate((np.expand_dims(classes_tags, axis=0).T, reformat_bboxes),
                                                   axis=1).tolist()
        reformat_tags = [[int(bbox[0])] + bbox[1:] for bbox in reformat_tags]

        # Drop excess classes.
        reformat_bboxes_new_classes = [tag for tag in reformat_tags if tag[0] is not CLASSES_MARS_KEYS['drop']]

        # Insert new format frame tags to the list.
        frame_tags_map.append((frame, reformat_bboxes_new_classes))

    return frame_tags_map


def save_tage_per_frame(frame_tags_map, out, video):
    for (frame, tags) in frame_tags_map:
        path = join(out, video + '_' + '0' * (7 - len(frame)) + frame + '.txt')
        with open(path, 'w') as f:
            for bbox in tags:
                line = ' '.join([str(var) for var in bbox]) + '\n'
                f.write(line)


if __name__ == '__main__':
    """
    Reformat visDrone annotations from source format to COCO format in order to be able to train with ultralytics YOLO.  
    
    Original format:
    <frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
    Bounding box type: x, y, w, h absolute.

    YOLO (COCO) required format:
    <object_category>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>
    Bounding box type: xc, yc, 0.5w, 0.5h relative.

    """
    out = r'/home/workplace/garage/out/labels/'
    annotations_path = r'/home/general_vol/visDrone/VisDrone2019-VID-train/annotations'
    videos_path = r'/home/general_vol/visDrone/VisDrone2019-VID-train/sequences'
    videos = [f for f in os.listdir(videos_path)]
    for idx, video in enumerate(videos):
        print(fr'({idx+1}|{len(videos)}) current video: {video}')
        # Retrieve annotation file.
        video_tags_path = join(annotations_path, video + '.txt')
        if not os.path.isfile(video_tags_path):
            continue

        # Retrieve first frame for dimension
        first_video_frame = os.listdir(join(videos_path, video))[0]
        (h, w, _) = cv2.imread(join(videos_path, video, first_video_frame)).shape

        # Reformat video tags, from Web source to COCO format.
        frame_tags_map = reformat_video_annotations(video_tags_path, w, h)
        # save annotations file for each frame in shared directory
        save_tage_per_frame(frame_tags_map, out, video)
