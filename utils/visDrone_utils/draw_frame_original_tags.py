import numpy as np
import os
import cv2
import sys


def draw_coco(frame, tags):
    # take tags of frame 0
    file = open(tags, "r+")
    lines = file.readlines()
    lines = [[float(i) for i in line.split(' ')] for line in lines]

    frame = cv2.imread(frame)
    (h, w, _) = frame.shape

    # classes = {'ignored regions': 0, 'pedestrian': 1, 'people': 2, 'bicycle': 3, 'car': 4, 'van': 5, 'truck': 6,
    #            'tricycle': 7, 'awning - tricycle': 8, 'bus': 9, 'motor': 10, 'others': 11}
    # classes = {val: key for key, val in classes.items()}

    for tag in lines:
        cls = str(tag[0])
        bbox = np.array(tag[0:5])

        x = int(w * (bbox[1] - bbox[3]))
        y = int(h * (bbox[2] - bbox[4]))
        x2 = int(w * (bbox[1] + bbox[3]))
        y2 = int(h * (bbox[2] + bbox[4]))

        # x, y, x2, y2 = int(bbox[0] * w), int(bbox[1] * h), int(bbox[2] * w), int(bbox[3] * h)

        frame = cv2.putText(frame, cls, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)

    return frame


def draw_visDrone(frame, tags, frame_num):
    # Take only tags of frame frame_num
    file1 = open(tags, "r+")
    lines = file1.readlines()
    lines = [[int(i) for i in line.split(',')] for line in lines]
    frame_tags = [line for line in lines if line[0] == frame_num]

    # load frame 0
    frame = cv2.imread(frame)

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
    classes = {val: key for key, val in classes.items()}

    for tag in frame_tags:
        cls = classes[tag[-3]]
        bbox = tag[2:6]
        x, y, x2, y2 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]

        frame = cv2.putText(frame, cls, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)

    return frame


if __name__ == '__main__':
    """
    Use this script to draw tags of some frame in visDrone.
    important note: this scrip works with original format of visDrone.
    """
    frame = r'/home/general_vol/visDrone/VisDrone2019-VID-train/sequences/uav0000143_02250_v/0000001.jpg'
    tags = r'/home/general_vol/visDrone/VisDrone2019-VID-train/annotations/uav0000143_02250_v.txt'
    frame_num = 1
    out = r'/home/workplace/garage/out/file.jpg'
    # for terminal control
    if len(sys.argv) == 5:
        frame = sys.argv[1]
        tags = sys.argv[2]
        out = sys.argv[3]

    tag_image = draw_visDrone(frame, tags, frame_num)
    cv2.imwrite(out, tag_image)
