import numpy as np
import os
import cv2
import sys

CLASSES_MARS_NAMES = {'person': 0,
                     'vehicle': 1}
CLASSES_MARS_KEYS = {val: key for key, val in CLASSES_MARS_NAMES.items()}


def draw_visDrone(frame, tags, frame_num):
    # Take only tags of frame frame_num
    file1 = open(tags, "r+")
    lines = file1.readlines()
    frame_tags = [[float(i) for i in line.split(' ')] for line in lines]

    # load frame 0
    frame = cv2.imread(frame)
    (h, w, _) = frame.shape

    for tag in frame_tags:
        cls = str(tag[0])
        bbox = np.array(tag[0:5])

        x = int(w * (bbox[1] - bbox[3]))
        y = int(h * (bbox[2] - bbox[4]))
        x2 = int(w * (bbox[1] + bbox[3]))
        y2 = int(h * (bbox[2] + bbox[4]))

        frame = cv2.putText(frame, cls, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)

    return frame


if __name__ == '__main__':
    """
    Use this script to draw tags of some frame in visDrone.
    important note: this scrip works with original format of visDrone.
    """
    frame = r'/home/general_vol/visDrone/VisDrone2019-VID-train/sequences/uav0000071_03240_v/0000001.jpg'
    tags = r'/home/workplace/garage/out/labels/uav0000071_03240_v_0000001.txt'
    frame_num = 1
    out = r'/home/workplace/garage/out/file.jpg'
    # for terminal control
    if len(sys.argv) == 5:
        frame = sys.argv[1]
        tags = sys.argv[2]
        out = sys.argv[3]

    tag_image = draw_visDrone(frame, tags, frame_num)
    cv2.imwrite(out, tag_image)
