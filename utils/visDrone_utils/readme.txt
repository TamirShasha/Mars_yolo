visDrone-Dataset creation steps:

1. Download "Task 2: Object Detection in Videos" *.zip from https://github.com/VisDrone/VisDrone-Dataset
2. Unzip to folders: train_source, val_source, test_source, challenge_source
2. Make folders (use the following hierarchy):
                            visDrone, visDrone/images, visDrone/images/train, visDrone/images/val,
                            visDrone/labels, visDrone/labels/train, visDrone/labels/val.
3. In order to convert labels for yolo_Mars, run:
   python3 convert_visDrone_format.py  train_source/annotations train_source/sequences  visDrone/labels/train
   python3 convert_visDrone_format.py  val_source/annotations val_source/sequences  visDrone/labels/val
   python3 convert_visDrone_format.py  test_source/annotations test_source/sequences  visDrone/labels/test
4. In order to rearrange videos name (make copy with new names) for yolo_Mars, run:
   python3 rearrange_data_videos.py train_source/sequences visDrone/images/train
   python3 rearrange_data_videos.py val_source/sequences visDrone/images/val
   python3 rearrange_data_videos.py test_source/sequences visDrone/images/test

