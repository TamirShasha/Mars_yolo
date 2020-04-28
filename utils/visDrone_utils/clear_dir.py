import sys
import os
import shutil


def remove(folder, f):
    print("remove ", f)
    os.remove(os.path.join(folder, f))


if __name__ == '__main__':
    """
    Use this script if you wish to remove some directory contents.
    """

    if len(sys.argv) < 2:
        dir_for_cleaning = r'/home/workplace/garage/out/images'
    else:
        dir_for_cleaning = sys.argv[1]

    filelist = [f for f in os.listdir(dir_for_cleaning)]
    for f in filelist:
        if os.path.isdir(dir_for_cleaning + "/" + f):
            folder = dir_for_cleaning + "/" + f
            [remove(folder, cc) for cc in os.listdir(folder)]
            shutil.rmtree(folder)
        else:
            remove(dir_for_cleaning, f)