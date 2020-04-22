from os.path import join
from shutil import copyfile
import os



if __name__ == '__main__':
    """
    Rename all videos DATA: video_name/frame_num --> images/video_name+_+frame_num 
    """

    out = r'/home/workplace/garage/out/images'
    videos_path = r'/home/general_vol/visDrone/VisDrone2019-VID-train/sequences'
    videos_list = [f for f in os.listdir(videos_path)]

    for idx, video in enumerate(videos_list):
        print(fr'({idx+1}|{len(videos_list)}) current video: {video}')
        frames = [f for f in os.listdir(join(videos_path, video))]
        for frame in frames:
            src = join(videos_path, video, frame)
            dst = join(out, video+'_'+frame)
            copyfile(src, dst)