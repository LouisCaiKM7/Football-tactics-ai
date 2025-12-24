import os.path

from VideoProcessor.video_cutter import VideoCutter


def main():
    # test_video_cutter()
    test_video_selector()

def test_video_cutter():
    test_path = r"E:\0_projects\00_football_system\testing_videos\test_01.mp4"
    video_cutter = VideoCutter("output_video")
    video_cutter.cut_video(test_path, start_time=0.0, end_time=1.0)



if __name__ == '__main__':
    main()