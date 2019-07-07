import cv2
import os
import sys
import argparse
import numpy as np


def main(args):
    videos_stream = [cv2.VideoCapture(video_path) for video_path in args.video_path]
    cv2.namedWindow('Comparing Videos', cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Comparing Videos", 1200, 1200)
    delay = args.delay

    while True:
        data = [video_stream.read() for video_stream in videos_stream]
        rets = [i[0] for i in data]
        frames = [i[1] for i in data]

        # If any of "rets" is None
        if not all(rets):
            break

        frames = np.hstack(frames)
        cv2.imshow("Comparing Videos", frames)
        if cv2.waitKey(delay) == ord("q"):
            break


def parse_arguments(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--video_path', type=str, nargs='+',
        help='Path to all the videos, separate by space (i.e. \' \').')
    parser.add_argument('--delay', type=int, default=60,
        help='Number of miliseconds to delay between each frame.')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
