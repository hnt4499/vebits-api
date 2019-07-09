from tf_utils import others_util, bbox_util, vis_util, labelmap_util
import pandas as pd
import numpy as np
import os
import argparse
import sys
import cv2
from shutil import copy

def main(args):
    df = pd.read_csv(args.csv_path)
    img_list = df.filename.unique()
    index = args.start
    dataset_dir = args.dataset_dir
    incorrect_dir = args.incorrect_dir
    labelmap_dict = labelmap_util.get_label_map_dict(args.labelmap_path)

    cv2.namedWindow('Inspecting Dataset', cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Inspecting Dataset", 800, 800)

    while True:
        img_name = img_list[index]
        img_path = os.path.join(dataset_dir, img_name)

        if not os.path.isfile(img_path):
            print(">>> File not found: {}".format(img_path))
            index += 1
            continue

        img = cv2.imread(img_path)
        bboxes, classes = bbox_util.get_bboxes_array_and_classes(df, img_name)
        vis_util.draw_boxes_on_image(img, bboxes, classes, labelmap_dict)
        vis_util.draw_number(img, index)
        cv2.imshow("Inspecting Dataset", img)

        key = cv2.waitKey(0)
        if key == ord("d"):
            index += 1
        elif key == ord("a"):
            index -= 1
        elif key == ord("w"):
            index += 1
            copy(img_path, os.path.join(incorrect_dir, img_name))
        elif key == ord("q"):
            print(">>> Current index: {}".format(index))
            break


def parse_arguments(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset_dir', type=str,
        help='Directory to the dataset.')
    parser.add_argument('csv_path', type=str,
        help='Path to the csv file.')
    parser.add_argument('labelmap_path', type=str,
        help='Path to the labelmap.')
    parser.add_argument('incorrect_dir', type=str,
        help='Directory to which all the incorrectly \
        labelled images will be saved.')
    parser.add_argument('--start', type=int, default=0,
        help='Starting point. Default=0 (i.e. no image has been processed.)')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
