######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/20/18
# Description:
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a webcam feed.
# It draws boxes and scores around the objects of interest in each frame from
# the webcam.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.


# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import argparse

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Import utilites
from tf_utils import detector_util, labelmap_util, im_util
from utils import visualization_utils as vis_util


def main(args):
    category_index = labelmap_util.load_category_index(args.labelmap_path, args.num_classes)
    tensors = detector_util.load_inference_graph(args.inference_graph_path)

    dual = False if args.inference_graph_path_2 is None else True
    if dual:
        category_index_2 = labelmap_util.load_category_index(args.labelmap_path_2, args.num_classes_2)
        tensors_2 = detector_util.load_inference_graph(args.inference_graph_path_2)

    frame_width = args.frame_width
    frame_height = args.frame_height
    # Set display parameter.
    display_width = frame_width * 2 if dual else frame_width
    display_height = frame_height
    # Set ouput video parameter.
    output_width = args.output_width
    output_height = args.output_height
    # Initialize webcam feed
    cam = cv2.VideoCapture(0)
    ret = cam.set(3, output_width)
    ret = cam.set(4, output_height)

    if args.output_video is not None:
        save = True
        video = cv2.VideoWriter(args.output_video,
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                20,
                                (output_width, output_height))
    else:
        save = False

    while(True):

        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        ret, frame = cam.read()
        display = im_util.resize_padding(frame, (frame_height, frame_width))
        display_2 = display.copy()

        if save:
            video.write(frame)

        boxes, scores, classes = detector_util.detect_objects(display, tensors)
        vis_util.visualize_boxes_and_labels_on_image_array(
            display,
            boxes,
            classes.astype(np.int32),
            scores,
            category_index,
            use_normalized_coordinates=True,
            line_thickness=6,
            min_score_thresh=0.6)

        if dual:
            boxes_2, scores_2, classes_2 = detector_util.detect_objects(display_2, tensors_2)
            vis_util.visualize_boxes_and_labels_on_image_array(
                display_2,
                boxes_2,
                classes_2.astype(np.int32),
                scores_2,
                category_index_2,
                use_normalized_coordinates=True,
                line_thickness=6,
                min_score_thresh=0.6)

            display = np.hstack([display, display_2])

        cv2.imshow('Object detector', display)

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    # Clean up
    cam.release()
    cv2.destroyAllWindows()


def parse_arguments(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('inference_graph_path', type=str,
        help='Path to the first inference graph.')
    parser.add_argument('labelmap_path', type=str,
        help='Path to the label map of the first model.')
    parser.add_argument('num_classes', type=int,
        default=2, help='Number of classes the first model can detect.')
    parser.add_argument('class_to_be_detected', type=str,
        help='The class(es) to be predicted. If multiple, \
        separate each class with comma (e.g \'phone,not_phone\'). \
        Specify \'all\' to use all classes.')

    parser.add_argument('-i', '--inference_graph_path_2', type=str, default=None,
        help='Path to the second inference graph.')
    parser.add_argument('-l', '--labelmap_path_2', type=str, default=None,
        help='Path to the label map of the second model.')
    parser.add_argument('-n', '--num_classes_2', type=int, default=None,
        help='Number of classes the second model can detect.')
    parser.add_argument('-c', '--class_to_be_detected_2', type=str, default=None,
        help='The class(es) to be predicted. If multiple, \
        separate each class with comma (e.g \'phone,not_phone\')')

    parser.add_argument('--frame_width', type=int, default=640,
        help='Display frame width.')
    parser.add_argument('--frame_height', type=int, default=480,
        help='Display frame height.')

    parser.add_argument('--output_width', type=int, default=640,
        help='Output frame width.')
    parser.add_argument('--output_height', type=int, default=480,
        help='Output frame height.')

    parser.add_argument('--output_video', type=str, default=None,
        help='Path to output the video.')


    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
