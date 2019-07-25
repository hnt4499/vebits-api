######## Video Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/16/18
# Description:
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a video.
# It draws boxes and scores around the objects of interest in each frame
# of the video.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import sys
import argparse
import datetime

from tqdm import tqdm
import cv2
import imutils
import numpy as np
import tensorflow as tf

# Import utilites
from vebits_api import bbox_util, detector_util, im_util, others_util
from vebits_api.xml_util import create_xml_file

FONT = cv2.FONT_HERSHEY_SIMPLEX
CONFIDENCE_THRESHOLD = 0.5
IMG_HEIGHT = 480
IMG_WIDTH = 640

DESCRIPTION="""This script loads Object Detection API's model(s) (up to two),
makes predictions on images provided and save annotated images as well as
their labels to *.xml files. Each model can specifically be used to predict
specific classes.
"""


def load_tensors(inference_graph_path,
                 labelmap_path,
                 num_classes,
                 class_to_be_detected):
    """
    This function extends `vebits_api.bbox_util.load_tensors` by providing one
    more metadata: class to be detected.
    """
    tensors = detector_util.load_tensors(
                                inference_graph_path,
                                labelmap_path,
                                num_classes)
    class_to_be_detected = others_util.get_classes(class_to_be_detected,
                                                   tensors["labelmap_dict"])
    tensors["class_to_be_detected"] = class_to_be_detected
    return tensors


def get_filtered_boxes(boxes,
                       scores,
                       classes,
                       class_to_be_detected,
                       labelmap_dict_inverse,
                       confidence_threshold,
                       img_size):
    # Filter unwanted boxes
    boxes, scores, classes = bbox_util.filter_boxes(
                                    boxes=boxes,
                                    scores=scores,
                                    classes=classes,
                                    cls=class_to_be_detected,
                                    confidence_threshold=confidence_threshold,
                                    img_size=img_size)
    # Convert to BBox format
    bboxes = []
    for i in range(boxes.shape[0]):
        bboxes.append(bbox_util.BBox(
                             labelmap_dict_inverse[classes[i]],
                             boxes[i])
                     )
    return bboxes


def process_frame_batch(frames,
                        img_save_paths,
                        num_transform,
                        sequence,
                        tensors,
                        tensors_2,
                        confidence_threshold):

    # Perform augmentation
    if num_transform > 0:
        frames_aug = sequence(images=frames * num_transform)
    else: frames_aug = []
    # Get names for the whole batch
    img_save_paths_aug = (
        others_util.get_batch_names(img_save_paths[i], num_transform)
        for i in range(len(frames))
    )
    for img_aug_names in zip(*img_save_paths_aug):
        img_save_paths.extend(list(img_aug_names))
    # Get all images (un-augmented and augmented ones)
    frames.extend(frames_aug)
    frames = np.array(frames)
    # Perform detection
    boxes, scores, classes = detector_util.detect_objects(frames, tensors)
    if tensors_2 is not None:
        boxes_2, scores_2, classes_2 = detector_util.detect_objects(frames, tensors_2)

    frame_height, frame_width = frames.shape[1:3]
    # Filter out unwanted boxes
    for i in range(boxes.shape[0]):
        bboxes = get_filtered_boxes(
                            boxes=boxes[i],
                            scores=scores[i],
                            classes=classes[i],
                            class_to_be_detected=tensors["class_to_be_detected"],
                            labelmap_dict_inverse=tensors["labelmap_dict_inverse"],
                            confidence_threshold=confidence_threshold,
                            img_size=(frame_height, frame_width),
                        )

        if tensors_2 is not None:
            bboxes_2 = get_filtered_boxes(
                                boxes=boxes_2[i],
                                scores=scores_2[i],
                                classes=classes_2[i],
                                class_to_be_detected=tensors_2["class_to_be_detected"],
                                labelmap_dict_inverse=tensors_2["labelmap_dict_inverse"],
                                confidence_threshold=confidence_threshold,
                                img_size=(frame_height, frame_width),
                            )
            bboxes = bboxes + bboxes_2
        # Generate *.xml files simultaneously
        xml_util.create_xml_file(
            img_save_paths[i],
            frame_width,
            frame_height,
            bboxes,
        )
        # Save images
        im_util.save_imgs(frames, img_save_paths)


def main(args):
    # Load tensors
    tensors = load_tensors(
                    args.inference_graph_path,
                    args.labelmap_path,
                    args.num_classes,
                    args.class_to_be_detected)

    if args.inference_graph_path_2 is None:
        tensors_2 = None
    else:
        tensors_2 = load_tensors(
                        args.inference_graph_path_2,
                        args.labelmap_path_2,
                        args.num_classes_2,
                        args.class_to_be_detected_2)

    img_dirs = args.img_dirs
    output_dirs = args.output_dirs
    batch_size = args.batch_size

    sequence = im_util.create_sequence()
    num_transform = args.num_transform

    num_frame_processed = 0
    num_img_generated = 0
    frames = []
    img_save_paths = []

    for img_dir, output_dir in zip(img_dirs, output_dirs):
        img_list = sorted(os.listdir(img_dir))[args.start:]
        with tqdm(img_list) as t:
            for img_name in t:
                _, ext = os.path.splitext(img_name)
                if ext not in [".jpg", ".png"]:
                    continue

                # Update the variables.
                num_frame_processed += 1
                # Read image and resize to desired size
                img_path = os.path.join(img_dir, img_name)
                img = cv2.imread(img_path)
                img = im_util.resize_padding(img, (IMG_HEIGHT, IMG_WIDTH))

                img_save_paths.append(os.path.join(output_dir, img_name))
                frames.append(img)
                # Wait until batch_size number of frames are grabbed
                if num_frame_processed % batch_size != 0:
                    continue
                # Process grabbed batch
                process_frame_batch(frames=frames,
                                    img_save_paths=img_save_paths,
                                    num_transform=num_transform,
                                    sequence=sequence,
                                    tensors=tensors,
                                    tensors_2=tensors_2,
                                    confidence_threshold=CONFIDENCE_THRESHOLD)

                num_img_generated += len(img_save_paths)
                t.set_postfix(generated=num_img_generated)

                frames = []
                img_save_paths = []
        # At the end of the loop, there might be some images that
        # have not been processed
        if frames != []:
            process_frame_batch(frames=frames,
                                img_save_paths=img_save_paths,
                                num_transform=num_transform,
                                sequence=sequence,
                                tensors=tensors,
                                tensors_2=tensors_2,
                                confidence_threshold=CONFIDENCE_THRESHOLD)

            num_img_generated += len(img_save_paths)

        print('>>> Results: {} images generated to {}'.format(num_img_generated, output_dir))
        # Reset parameters to continue processing with the next folders
        frames = []
        img_save_paths = []
        num_img_generated = 0


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=DESCRIPTION)

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

    parser.add_argument('-v', '--img_dirs', type=str, nargs='+', required=True,
        help='All the directories containing images you want to process,\
        separate by space (i.e. \' \').')
    parser.add_argument('-o', '--output_dirs', type=str, nargs='+', required=True,
        help='Directories to which images and their labels will be saved.')

    parser.add_argument('--batch_size', type=int, default=4,
        help='Number of images to process each loop.')
    parser.add_argument('--num_transform', type=int, default=3,
        help='Number of times to perform augmentation.')
    parser.add_argument('--start', type=int, default=0,
        help='Index to start. Helpful when continuing from previous work.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
