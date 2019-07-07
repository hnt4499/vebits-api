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
import cv2
import imutils
import numpy as np
import tensorflow as tf
import sys
import argparse
import datetime

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from tf_utils import bbox_util, detector_util, labelmap_util, xml_util, im_util
from utils import visualization_utils as vis_util


FONT = cv2.FONT_HERSHEY_SIMPLEX
CONFIDENCE_THRESHOLD = 0.5


def get_classes(class_to_be_detected, labelmap_dict):
    if class_to_be_detected == "all":
        return "all"
    else:
        return [labelmap_dict[item] for item in class_to_be_detected.split(',')]


def process_frame(frame,
                  tensors,
                  category_index,
                  labelmap_dict,
                  cls,
                  confidence_threshold):
    labelmap_dict_inverse = labelmap_util.get_label_map_dict_inverse(labelmap_dict)
    (boxes, scores, classes) = detector_util.detect_objects(frame, tensors)

    frame_height, frame_width = frame.shape[:2]
    fi, boxes_filtered = bbox_util.filter_boxes(boxes=boxes,
                                                scores=scores,
                                                classes=classes,
                                                cls=cls,
                                                confidence_threshold=confidence_threshold,
                                                img_size=(frame_height, frame_width),
                                            )

    classes_filtered = classes[fi]
    if classes_filtered.shape[0] != 0:
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            boxes[fi],
            classes_filtered.astype(np.int32),
            scores[fi],
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=confidence_threshold)

    bboxes = []
    for i in range(boxes_filtered.shape[0]):
        bboxes.append(bbox_util.BBox(
                             labelmap_dict_inverse[classes_filtered[i]],
                             boxes_filtered[i]
                          )
                     )

    return frame, bboxes


def main(args):
    category_index = labelmap_util.load_category_index(args.labelmap_path, args.num_classes)
    labelmap_dict = labelmap_util.get_label_map_dict(args.labelmap_path)
    tensors = detector_util.load_inference_graph(args.inference_graph_path)
    class_to_be_detected = get_classes(args.class_to_be_detected, labelmap_dict)

    dual = False if args.inference_graph_path_2 is None else True
    save_dir, video_name = os.path.split(args.output_path)
    name, _ = os.path.splitext(video_name)

    if dual:
        category_index_2 = labelmap_util.load_category_index(args.labelmap_path_2, args.num_classes_2)
        labelmap_dict_2 = labelmap_util.get_label_map_dict(args.labelmap_path_2)
        tensors_2 = detector_util.load_inference_graph(args.inference_graph_path_2)
        class_to_be_detected_2 = get_classes(args.class_to_be_detected_2, labelmap_dict_2)

    # Open video file
    video = cv2.VideoCapture(args.benchmark_video_path)

    beginning = True
    count = 0
    scale = args.scale
    export_xml = args.export_xml

    rotate = args.rotate
    if rotate % 90 != 0:
        raise ValueError("Invalid value for \'rotate\'")
    else:
        rotate = rotate % 360

    while True:
        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        ret, frame = video.read()
        count += 1
        if not ret:
            break
        if beginning:
            if rotate == 90 or rotate == 270:
                frame_width, frame_height = frame.shape[:2]
            else:
                frame_height, frame_width = frame.shape[:2]

            if scale is not None:
                frame_width = int(frame_width * scale)
                frame_height = int(frame_height * scale)
            else:
                frame_width = args.frame_width
                frame_height = args.frame_height

            if dual: out_size = (frame_width * 2, frame_height)
            else: out_size = (frame_width, frame_height)

            out = cv2.VideoWriter(args.output_path,
                                  cv2.VideoWriter_fourcc(*"mp4v"),
                                  20,
                                  out_size
                            )
            beginning = False

        if rotate is not None:
            frame = imutils.rotate_bound(frame, rotate)

        frame = im_util.resize_padding(frame, (frame_height, frame_width))

        if dual:
            frame_2 = frame.copy()
            frame_2, bboxes_2 = process_frame(frame_2,
                                              tensors_2,
                                              category_index_2,
                                              labelmap_dict_2,
                                              class_to_be_detected_2,
                                              CONFIDENCE_THRESHOLD)

            img_save_name = "{}_2_{}.jpg".format(name, count)
            img_save_path = os.path.join(save_dir, "xml_2", img_save_name)
            if export_xml:
                xml_util.create_xml_file(
                    img_save_path,
                    frame_width,
                    frame_height,
                    bboxes_2,
                )

        frame, bboxes = process_frame(frame,
                                      tensors,
                                      category_index,
                                      labelmap_dict,
                                      class_to_be_detected,
                                      CONFIDENCE_THRESHOLD)

        img_save_name = "{}_1_{}.jpg".format(name, count)
        img_save_path = os.path.join(save_dir, "xml_1", img_save_name)
        if export_xml:
            xml_util.create_xml_file(
                img_save_path,
                frame_width,
                frame_height,
                bboxes,
            )

        cv2.putText(frame, str(count), (20, 50), FONT, 2, (0, 255, 0), 2, cv2.LINE_AA)

        if dual:
            frame = np.concatenate([frame, frame_2], axis=1)
        # All the results have been drawn on the frame, so it's time to display it.
        out.write(frame)

    print("Successfully saved the annotated video.")

    # Clean up
    video.release()
    out.release()
    cv2.destroyAllWindows()


def parse_arguments(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('benchmark_video_path', type=str,
        help='Path to the benchmarking video.')
    parser.add_argument('output_path', type=str,
        help='Path to the output video.')


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

    parser.add_argument('--rotate', type=int, default=0,
        help='Degree to rotate the video.')
    parser.add_argument('--export_xml', action="store_true",
        help='Whether to export the xml files for each frame.')
    parser.add_argument('--scale', type=float, default=None,
        help='Scale to resize the images. If specified, \
        frame_width and frame_height will not be used.')
    parser.add_argument('--frame_width', type=int, default=640,
        help='Destination frame width.')
    parser.add_argument('--frame_height', type=int, default=480,
        help='Destination frame height.')


    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
