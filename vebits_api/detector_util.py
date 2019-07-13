# Utilities for object detector.

import os
import sys
from threading import Thread
from datetime import datetime
from collections import defaultdict

import numpy as np
import tensorflow as tf
import cv2

from . import labelmap_util


# Load a frozen infrerence graph into memory
def load_inference_graph(inference_graph_path):

    # load frozen tensorflow model into memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(inference_graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    tensors = {
        "sess": sess,
        "image_tensor": image_tensor,
        "detection_boxes": detection_boxes,
        "detection_scores": detection_scores,
        "detection_classes": detection_classes,
        "num_detections": num_detections
    }

    return tensors


def load_tensors(inference_graph_path, labelmap_path, num_classes=None):
    tensors = load_inference_graph(inference_graph_path)
    labelmap_dict = labelmap_util.get_label_map_dict(labelmap_path)
    labelmap_dict_inverse = labelmap_util.get_label_map_dict_inverse(labelmap_dict)
    # If `num_classes` is not specified, it will be inferred from labelmap.
    if num_classes is None:
        num_classes = len(labelmap_dict)
    category_index = labelmap_util.load_category_index(labelmap_path, num_classes)

    tensors["labelmap_dict"] = labelmap_dict
    tensors["labelmap_dict_inverse"] = labelmap_dict_inverse
    tensors["category_index"] = category_index

    return tensors


# Actual detection .. generate scores and bounding boxes given an image
def detect_objects(img, tensors):
    sess = tensors["sess"]
    image_tensor = tensors["image_tensor"]
    detection_boxes = tensors["detection_boxes"]
    detection_scores = tensors["detection_scores"]
    detection_classes = tensors["detection_classes"]
    num_detections = tensors["num_detections"]

    dims = img.ndim
    if dims == 3:
        img = np.expand_dims(img, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
            detection_classes, num_detections],
        feed_dict={image_tensor: img})

    if dims == 3:
        return np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)
    else:
        return boxes, scores, classes


# Code to thread reading camera input.
# Source : Adrian Rosebrock
# https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
class WebcamVideoStream:
    def __init__(self, src, width, height):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def size(self):
        # return size of the capture device
        return self.stream.get(3), self.stream.get(4)

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
