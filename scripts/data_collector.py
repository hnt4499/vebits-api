
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import argparse
import imgaug as ia
import imgaug.augmenters as iaa
import time
import xml.etree.ElementTree as ET

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Some constants
WIDTH = 1280
HEIGHT = 720
CONFIDENCE_THRESHOLD = 0.6


def create_sequence():
    aug = iaa.Sequential([
        iaa.CropAndPad(
            percent=(-0.15, 0.15),
            pad_mode=ia.ALL,
            pad_cval=(0, 128)
        ),
        iaa.Affine(
            scale={"x": (0.85, 1.15), "y": (0.85, 1.15)},
            mode=ia.ALL,
            cval=(0, 255)
        ),
        iaa.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            mode=ia.ALL,
            cval=(0, 255)
        ),
        iaa.Affine(
            rotate=(-15, 15),
            mode=ia.ALL,
            cval=(0, 255)
        ),
        iaa.Affine(
            shear=(-10, 10),
            mode=ia.ALL,
            cval=(0, 255)
        ),
        iaa.Fliplr(0.5),
    ])

    return aug


def save_imgs(imgs, save_paths):
    for save_path, img in zip(save_paths, imgs):
        cv2.imwrite(save_path, img)


def get_names(save_dir, prefix, suffix, num_imgs):
    suffices = [suffix + i for i in range(num_imgs)]
    img_paths = [os.path.join(save_dir, "{}_{:06d}.jpg".format(prefix, suffix)) for suffix in suffices]
    xml_paths = [os.path.join(save_dir, "{}_{:06d}.xml".format(prefix, suffix)) for suffix in suffices]
    return img_paths, xml_paths


def get_and_save_xml(xml_template, img_path, bbox, save_path):
    root = xml_template.getroot()
    _, img_name = os.path.split(img_path)

    root[1].text = img_path
    root[2].text = save_path
    root[6][0].text = "phone"
    for index, coord in enumerate(bbox):
        root[6][4][index].text = str(coord)

    xml_template.write(save_path)


def filter_boxes(boxes, scores, classes, confidence_threshold, img_size):
    height, width = img_size
    fi = (scores > confidence_threshold) * (classes == 3)
    boxes = boxes[fi]

    boxes = boxes * [height, width, height, width]
    boxes[:, 0], boxes[:, 1] = boxes[:, 1].copy(), boxes[:, 0].copy()
    boxes[:, 2], boxes[:, 3] = boxes[:, 3].copy(), boxes[:, 2].copy()

    return np.asarray(boxes, dtype=np.int)


def main(args):

    count = 0
    start = False
    frame_processed = 0

    num_frame = args.num_frame
    num_transform = args.num_transform

    subfolder = args.subfolder
    save_dir = os.path.join(args.save_dir, subfolder)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    labelmap_dir = args.labelmap_dir
    inference_graph_dir = args.inference_graph_dir

    sequence = create_sequence()

    label_map = label_map_util.load_labelmap(labelmap_dir)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=2, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(inference_graph_dir, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    #Font for text
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Initialize webcam feed
    video = cv2.VideoCapture(0)
    ret = video.set(3, WIDTH)
    ret = video.set(4, HEIGHT)

    xml_tree = ET.parse(args.xml_template)

    while(True):

        ret, frame = video.read()

        key = cv2.waitKey(1)
        # Press 'q' to quit
        if key == ord('q'):
            break
        # Press 's' to start
        elif key == ord('s'):
            start = True
            start_time = time.time()

        if not start:
            cv2.putText(frame, "Waiting", (20, 40), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "Press s to start", (20, 70), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Data collector', frame)
            continue

        count += 1
        if count % num_frame == 0:
            count = 0

            frame_processed += 1

            frames = np.array([frame for _ in range(num_transform)], dtype=np.uint8)
            frames = sequence(images=frames)
            frames = np.vstack([frames, np.expand_dims(frame, axis=0)])

            img_paths, xml_paths = get_names(save_dir, subfolder, frame_processed, num_transform + 1)
            save_imgs(frames, img_paths)
            frame_processed += num_transform + 1

            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: frames})

            for i in range(boxes.shape[0]):
                box = filter_boxes(
                    boxes[i],
                    scores[i],
                    classes[i],
                    CONFIDENCE_THRESHOLD,
                    (HEIGHT, WIDTH)
                )
                if box.shape[0] == 0:
                    continue
                get_and_save_xml(xml_tree, img_paths[i], box[0], xml_paths[i])

            text = "{:.1f}s".format(time.time() - start_time)
            cv2.putText(frame, text, (20, 40), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Data collector', frame)


    # Clean up
    video.release()
    cv2.destroyAllWindows()



def parse_arguments(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('save_dir', type=str,
        help='Save directory.')
    parser.add_argument('subfolder', type=str,
        help='Subfolder.')
    parser.add_argument('num_frame', type=int,
        help='Number of frames between two adjacent captures.')
    parser.add_argument('num_transform', type=int,
        help='Number of transform for each image by data augmentation.')
    parser.add_argument('inference_graph_dir', type=str,
        help='Inference graph directory.')
    parser.add_argument('labelmap_dir', type=str,
        help='Labelmap directory.')
    parser.add_argument('xml_template', type=str,
        help='Path to the xml template.')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
