
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import argparse
import imgaug as ia
import imgaug.augmenters as iaa
import datetime
from data_collect import BBox, create_xml_file
from tf_utils import detector_util, bbox_util

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

LABELS = {
    1: 'face',
    2: 'hand',
    3: 'phone'
}
# Font for text
FONT = cv2.FONT_HERSHEY_SIMPLEX


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
        iaa.Fliplr(0.6),
    ])

    return aug


def detect_and_save_imgs(tensors,
                         src_dir,
                         correct_dir,
                         incorrect_dir,
                         img,
                         img_name,
                         img_size,
                         num_transform,
                         cls,
                         count,
                         score_thresh):

    stop = False
    sequence = create_sequence()

    img_height, img_width = img_size

    imgs = np.array([img for _ in range(num_transform)], dtype=np.uint8)
    imgs = sequence(images=imgs)
    imgs = np.vstack([imgs, np.expand_dims(img, axis=0)])

    (boxes, scores, classes) = detector_util.detect_objects(imgs, tensors)

    for i in range(num_transform + 1):

        count += 1
        img = imgs[i]
        display = img.copy()

        _, boxes_filtered = bbox_util.filter_boxes(
                                    boxes=boxes[i],
                                    scores=scores[i],
                                    classes=classes[i],
                                    cls=cls,
                                    confidence_threshold=score_thresh,
                                    img_size=(img_height, img_width)
                                )

        bboxes = []
        # Get each box of the same image.
        for j in range(boxes_filtered.shape[0]):
            bboxes.append(bbox_util.BBox(
                                         LABELS[classes[i, j]],
                                         boxes_filtered[j]
                                    )
                         )

            cv2.rectangle(
                display,
                (box_filtered[0], box_filtered[1]),
                (box_filtered[2], box_filtered[3]),
                (0, 255, 0),
                3,
                1
            )

        cv2.putText(display, str(count), (20, 40), FONT, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Data Collector', display)

        key = cv2.waitKey(0)

        name, ext = os.path.splitext(img_name)
        img_new_name = name + "_{}".format(i) + ext

        if key == ord("d"):
            create_xml_file(
                correct_dir,
                img_new_name,
                os.path.join(correct_dir, img_new_name),
                img_width,
                img_height,
                bboxes
            )

            cv2.imwrite(os.path.join(correct_dir, img_new_name), img)
            print("Correct image saved: ", img_new_name)

        elif key == ord("w"):
            cv2.imwrite(os.path.join(incorrect_dir, img_new_name), img)
            print("Incorrect image saved: ", img_new_name)

        elif key == ord("q"):
            stop = True
            break

        return count, stop


def main(args):

    count = 0
    stop = False
    num_transform = args.num_transform

    cls = args.class_to_be_detected
    score_thresh = args.score_thresh
    width = args.width
    height = args.height

    src_dir = args.src_dir
    correct_dir = args.correct_dir
    incorrect_dir = args.incorrect_dir
    labelmap_path = args.labelmap_path
    inference_graph_path = args.inference_graph_path

    sequence = create_sequence()
    tensors = detector_util.load_inference_graph(inference_graph_path)

    # Fix the size of the displaying window to be 600x600.
    cv2.namedWindow('Data Collector', cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Data Collector", 600, 600)

    stop = False

    if not args.load_video:
        img_list = os.listdir(src_dir)

        for img_name in img_list:
            if not img_name.endswith("jpg"):
                continue
            if stop:
                break

            img = cv2.imread(os.path.join(src_dir, img_name))
            img_size = img.shape[:2]

            count, stop = detect_and_save_imgs(tensors=tensors,
                                               src_dir=src_dir,
                                               correct_dir=correct_dir,
                                               incorrect_dir=incorrect_dir,
                                               img=img,
                                               img_name=img_name,
                                               img_size=img_size,
                                               num_transform=num_transform,
                                               cls=cls,
                                               count=count,
                                               score_thresh=score_thresh
                                        )
    else:
        video = cv2.VideoCapture(src_dir)
        num_frame_passed = 0
        num_frame = args.num_frame

        while(video.isOpened()):
            if stop:
                break
            # Skip frames.
            num_frame_passed += 1
            if num_frame_passed % num_frame != 0:
                continue

            ret, frame = video.read()
            img_name = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f') + ".jpg"
            img_size = frame.shape[:2]

            count, stop = detect_and_save_imgs(tensors=tensors,
                                               src_dir=src_dir,
                                               correct_dir=correct_dir,
                                               incorrect_dir=incorrect_dir,
                                               img=frame,
                                               img_name=img_name,
                                               img_size=img_size,
                                               num_transform=num_transform,
                                               cls=cls,
                                               count=count,
                                               score_thresh=score_thresh
                                        )

    cv2.destroyAllWindows()


def parse_arguments(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('src_dir', type=str,
        help='Directory to the images that need to be labelled.')
    parser.add_argument('correct_dir', type=str,
        help='Directory to which all correctly detected images will be saved.')
    parser.add_argument('incorrect_dir', type=str,
        help='Directory to which all incorrectly detected images will be saved.')

    parser.add_argument('--load_video', action="store_true",
        help='Whether the input source is a video. \
        Default=false, meaning working with sequence of images.')
    parser.add_argument('--class_to_be_detected', type=int, default=3,
        help='The numerical value of class to be predicted. Default=3 (i.e., \'phone\')')
    parser.add_argument('--num_transform', type=int, default=2,
        help='Number of transform for each image by data augmentation. Default=2')
    parser.add_argument('--num_frame', type=int, default=30,
        help='Number of frame to skip per image. Default=30')
    parser.add_argument('--inference_graph_path', default="base_model/frozen_inference_graph.pb",
        type=str, help='Inference graph directory. Default=\'"base_model/frozen_inference_graph.pb"\'')
    parser.add_argument('--labelmap_path', default="base_model/labelmap.pbtxt",
        type=str, help='Labelmap directory. Default=\'base_model/labelmap.pbtxt\'')
    parser.add_argument('--score_thresh', type=float, default=0.2,
        help='Score threshold for displaying bounding boxes. Default=0.2')
    parser.add_argument('--width', type=int, default=900,
        help='Width of the imgs in the video stream.')
    parser.add_argument('--height', type=int, default=900,
        help='Height of the imgs in the video stream.')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
