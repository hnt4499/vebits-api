from tf_utils import detector_util as detector_util
import xml.etree.ElementTree as ET

import cv2
import tensorflow as tf
import datetime
import argparse
import os
import sys

LABELS = {
    1: 'face',
    2: 'hand',
    3: 'phone'
}


class BBox():
    def __init__(self, name, xmin, ymin, xmax, ymax):
        self.name = name
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def get_name(self):
        return self.name

    def get_xmin(self):
        return self.xmin

    def get_xmax(self):
        return self.xmax

    def get_ymin(self):
        return self.ymin

    def get_ymax(self):
        return self.ymax
        

def create_xml_file(in_folder, in_filename, in_path, in_width, in_height, bbox_list):
    # create the file structure
    annotate = ET.Element('annotation')
    folder = ET.SubElement(annotate, 'folder')
    filename = ET.SubElement(annotate, 'filename')
    path = ET.SubElement(annotate, 'path')
    source = ET.SubElement(annotate, 'source')
    database = ET.SubElement(source, 'database')
    size = ET.SubElement(annotate, 'size')
    width = ET.SubElement(size, 'width')
    height = ET.SubElement(size, 'height')
    depth = ET.SubElement(size, 'depth')
    segmented = ET.SubElement(annotate, 'segmented')

    folder.text = str(in_folder)
    filename.text = str(in_filename)
    path.text = str(in_path)
    database.text = 'Unknown'
    width.text = str(in_width)
    height.text = str(in_height)
    depth.text = '3'
    segmented.text = '0'

    # Values in object are dynamic
    for bbox in bbox_list:
        obj = ET.SubElement(annotate, 'object')
        name = ET.SubElement(obj, 'name')
        pose = ET.SubElement(obj, 'pose')
        truncated = ET.SubElement(obj, 'truncated')
        difficult = ET.SubElement(obj, 'difficult')
        bndbox = ET.SubElement(obj, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        ymin = ET.SubElement(bndbox, 'ymin')
        xmax = ET.SubElement(bndbox, 'xmax')
        ymax = ET.SubElement(bndbox, 'ymax')

        name.text = str(bbox.get_name())
        pose.text = 'Unspecified'
        truncated.text = '0'
        difficult.text = '0'

        xmin.text = str(bbox.get_xmin())
        ymin.text = str(bbox.get_ymin())
        xmax.text = str(bbox.get_xmax())
        ymax.text = str(bbox.get_ymax())

    # create a new XML file with the results
    mydata = ET.tostring(annotate, encoding="unicode")

    name, _ = os.path.splitext(in_path)
    xml_name = name + '.xml'
    with open(xml_name, "w") as myfile:
        myfile.write(mydata)



def main(args):

    tensors = detector_util.load_inference_graph(args.inference_graph_path)

    correct_dir = args.correct_dir
    incorrect_dir = args.incorrect_dir
    score_thresh = args.score_thresh

    cap = cv2.VideoCapture(args.video_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    start_time = datetime.datetime.now()
    num_frames = 0
    im_width, im_height = (cap.get(3), cap.get(4))
    # max number of hands we want to detect/track
    num_hands_detect = 2

    cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_NORMAL)

    while True:
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        ret, image_np = cap.read()

        # image_np = cv2.flip(image_np, 1)
        # try:
        #     image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        # except:
        #     print("Error converting to RGB")

        display = image_np.copy()
        # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
        # while scores contains the confidence for each of these boxes.
        # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)

        boxes, scores, classes = detector_util.detect_objects(tensors)

        bboxes = []
        for i in range(num_hands_detect):
            if (scores[i] > 0.5):
                (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                            boxes[i][0] * im_height, boxes[i][2] * im_height)
                w_offset = int((right - left) * 0.10)
                h_offset = int((bottom - top) * 0.10)
                p1 = (int(left) - w_offset, int(top) - h_offset)
                p2 = (int(right) + w_offset, int(bottom) + h_offset)
                cv2.rectangle(display, p1, p2, (77, 255, 9), 3, 1)
                bboxes.append(BBox(LABELS[classes[i]], int(left), int(top), int(right), int(bottom)))
                cv2.putText(display,LABELS[classes[i]],p1, cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,100),2,cv2.LINE_AA)


        # Calculate Frames per second (FPS)
        num_frames += 1
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        fps = num_frames / elapsed_time


        detector_util.draw_fps_on_image("FPS : " + str(int(fps)),
                                                 display)
        cv2.imshow('Single-Threaded Detection', cv2.cvtColor(display, cv2.COLOR_RGB2BGR))

        # try:
        #     image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        # except:
        #     print("Error converting to BGR")


        key = cv2.waitKey(10)
        # Save image and annotation
        unique_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
        if key == ord("d"):
            create_xml_file(
                correct_dir,
                unique_id + '.jpg',
                os.path.join(correct_dir, unique_id + '.jpg'),
                im_width,
                im_height,
                bboxes
            )
            cv2.imwrite(os.path.join(correct_dir, unique_id + '.jpg'), image_np)
            print("Image Saved: ", unique_id)

        elif key == ord("w"):
            cv2.imwrite(os.path.join(incorrect_dir, unique_id + '.jpg'), image_np)

        elif key & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break



def parse_arguments(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('correct_dir', type=str,
        help='Directory to which all correctly detected images will be saved.')
    parser.add_argument('incorrect_dir', type=str,
        help='Directory to which all incorrectly detected images will be saved.')

    parser.add_argument('--inference_graph_path', type=str,
        default="inference_graph/frozen_inference_graph.pb",
        help='Path to the inference graph.')
    parser.add_argument('--score_thresh', type=float, default=0.2,
        help='Score threshold for displaying bounding boxes')
    parser.add_argument('--fps', type=int, default=1,
        help='Show FPS on detection/display visualization')
    parser.add_argument('--video_source', type=int, default=0,
        help='Device index of the camera.')
    parser.add_argument('--width', type=int, default=900,
        help='Width of the frames in the video stream.')
    parser.add_argument('--height', type=int, default=900,
        help='Height of the frames in the video stream.')
    parser.add_argument('--display', type=int, default=1,
        help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument('--num_workers', type=int, default=4,
        help='Number of workers.')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
