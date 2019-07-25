# Utilities for object detector.

from . import bbox_util
from . import im_util
from . import labelmap_util
from . import vis_util
from .others_util import check_import

import os
import sys
import time
from threading import Thread, Lock
from queue import Queue
from datetime import datetime
from collections import defaultdict

import numpy as np
import cv2
# Try importing Tensorflow
try:
    import tensorflow as tf
    tf_imported = True
except ModuleNotFoundError:
    print("No tensorflow installation found.")
    tf_imported = False
# Try importing DarkNet/Darkflow for YOLO
try:
    from darkflow.net.build import TFNet
    df_imported = True
except ModuleNotFoundError:
    print("No Darkflow installation found.")
    df_imported = False
# Multiprocessing
from multiprocessing.pool import ThreadPool
pool = ThreadPool()


# Load Tensorflow inference graph into memory
@check_import([tf_imported], ["tensorflow"])
def load_inference_graph_tf(inference_graph_path):
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
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
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


@check_import([df_imported], ["darkflow"])
def load_inference_graph_yolo(inference_graph_path, meta_path,
                              gpu_usage=0.95, confidence_threshold=0.5):
    """Load YOLO's inference graph into memory.

    Parameters
    ----------
    inference_graph_path : str
        Path to the inference graph generated by using
        --savepb option of Darkflow.
    meta_path : type
        Path to the metadata file generated by using
        --savepb option of Darkflow.
    gpu_usage: float
        By default, 95% of GPU power will be used.
    confidence_threshold: float

    Returns
    -------
    tensors
        List of tensors used for making inference.

    """
    # Pass a fake `FLAGS` object to Darkflow
    flags = {"pbLoad": inference_graph_path,
             "metaLoad": meta_path, "gpu": gpu_usage,
             "threshold": confidence_threshold}
    yolo_net = TFNet(flags)
    return {"yolo_net": yolo_net}


# Load a frozen infrerence graph into memory
@check_import([tf_imported, df_imported], ["tensorflow", "darkflow"])
def load_inference_graph(inference_graph_path, meta_path=None,
                         gpu_usage=0.95, confidence_threshold=0.5):
    """Interface to load either Tensorflow or Darknet's YOLO inference graph
    into memory.

    Parameters
    ----------
    inference_graph_path : str
    meta_path : str
        Path to the meta file generated by using option `--savepb` of Darkflow.
        If None, Tensorflow inference graph will be loaded.
        If not None, YOLO inference graph will be loaded.
    gpu_usage: float
        Used for YOLO graph when meta_path is specified. 95% of GPU memory
        will be used by default.
    confidence_threshold: float

    Returns
    -------
    tensors
        List of tensors used for making inference.

    """
    if meta_path is None:
        return load_inference_graph_tf(inference_graph_path)
    else:
        return load_inference_graph_yolo(inference_graph_path,
                                         meta_path, gpu_usage,
                                         confidence_threshold)


@check_import([tf_imported, df_imported], ["tensorflow", "darkflow"])
def load_tensors(inference_graph_path, labelmap_path,
                 num_classes=None, meta_path=None,
                 gpu_usage=0.95, confidence_threshold=0.5):
    """Interface to load either Tensorflow or Darknet's YOLO inference graph as
    well as other information such as label map into memory.

    Parameters
    ----------
    inference_graph_path : str
    labelmap_path : str
    num_classes : int
        Number of classes the model can detect. If not specified, it will
        be inferred from label map.
    meta_path : str
        Path to the meta file generated by using option `--savepb` of Darkflow.
        If None, Tensorflow inference graph will be loaded.
        If not None, YOLO inference graph will be loaded.
    confidence_threshold: float
        Confidence threshold.

    Returns
    -------
    tensors
        List of tensors used for making inference.

    """
    tensors = load_inference_graph(inference_graph_path, meta_path,
                                   gpu_usage, confidence_threshold)
    labelmap_dict = labelmap_util.get_label_map_dict(labelmap_path)
    labelmap_dict_inverse = labelmap_util.get_label_map_dict_inverse(
        labelmap_dict)
    # If `num_classes` is not specified, it will be inferred from labelmap.
    if num_classes is None:
        num_classes = len(labelmap_dict)
    category_index = labelmap_util.load_category_index(
        labelmap_path, num_classes)

    tensors["labelmap_dict"] = labelmap_dict
    tensors["labelmap_dict_inverse"] = labelmap_dict_inverse
    tensors["category_index"] = category_index

    return tensors


@check_import([tf_imported], ["tensorflow"])
def detect_objects_tf(imgs, tensors):
    sess = tensors["sess"]
    image_tensor = tensors["image_tensor"]
    detection_boxes = tensors["detection_boxes"]
    detection_scores = tensors["detection_scores"]
    detection_classes = tensors["detection_classes"]
    num_detections = tensors["num_detections"]

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
            detection_classes, num_detections],
        feed_dict={image_tensor: imgs})

    return boxes, scores, classes.astype(np.int32)


@check_import([df_imported], ["darkflow"])
def detect_objects_yolo(imgs, tensors):
    """This function makes use of multiprocessing to make predictions on batch.

    Parameters
    ----------
    imgs : list-like of images
    tensors : dict
        Contains tensors needed for making predictions.

    Returns
    -------
    boxes: tuple
        Tuple of length `n_images` containing list of boxes for each image.
    scores: tuple
    classes: tuple
        Note that this object already converts label index to label (e.g from 1
        to "phone").

    """

    yolo_net = tensors["yolo_net"]

    boxes_data = pool.map(lambda img: return_predict(yolo_net, img), imgs)
    boxes, scores, classes = list(zip(*boxes_data))
    return np.array(boxes), np.array(scores), np.array(classes)


def return_predict(net, img):
    """
    This function was modified from `darkflow.net.flow.return_predict`
    to work appropriately with this API.
    """
    height_orig, width_orig, _ = img.shape
    height, width = net.meta["inp_size"][:2]
    img = im_util.resize_padding(img, (height, width))
    img = net.framework.resize_input(img)
    this_inp = np.expand_dims(img, 0)
    feed_dict = {net.inp: this_inp}

    out = net.sess.run(net.out, feed_dict)[0]
    boxes = net.framework.findboxes(out)
    threshold = net.FLAGS.threshold
    boxes_out, scores, classes = [], [], []
    for box in boxes:
        tmpBox = process_box(box, img_size_feed=(height, width),
                             img_size_orig=(height_orig, width_orig),
                             threshold=threshold)
        if tmpBox is None:
            continue
        boxes_out.append(tmpBox[0])
        scores.append(tmpBox[1])
        # This API uses class index starting from 1
        classes.append(tmpBox[2] + 1)
    return np.array(boxes_out), np.array(scores), np.array(classes)


def process_box(box, img_size_feed, img_size_orig, threshold):
    """
    This function is used specifically for YOLO detections using Darkflow.
    This was implemented to be compatible with `resize_padding` function, since
    using this function to resize image seems to improve the performance.
    """
    height, width = img_size_feed
    max_indx = np.argmax(box.probs)
    max_prob = box.probs[max_indx]

    if max_prob > threshold:
        left = int((box.x - box.w / 2.) * width)
        right = int((box.x + box.w / 2.) * width)
        top = int((box.y - box.h / 2.) * height)
        bot = int((box.y + box.h / 2.) * height)
        box_out = bbox_util.boxes_padding_inverse([left, top, right, bot],
                                                  img_size=img_size_feed,
                                                  img_size_orig=img_size_orig)
        return (box_out, max_prob, max_indx)
    return None


@check_import([tf_imported, df_imported], ["tensorflow", "darkflow"])
def detect_objects(img, tensors):
    dims = img.ndim
    if dims == 3:
        img = np.expand_dims(img, axis=0)
    # Detect by yolo
    if "yolo_net" in tensors:
        boxes, scores, classes = detect_objects_yolo(img, tensors)
    # Detect by Tensorflow API
    else:
        boxes, scores, classes = detect_objects_tf(img, tensors)
    if dims == 3:
        return boxes[0], scores[0], classes[0]
    else:
        return boxes, scores, classes


class TFModel():
    @check_import([tf_imported], ["tensorflow"])
    def __init__(self, inference_graph_path, labelmap_path,
                 confidence_threshold=0.5,
                 class_to_be_detected="all"):
        self.tensors = load_tensors(inference_graph_path, labelmap_path)
        self.cls = class_to_be_detected
        self.threshold = confidence_threshold

    def detect_objects_on_single_image(self, img):
        """
        Parameters
        ----------
        img : ndarray
            Image to be detected. Can only be one image.

        Returns
        -------
        boxes, scores, classes: ndarrays
            Return coordinates, confidence scores and labels of bounding boxes.

        """
        self.img = img.copy()
        img_size = img.shape[:2]
        boxes, scores, classes = detect_objects(img, self.tensors)
        boxes, scores, classes = bbox_util.filter_boxes(boxes, scores,
                                                        classes, self.cls,
                                                        self.threshold,
                                                        img_size)

        self.boxes, self.scores, self.classes = boxes, scores, classes
        return boxes, scores, classes

    def draw_boxes_on_recent_image(self):
        """
        Note that this function returns a new annotated image. The original
        image fed to the model will not be affected.
        """
        return vis_util.draw_boxes_on_image(self.img, self.boxes, self.classes,
                                            self.tensors["labelmap_dict"])


class YOLOModel():
    @check_import([df_imported], ["darkflow"])
    def __init__(self, inference_graph_path, labelmap_path,
                 meta_path, confidence_threshold=0.5,
                 class_to_be_detected="all", gpu_usage=0.95):
        self.tensors = load_tensors(inference_graph_path, labelmap_path,
                                    meta_path=meta_path, gpu_usage=gpu_usage,
                                    confidence_threshold=confidence_threshold)

        self.threshold = confidence_threshold
        self.cls = class_to_be_detected

    def detect_objects_on_single_image(self, img):
        """
        Parameters
        ----------
        img : ndarray
            Image to be detected. Can only be one image.

        Returns
        -------
        boxes, scores, classes: ndarrays
            Return coordinates, confidence scores and labels of bounding boxes.

        """
        self.img = img.copy()
        boxes, scores, classes = detect_objects(img, self.tensors)

        self.boxes, self.scores, self.classes = boxes, scores, classes
        return boxes, scores, classes

    def draw_boxes_on_recent_image(self):
        """
        Note that this function returns a new annotated image. The original
        image fed to the model will not be affected.
        """
        return vis_util.draw_boxes_on_image(self.img, self.boxes, self.classes,
                                            self.tensors["labelmap_dict"])


class Model():
    """
    This model wraps up TFModel and YOLOModel for the sake of simplicity.
    """
    @check_import([tf_imported, df_imported], ["tensorflow", "darkflow"])
    def __init__(self, inference_graph_path, labelmap_path,
                 meta_path, confidence_threshold=0.5,
                 class_to_be_detected="all", gpu_usage=0.95):
        """
        If `meta_path` is specified, YOLOModel will be loaded. Otherwise,
        Tensorflow Object Detection API model will be loaded.
        """
        if meta_path is None:
            TFModel.__init__(self, inference_graph_path, labelmap_path,
                             confidence_threshold, class_to_be_detected)
            self.model = TFModel
        else:
            YOLOModel.__init__(self, inference_graph_path, labelmap_path,
                               meta_path, confidence_threshold,
                               class_to_be_detected, gpu_usage)
            self.model = YOLOModel

    def detect_objects_on_single_image(self, img):
        return self.model.detect_objects_on_single_image(self, img)

    def draw_boxes_on_recent_image(self):
        return self.model.draw_boxes_on_recent_image(self)


def is_queueing(queue, terminate_signal,
                num_tries=5, sleep_interval=0.1):
    """
    This function checks whether queueing is stopped or not.
    Return True if there are still frames in the queue.
    If stream is not stopped, try to wait a moment
    """
    tries = 0
    while queue.qsize() == 0 and not terminate_signal and tries < num_tries:
        time.sleep(0.1)
        tries += 1
    return self.Q.qsize() > 0


class VideoStream:
    """Convenience utility to handle Camera/Video stream.

    Parameters
    ----------
    src : str or int
        str: path to the video file
        int: index of the webcam device
    src_width : int
        If source is video, this option is ignored.
        If source is webcam, this specifies the width of the frame to capture
        from the webcam. Default=640
    src_height : int
        If source is video, this option is ignored.
        If source is webcam, this specifies the height of the frame to capture
        from the webcam. Default=480

    """
    def __init__(self, src, src_width=640, src_height=480):
        self.src = cv2.VideoCapture(src)
        self.count = -1
        self.mode = "webcam" if isinstance(src, int) else "video"
        self.terminate = False

        if self.mode == "webcam":
            self.src_width = src_width
            self.src_height = src_height
            self.src.set(3, src_width)
            self.src.set(4, src_height)

        else:
            # Grab the first frame to get frame width and height
            tmp_frame = self.src.read()[1]
            self.src_height, self.src_width = tmp_frame.shape[:2]
            # Release and reset
            self.src.release()
            self.src = cv2.VideoCapture(src)

        # Set default parameters for displaying
        self.set_display_params("OpenCV", self.src_width,
                                self.src_height, init=False)
        self.out = None # No output by default

    def set_display_params(self, display_name,
                           display_width, display_height,
                           terminate_key=113, pause_key=32,
                           delay=1, draw_count=False,
                           init=False):
        """
        By default, press "q" (code: 113) to terminate displaying
        and spacebar (code: 32) to pause. Must be passed as a Unicode
        value. `delay`: number of miliseconds to wait until next frame.
        """
        self.display_name = display_name
        self.display_width = display_width
        self.display_height = display_height

        self.terminate_key = terminate_key
        self.pause_key = pause_key
        self.delay = delay
        self.draw_count = draw_count
        self.init = init

        if init:
            self.init_display()

    def init_display(self):
        cv2.namedWindow(self.display_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.display_name, self.display_width,
                         self.display_height)
        self.init = True

    def set_output_params(self, output_path, output_width=None,
                          output_height=None, resize_func=None,
                          fourcc="mp4v", fps=20):
        """Set parameters for saving video output.

        Parameters
        ----------
        output_path : str
        output_width : int
            If None, use source width.
        output_height : int
            If None, use source height.
        resize_func : callable
            If `output_width` or `output_height` is different from `src_width`
            and `src_height`, respectively, resize function will be used to
            resize frames. It takes exactly one argument, which is an image,
            and return an image that has the same shape as specified
            output shape. If None, `vebits_api.im_util.resize_padding` will
            be used.
        fourcc: str of length 4
            4-byte code used to specify the video codec. By default, "mp4v" is
            used to save `*.mp4` files.
        fps: int

        """
        if output_width is None:
            output_width = self.src_width
        if output_height is None:
            output_height = self.src_height

        self.output_size = (output_height, output_width)
        self.out = cv2.VideoWriter(output_path,
                                   cv2.VideoWriter_fourcc(*fourcc),
                                   fps, self.output_size[::-1])
        # Set self.diff, a parameter controls whether output video has the
        # same shape as input stream. If self.diff is True, resize_func will
        # be used to resize input frames to desired shape.
        if output_width != self.src_width or output_height != self.src_height:
            self.diff = True
        else:
            self.diff = False

        if resize_func is None:
            self.resize_func = lambda x: im_util.resize_padding(x, self.output_size)
        else:
            self.resize_func = resize_func

    def grab(self):
        """
        Return False if end of streaming.
        """
        self.ret, self.frame = self.src.read()
        if self.ret:
            self.count += 1
        else:
            self.terminate = True
        return self.frame

    def __iter__(self):
        return self

    def __next__(self):
        # Grab a new frame and get the `terminate` value.
        # If it is True, terminate the loop.
        self.grab()
        if self.terminate:
            self.stop()
            raise StopIteration
        else:
            return self.frame

    def display_frame(self, frame=None):
        """
        If `frame` is None, display the current frame grabbed. Otherwise,
        display the desired frame. Return False if terminate signal is fired.
        """
        # Check whether displaying utility is initialized or not
        if not self.init:
            self.init_display()
        if frame is None:
            frame = self.frame
        # Draw frame count
        if self.draw_count:
            frame = vis_util.draw_number(frame, self.count)
        cv2.imshow(self.display_name, frame)
        # Capture key
        key = cv2.waitKey(self.delay)
        # If terminated
        if key == self.terminate_key:
            self.terminate = True
            return False
        # If paused
        if key == self.pause_key:
            while True:
                key = cv2.waitKey(self.delay)
                # Wait until received the pause key again
                if key == self.pause_key:
                    break
        return True

    def draw_count_on_frame(self, frame=None, count=None):
        if frame is None:
            frame = self.frame
        if count is None:
            count = self.count
        return vis_util.draw_number(frame, count)

    def write_frame(self, frame=None):
        """
        If `frame` is None, display the last frame grabbed.
        Otherwise, display the desired frame.
        """
        if self.out is None:
            raise ValueError("Output parameters are not set. "
                             "Set by using `set_output_params`.")
        if frame is None:
            frame = self.frame
        # Resize frame
        if self.diff:
            frame = self.resize_func(frame)
        self.out.write(frame)
    # Release utilities
    def release_in(self):
        self.src.release()

    def release_out(self):
        self.out.release()

    def stop(self):
        # Release video i/o
        self.release_in()
        if self.out is not None:
            self.release_out()


class MultiThreadingVideoStream(VideoStream):
    """
    Video streaming using multithreading.
    """
    def __init__(self, src, src_width=640,
                 src_height=480, queue_size=128,
                 num_threads=1):
        # Super init
        super().__init__(src, src_width, src_height)
        # Multithreading and queueing
        self.Q = FrameQueue(maxsize=queue_size)
        # Initialize threads
        self.threads = []
        self.locker = Lock()
        for i in range(num_threads):
            thread = CustomThread(self.grab_inf, i, self.locker)
            thread.daemon = True
            thread.start()
            self.threads.append(thread)

    def grab(self):
        # Read and add frame to the queue
        self.ret, self.frame = self.src.read()
        if self.ret:
            self.count += 1
            self.Q.put(self.frame, self.count)
        else:
            self.terminate = True
        return self.terminate

    def grab_inf(self):
        # Read indefinitely
        while not self.terminate:
            # Check if the queue is full
            if self.Q.full():
                time.sleep(0.1)
            else:
                self.grab()

    def __iter__(self):
        return self

    def __next__(self):
		# Read frame from the queue
        # If there remains frames in queue and terminate signal is not fired.
        if self.more():
            return self.Q.get()[0]
        else:
            self.stop()
            raise StopIteration
    # Function to handle when to stop, taken from
    #   https://github.com/jrosebr1/imutils/blob/master/imutils/video/filevideostream.py
    def more(self):
        # Return True if there are still frames in the queue.
        # If stream is not stopped, try to wait a moment
        tries = 0
        while self.Q.qsize() == 0 and not self.terminate and tries < 5:
            time.sleep(0.1)
            tries += 1

        return self.Q.qsize() > 0

    def stop(self):
        super().stop()
        for thread in self.threads:
            thread.join()


class CustomThread(Thread):
   def __init__(self, target, thread_id, locker, **kwargs):
      super().__init__()
      self.thread_id = thread_id
      self.locker = locker
      self.func = target
      self.func_kwargs = kwargs

   def run(self):
      # Get lock to synchronize threads
      self.locker.acquire()
      self.func()
      # Free lock to release next thread
      self.locker.release()


class FrameQueue(Queue):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id_queue = Queue(*args, **kwargs)

    def put(self, frame, frame_count):
        super().put(frame)
        self.id_queue.put(frame_count)

    def get(self):
        return super().get(), self.id_queue.get()


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
