import numpy as np
import pandas as pd

from .others_util import raise_type_error, convert, assert_type
from .xml_util import create_xml_file

BBOX_COLS = ["xmin", "ymin", "xmax", "ymax"]


def get_bboxes_array(data, bbox_cols=BBOX_COLS):
    if isinstance(data, pd.DataFrame):
        return df.loc[:, bbox_cols].to_numpy(dtype=np.int32)
    elif isinstance(data, pd.Series):
        return df.loc[bbox_cols].to_numpy(dtype=np.int32)
    elif isinstance(data, BBox) or isinstance(data, BBoxes):
        return data.to_xyxy_array()
    else:
        raise_type_error(type(data), [pd.DataFrame, pd.Series, BBox, BBoxes])


def get_bboxes_array_and_label(data, bbox_cols=BBOX_COLS):
    bboxes = get_bboxes_array(data, bbox_cols)
    if isinstance(data, pd.DataFrame):
        return bboxes, data.loc[:, "class"]
    elif isinstance(data, pd.Series):
        return bboxes, data.loc["class"]
    elif isinstance(data, BBox) or isinstance(data, BBoxes):
        return data.to_xyxy_array_and_label()
    else:
        raise_type_error(type(data), [pd.DataFrame, pd.Series, BBox, BBoxes])


def scores_mask(scores, confidence_threshold):
    return scores > confidence_threshold


def classes_mask(classes, classes_to_keep):
    return np.isin(classes, classes_to_keep)


def get_mask(boxes, scores, classes,
             classes_to_keep, confidence_threshold):
    score_mask = scores_mask(scores, confidence_threshold)
    if classes_to_keep != "all" and classes_to_keep is not None:
        class_mask = classes_mask(classes, cls)
        return score_mask * class_mask
    else:
        return score_mask


def filter_boxes(boxes, scores, classes, classes_to_keep,
                 confidence_threshold, img_size):
    """
    This function is used to process bounding boxes returned by Tensorflow
    Object Detection API only.
    """
    mask = get_mask(boxes, scores, classes,
                    classes_to_keep, confidence_threshold)
    boxes = boxes[mask]
    scores = scores[mask]
    classes = classes[mask]
    # Because `boxes` is of floating points with relative to image size,
    # we need to convert it back to coordinates.
    height, width = img_size
    boxes = boxes * [height, width, height, width]
    # Now the boxes is `ymin, xmin, ymax, xmax`.
    boxes = np.asarray(boxes[:, [1, 0, 3, 2]], dtype=np.int32)
    # Ensure type of classes is correct
    classes = classes.astype(np.int32)
    return boxes, scores, classes


def _area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def area(bbox):
    if isinstance(bbox, BBox):
        return _area(bbox.to_xyxy_array())
    else:
        return _area(bbox)


def _intersection(bbox_1, bbox_2):
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(bbox_1[0], bbox_2[0])
    yA = max(bbox_1[1], bbox_2[1])
    xB = min(bbox_1[2], bbox_2[2])
    yB = min(bbox_1[3], bbox_2[3])

    # Compute the area of intersection rectangle
    inter = max(0, xB - xA) * max(0, yB - yA)
    return inter


def intersection(bbox_1, bbox_2):
    if isinstance(bbox_1, BBox) and isinstance(bbox_2, BBox):
        return _intersection(bbox_1.to_xyxy_array(), bbox_2.to_xyxy_array())
    else:
        return _intersection(bbox_1, bbox_2)


def _union(bbox_1, bbox_2):
    inter = intersection(bbox_1, bbox_2)
    bbox_1_area = _area(bbox_1)
    bbox_2_area = _area(bbox_2)
    return bbox_1_area + bbox_2_area - inter


def union(bbox_1, bbox_2):
    if isinstance(bbox_1, BBox) and isinstance(bbox_2, BBox):
        return _union(bbox_1.to_xyxy_array(), bbox_2.to_xyxy_array())
    else:
        return _union(bbox_1, bbox_2)


def _iou(bbox_1, bbox_2):
    return float(_intersection(bbox_1, bbox_2)) / float(_union(bbox_1, bbox_2))


def iou(bbox_1, bbox_2):
    if isinstance(bbox_1, BBox) and isinstance(bbox_2, BBox):
        return _iou(bbox_1.to_xyxy_array(), bbox_2.to_xyxy_array())
    else:
        return _iou(bbox_1, bbox_2)


def boxes_padding_inverse(bboxes, img_size, img_size_orig):
    """This function is used to calculate the coordinates of the bounding boxes
    before its corresponding image is resized by `resize_padding` function given
    the bouding boxes after the image is resized.

    Parameters
    ----------
    bboxes : array-like
        ndarray of shape (n, 4) o (4,), where n is the number of
        bounding boxes of the same image.
    img_size : tuple-like
        `(height, width)` of the image after resized.
    img_size_orig : tuple-like
        `(height, width)` of the image before resized.

    """
    height, width = img_size
    height_orig, width_orig = img_size_orig
    scale = min(height / float(height_orig), width / float(width_orig))
    # Calculate offsets along two dimensions
    offset_x = (width - width_orig * scale) / 2
    offset_y = (height - height_orig * scale) / 2
    # Calculate original bounding boxes
    offsets = np.array([offset_x, offset_y] * 2)
    bboxes_orig = bboxes - offsets
    # Scale back to original size
    bboxes_orig /= scale

    return bboxes_orig



class BBox():
    def __init__(self, label=None, bbox_array=None, bbox_series=None):
        if bbox_array is not None:
            self.from_xyxy_array(bbox_array)
            self.label = label
        elif bbox_series is not None:
            self.from_series(bbox_series)
            # In case `label` is set explicitly
            if label is not None:
                self.label = label
        else:
            self.bbox = None
            self.label = label

    # Functions for reading data
    def _get_coord(self):
        self.xmin = self.bbox[0]
        self.ymin = self.bbox[1]
        self.xmax = self.bbox[2]
        self.ymax = self.bbox[3]

    def from_series(self, series):
        series = convert(series, pd.Series, pd.Series)

        self.bbox, self.label = get_bboxes_array_and_label(series)
        self._get_coord()

    def from_xyxy_array(self, array, label=None):
        array = convert(array,
                        lambda x: np.asarray(x, dtype=np.int32),
                        np.ndarray)

        array = np.squeeze(array)
        if array.shape != (4,):
            raise ValueError("Input bounding box must be of shape (4,), "
                             "got shape {} instead".format(array.shape))
        self.bbox = array
        self._get_coord()

        if label is not None:
            self.label = label

    # Functions for outputting data
    def to_series(self, filename, width, height):
        cols = ["filename", "width", "height", "class"] + BBOX_COLS
        values = [filename, width, height, self.label,
                  self.xmin, self.ymin, self.xmax, self.ymax]
        return pd.Series(dict(zip(cols, values)))

    def to_xyxy_array(self):
        return self.bbox

    def to_xyxy_array_and_label(self):
        return self.bbox, self.label

    # Calculation utilities
    def area(self):
        return _area(self.bbox)

    def intersection(self, bbox):
        if isinstance(bbox, BBox):
            return _intersection(self.bbox, bbox.to_xyxy_array())
        else:
            return _intersection(self.bbox, bbox)

    def union(self, bbox):
        if isinstance(bbox, BBox):
            return _union(self.bbox, bbox.to_xyxy_array())
        else:
            return _union(self.bbox, bbox)

    def iou(self, bbox):
        if isinstance(bbox, BBox):
            return _iou(self.bbox, bbox.to_xyxy_array())
        else:
            return _iou(self.bbox, bbox)

    # Funtions for getting attributes
    def get_label(self):
        return self.label

    def get_xmin(self):
        return self.xmin

    def get_xmax(self):
        return self.xmax

    def get_ymin(self):
        return self.ymin

    def get_ymax(self):
        return self.ymax

    # Funtions for setting attributes
    def set_label(self, label):
        self.label = label

    # Other utilities
    def draw_on_image(self, img):
        if self.label is None:
            pass


class BBoxes():
    def __init__(self, df=None, bboxes_list=None,
                 filename=None, width=None, height=None):
        self.df = None
        self.bboxes_list = None
        if df is not None:
            self.from_dataframe(df)

        elif bboxes_list is not None:
            self.from_bboxes_list(bboxes_list, filename, width, height)

    def _get_info_from_df(self, df):
        filename = df.filename.unique().tolist()
        width = df.width.unique().tolist()
        height = df.height.unique().tolist()
        if not (len(filename) == len(width) == len(height) == 1):
            raise ValueError("`filename`, `width` and `height` must be unique, "
                             "but got `filename`: {}, `width`: {}, `height`: "
                             "{}".format(filename, width, height))

        self.filename = filename[0]
        self.width = width[0]
        self.height = height[0]

    def from_dataframe(self, df):
        df = convert(df, pd.DataFrame, pd.DataFrame)
        self._get_info_from_df(df)
        self.df = df.copy()

    def from_bboxes_list(self, bboxes_list, filename, width, height):
        if filename is None or width is None or height is None:
            raise TypeError("Arguments required: filename, width, height")

        bboxes_list = convert(bboxes_list, list, list)
        # If not all objects in list are of BBox class.
        if not all([isinstance(obj, BBox) for obj in bboxes_list]):
            raise TypeError("Invalid data type. "
                            "Expected list-like of BBox objects")

        self.bboxes_list = bboxes_list
        self.filename = filename
        self.width = width
        self.height = height
        # By default, dataframe is the main source of data
        self.to_dataframe()

    def to_dataframe(self):
        if self.df is None and self.bboxes_list is None:
            raise ValueError("Please provide either dataframe of "
                             "bounding boxes or list of BBox objects")
        if self.df is None:
            df = []
            for bbox in self.bboxes_list:
                df.append(bbox.to_series(
                                self.filename, self.width, self.height))
            self.df = pd.DataFrame(df)
        return self.df

    def to_bboxes_list(self):
        if self.df is None and self.bboxes_list is None:
            raise ValueError("Please provide either dataframe of "
                             "bounding boxes or list of BBox objects")
        if self.bboxes_list is None:
            labels = self.df.loc[:, "class"].to_numpy()
            bboxes_array = get_bboxes_array(self.df)
            self.bboxes_list = []

            for label, bbox in zip(labels, bboxes_array):
                self.bboxes_list.append(BBox(label, bbox))
        return self.bboxes_list

    def to_xml(self, img_path, xml_path=None):
        self.to_bboxes_list()
        create_xml_file(img_path, self.width, self.height,
                        self.bboxes_list, xml_path)

    def to_xyxy_array(self):
        # Sanity check
        if self.df is None and self.bboxes_list is None:
            raise ValueError("Please provide either dataframe of "
                             "bounding boxes or list of BBox objects")
        return self.df.loc[:, BBOX_COLS].to_numpy(dtype=np.int32)

    def to_xyxy_array_and_label(self):
        # Sanity check
        if self.df is None and self.bboxes_list is None:
            raise ValueError("Please provide either dataframe of "
                             "bounding boxes or list of BBox objects")
        return self.df.loc[:, BBOX_COLS].to_numpy(dtype=np.int32), self.df.loc[:, "class"].to_numpy()
