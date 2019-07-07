import numpy as np
import pandas as pd
from vebits_api.others_util import assert_type, convert
from vebits_api.xml_util import create_xml_file

def get_bboxes_array(df, img_name=None):
    cols = ["xmin", "ymin", "xmax", "ymax"]
    if img_name is not None:
        return df[df.filename == img_name].loc[:, cols].to_numpy(dtype=np.int32)
    else:
        return df.loc[:, cols].to_numpy(dtype=np.int32)


def get_bboxes_array_and_classes(df, img_name):
    img_data = df[df.filename == img_name]
    bbox = np.asarray(img_data.loc[:, ["class", "xmin", "ymin", "xmax", "ymax"]])
    return bbox[:, 1:].astype(np.int), bbox[:, 0]


def filter_scores(scores, confidence_threshold):
    return scores > confidence_threshold


def filter_class(classes, classes_to_keep):
    pass

def filter_boxes(boxes, scores, classes, cls, confidence_threshold, img_size):
    height, width = img_size

    if cls == "all":
        fi = (scores > confidence_threshold)
    else:
        class_filter = np.isin(classes, cls)
        fi = (scores > confidence_threshold) * class_filter

    boxes = boxes[fi]

    boxes = boxes * [height, width, height, width]
    boxes[:, 0], boxes[:, 1] = boxes[:, 1].copy(), boxes[:, 0].copy()
    boxes[:, 2], boxes[:, 3] = boxes[:, 3].copy(), boxes[:, 2].copy()

    return fi, np.asarray(boxes, dtype=np.int)


class BBox():
    def __init__(self, label=None, bbox_array=None, bbox_series=None):
        if bbox_array is not None:
            self.from_xyxy_array(bbox_array)
            self.label = label
        elif bbox_series is not None:
            self.from_series(bbox_series)
        else:
            self.bbox = None
            self.label = label

    def _get_coord(self):
        self.xmin = self.bbox[0]
        self.ymin = self.bbox[1]
        self.xmax = self.bbox[2]
        self.ymax = self.bbox[3]

    def from_series(self, series):
        series = convert(series, pd.Series, pd.Series)

        self.bbox = series.loc[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=np.int32)
        self._get_coord()
        self.label = series.loc["class"]

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

    def to_series(self, filename, width, height):
        cols = ["filename", "width", "height", "class",
                "xmin", "ymin", "xmax", "ymax"]
        values = [filename, width, height, self.label,
                  self.xmin, self.ymin, self.xmax, self.ymax]
        return pd.Series(dict(zip(cols, values)))

    def to_xyxy_array(self):
        return self.bbox

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
