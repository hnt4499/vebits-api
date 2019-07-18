import cv2
import numpy as np

from .bbox_util import BBox, BBoxes
from .others_util import convert, raise_type_error
from .labelmap_util import get_label_map_dict_inverse

FONT = cv2.FONT_HERSHEY_SIMPLEX
COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (100, 100, 100), (0, 255, 0)]


def _draw_box_on_image(img, box, label, color,
                       text_scale=0.75, thickness=2):
    # Use default color if `color` is not specified.
    if color is None:
        color = COLORS[0]
    p1 = (int(box[0]), int(box[1]))
    p2 = (int(box[2]), int(box[3]))
    cv2.rectangle(img, p1, p2, color, thickness=thickness, lineType=1)
    if label is not None:
        cv2.putText(img, label, p1, FONT, fontScale=text_scale, color=color,
                    thickness=thickness, lineType=cv2.LINE_AA)
    return img


def draw_box_on_image(img, box, label=None, color=None, **kwargs):
    # When no box is detected
    if box is None:
        return img
    if isinstance(box, BBox):
        if label is None:
            return _draw_box_on_image(img, box.to_xyxy_array(), label, color,
                                      **kwargs)
        else:
            return _draw_box_on_image(img, box.to_xyxy_array(),
                                      box.get_label(), color,
                                      **kwargs)
    else:
        try:
            box = convert(box,
                          lambda x: np.asarray(x, dtype=np.int32),
                          np.ndarray)
        except TypeError:
            raise_type_error(type(box), [BBox, np.ndarray])
    # When no box is detected
    if box.shape[0] == 0:
        return img
    if box.shape != (4,):
        raise ValueError("Input bounding box must be of shape (4,), "
                         "got shape {} instead".format(box.shape))
    else:
        return _draw_box_on_image(img, box, label, color,
                                  **kwargs)


def _draw_boxes_on_image(img, boxes, labels_index,
                         labelmap_dict, **kwargs):
    """
    This function only accepts boxes as a ndarray.
    """
    labelmap_dict_inverse = get_label_map_dict_inverse(labelmap_dict)
    for i in range(boxes.shape[0]):
        if labels_index is None:
            img = _draw_box_on_image(img, boxes[i], None, None,
                                     **kwargs)
        else:
            label = labels_index[i]
            label_text = labelmap_dict_inverse[label]
            img = _draw_box_on_image(img, boxes[i], label_text, COLORS[label],
                                     **kwargs)
    return img


def draw_boxes_on_image(img, boxes, labels_index, labelmap_dict,
                        **kwargs):
    """Short summary.

    Parameters
    ----------
    img : ndarray
        Input image.
    boxes : ndarray-like
        It must has shape (n ,4) where n is the number of
        bounding boxes.
    labels_index : ndarray-like
        An array containing index of labels of bounding boxes. If None, only
        bouding boxes will be drawn.
    labelmap_dict : dict
        A dictionary mapping labels with its index.

    Returns
    -------
    img
        Return annotated image.

    """
    # When no box is detected
    if boxes is None:
        return img
    try:
        boxes = convert(boxes,
                      lambda x: np.asarray(x, dtype=np.int32),
                      np.ndarray)
    except TypeError:
        raise_type_error(type(boxes), [np.ndarray])
    # When no box is detected
    if boxes.shape[0] == 0:
        return img
    if boxes.shape[1] != 4 or boxes.ndim != 2:
        raise ValueError("Input bounding box must be of shape (n, 4), "
                         "got shape {} instead".format(boxes.shape))
    else:
        return _draw_boxes_on_image(img, boxes, labels_index,
                                    labelmap_dict, **kwargs)


def draw_number(img, number, loc=None):
    loc = (20, 50) if loc is None else loc
    cv2.putText(img, str(number), loc,
                FONT, 1.25, COLORS[0], 2)
    return img
