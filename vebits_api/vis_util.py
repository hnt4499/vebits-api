from functools import wraps

import cv2
import numpy as np

from .bbox_util import BBox, BBoxes
from .others_util import convert, raise_type_error
from .labelmap_util import get_label_map_dict_inverse

FONT = cv2.FONT_HERSHEY_SIMPLEX
COLORS = [(119, 76, 219), (154, 85, 16), (200, 162, 60), (23, 203, 176),
          (0, 255, 0), (152, 34, 82), (1, 122, 254), (171, 21, 192),
          (57, 40, 234), (0, 221, 254), (204, 159, 120), (0, 255, 152),
          (20, 102, 172), (77, 80, 244), (75, 202, 242), (130, 78, 139),
          (194, 153, 194)]


def transparent(function):
    """
    This acts as a wrapper/decorator to provide ability to
    draw transparent line/rectangle/etc. Can be used to wrap
    any function that involves drawing.

    Properties: the first argument of the function to be wrapped must be an
    array representing an image. `alpha` should be passed as a keyword argument.
    By default, `alpha` = 1.0 is used, meaning that no transparency is applied.
    """
    @wraps(function) # to preserve `function` metadata
    def overlay(*args, **kwargs):
        if len(args) > 0:
            img = args[0].copy()
        else:
            img = kwargs["img"].copy()
        # Get `alpha` value.
        if "alpha" in kwargs:
            alpha = kwargs["alpha"]
            kwargs = {k:v for k, v in kwargs.items() if k != "alpha"}
        else:
            alpha = 1.0
        img_drawn = function(*args, **kwargs)
        cv2.addWeighted(img_drawn, alpha, img, 1 - alpha,
                        0, img)
        return img
    return overlay


@transparent
def _draw_box_on_image(img, box, label, color,
                       text_scale=0.75, thickness=2,
                       line_type=cv2.LINE_AA):
    # Use default color if `color` is not specified.
    if color is None:
        color = COLORS[0]
    p1 = (int(box[0]), int(box[1]))
    p2 = (int(box[2]), int(box[3]))
    cv2.rectangle(img, p1, p2, color, thickness=thickness, lineType=1)
    if label is not None:
        cv2.putText(img, label, p1, FONT, fontScale=text_scale, color=color,
                    thickness=thickness, lineType=line_type)
    return img


@transparent
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


@transparent
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
            color = COLORS[label % len(COLORS)]
            img = _draw_box_on_image(img, boxes[i], label_text,
                                     color, **kwargs)
    return img


@transparent
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
        bounding boxes will be drawn.
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


@transparent
def draw_number(img, number, loc=None,
                text_scale=1.25, color=None,
                thickness=2):
    """
    This function is used mainly to draw frame number, but it is also can be
    used to draw any text information on image.
    """
    # If `loc` is None, automatically calculate appropriate location.
    if loc is None:
        size = cv2.getTextSize(str(number), FONT, text_scale, thickness)
        loc = (10, size[0][1] + 10) # margin is 10 to top left corner
    if color is None:
        color = COLORS[0]
    cv2.putText(img, str(number), loc,
                FONT, text_scale, color,
                thickness)
    return img
