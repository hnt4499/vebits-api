import cv2
from vebits_api.bbox_util import BBox, BBoxes
from vebits_api.others_util import convert, raise_type_error

FONT = cv2.FONT_HERSHEY_SIMPLEX
colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (100, 100, 100), (0, 255, 0)]


def _draw_box_on_image(img, box, label, color):
    # Use default color if `color` is not specified.
    if color is None:
        color = colors[0]
    p1 = (int(box[0]), int(box[1]))
    p2 = (int(box[2]), int(box[3]))
    cv2.rectangle(img, p1, p2, color, 3, 1)
    if label is not None:
        cv2.putText(img, label, p1, FONT, 0.75, color, 2, cv2.LINE_AA)
    return img


def draw_box_on_image(img, box, label=None, color=None):
    if isinstance(box, BBox):
        if label is None:
            return _draw_box_on_image(img, box.to_xyxy_array(), label, color)
        else:
            return _draw_box_on_image(img, box.to_xyxy_array(),
                                      box.get_label(), color)
    else:
        try:
            box = convert(box,
                          lambda x: np.asarray(x, dtype=np.int32),
                          np.ndarray)
            if box.shape != (4,):
                raise ValueError("Input bounding box must be of shape (4,), "
                                 "got shape {} instead".format(box.shape))
            else:
                return _draw_box_on_image(img, box, label, color)
        except:
            raise_type_error(type(box), [BBox, np.ndarray])


def _draw_boxes_on_image(img, boxes, labels, labelmap_dict):
    for i in range(boxes.shape[0]):
        if labels is None:
            img = _draw_box_on_image(img, boxes[i], None, None)
        else:
            label = labels[i]
            img = _draw_box_on_image(img, boxes[i], label, labelmap_dict[label])
    return img


def draw_boxes_on_image(img, boxes, labels, labelmap_dict):
    """Short summary.

    Parameters
    ----------
    img : ndarray
        Input image.
    boxes : ndarray-like or BBoxes
        If `ndarray`, it must has shape (n ,4) where n is the number of
        bounding boxes.
    labels : ndarray-like
        If set to None:
            - If `boxes` is of class `BBoxes`, internal bounding boxes
              inferred from `boxes` are used.
            - If `boxes` is `ndarray`-like, no label is used.
        If explicitly set, `label` from `BBoxes` will not be used.
    labelmap_dict : dict
        A dictionary mapping labels with its index.

    Returns
    -------
    img
        Return annotated image.

    """
    if isinstance(boxes, BBoxes):
        boxes, labels_in = boxes.to_xyxy_array_and_label()
        # If `labels` is not explicitly set
        if labels is None:
            return _draw_boxes_on_image(img, boxes.to_xyxy_array(),
                                        labels_in, labelmap_dict)
        else:
            return _draw_boxes_on_image(img, boxes.to_xyxy_array(),
                                        labels, labelmap_dict)
    else:
        try:
            boxes = convert(boxes,
                          lambda x: np.asarray(x, dtype=np.int32),
                          np.ndarray)
            if boxes.shape[1] != 4:
                raise ValueError("Input bounding box must be of shape (n, 4), "
                                 "got shape {} instead".format(boxes.shape))
            else:
                return _draw_boxes_on_image(img, boxes, labels, labelmap_dict)
        except:
            raise_type_error(type(boxes), [BBoxes, np.ndarray])


def draw_number(img, number, loc=None):
    loc = (20, 50) if loc is None else loc
    cv2.putText(img, str(number), loc,
                FONT, 1.25, colors[0], 2)
    return img
