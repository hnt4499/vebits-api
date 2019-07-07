import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX
colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (100, 100, 100), (0, 255, 0)]


def draw_boxes_and_labels(img, boxes, classes=None, labelmap_dict=None):
    for i in range(boxes.shape[0]):
        box = boxes[i]
        cl = classes[i]
        p1 = (int(box[0]), int(box[1]))
        p2 = (int(box[2]), int(box[3]))

        if classes is not None:
            cl_num = labelmap_dict[cl]
            cv2.putText(img, cl, p1, FONT, 0.75, colors[cl_num], 2, cv2.LINE_AA)
            cv2.rectangle(img, p1, p2, colors[cl_num], 3, 1)
        else:
            cv2.rectangle(img, p1, p2, colors[0], 3, 1)
    return img


def draw_number(img, number, loc=None):
    loc = (20, 50) if loc is None else loc
    cv2.putText(img, str(number), loc,
                FONT, 1.25, colors[0], 2)
    return img
