import cv2
import numpy as np

def resize_padding(img, size, bboxes=None):
    canvas = np.full((size[0], size[1], 3), 0, dtype=np.uint8)

    img_height, img_width = img.shape[:2]

    scale = min(size[0] / img_height, size[1] / img_width)

    img_new_height = int(scale * img_height)
    img_new_width = int(scale * img_width)
    img_resized = cv2.resize(img, (img_new_width, img_new_height))

    del_h = (size[0] - img_new_height) // 2
    del_w = (size[1] - img_new_width) // 2
    canvas[del_h:del_h + img_new_height, del_w:del_w + img_new_width, :] = img_resized

    if bboxes is not None:
        bboxes = bboxes * scale
        bboxes += np.array([del_w, del_h, del_w, del_h])
        return canvas, bboxes

    else:
        return canvas
