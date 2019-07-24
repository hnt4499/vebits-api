import cv2
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa

from .bbox_util import BBox, BBoxes


def resize_padding(img, size, bboxes=None):
    """Resize image to desired size while keep aspect ratio by adding
    padding to necessary edges.

    Parameters
    ----------
    img : ndarray
        Image to be resized. Must have three channels.
    size : tuple-like
        Resulting image size.
    bboxes : ndarray, BBox or BBoxes
        Bounding box(es) of type ndarray, BBox or BBoxes. If ndarray, it
        must have shape of (4,) or (n, 4).

    """
    # Generate a canvas for storing resulting image
    canvas = np.full((size[0], size[1], 3), 0, dtype=np.uint8)
    img_height, img_width = img.shape[:2]
    # Calculate scale
    scale = min(size[0] / img_height, size[1] / img_width)
    # Calculate new parameters for new image
    img_new_height = int(scale * img_height)
    img_new_width = int(scale * img_width)
    # Resize by aspect ratio
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


def create_sequence():
    """
    This function creates a sequence of image transformation (augmentation)
    for training process.
    """
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


def save_imgs(imgs, img_save_paths):
    """
    Convenience function used to save images by batch.
    """
    for img, save_path in zip(imgs, img_save_paths):
        cv2.imwrite(save_path, img)
