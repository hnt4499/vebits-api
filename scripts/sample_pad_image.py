import os
import cv2
import numpy as np
import pandas as pd
import sys
import argparse
from tqdm import tqdm
from math import ceil

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

def create_sequence():
    aug_1 = iaa.Pad(
        percent=((0.15, 0.3), (0.15, 0.3), (0.15, 0.3), (0.15, 0.3)),
        pad_mode="constant",
        pad_cval=(0, 255))
    aug_2 = iaa.Pad(
        percent=((0.45, 0.6), (0.45, 0.6), (0.45, 0.6), (0.45, 0.6)),
        pad_mode="constant",
        pad_cval=(0, 255))
    aug_3 = iaa.Pad(
        percent=((0.75, 0.9), (0.75, 0.9), (0.75, 0.9), (0.75, 0.9)),
        pad_mode="constant",
        pad_cval=(0, 255))

    return aug_1, aug_2, aug_3


def get_bboxes(df, imgs_name, img_shape):
    bboxes_iaa = []
    for img_name in imgs_name:
        img_data = df[df.filename == img_name]
        bboxes = img_data.loc[:, ["xmin", "ymin", "xmax", "ymax"]].to_numpy()
        bboxes = BoundingBoxesOnImage.from_xyxy_array(bboxes, shape=img_shape)
        bboxes_iaa.append(bboxes)

    return bboxes_iaa


def get_df_aug(df_base, imgs, dest_dir):
    df_out = pd.DataFrame(columns=df_base.columns)

    img_name = df_base.filename.tolist()[0]
    name, ext = os.path.splitext(img_name)

    for i in range(3):
        img_new_name = name + "_{}".format(i) + ext
        img_new_path = os.path.join(dest_dir, img_new_name)
        cv2.imwrite(img_new_path, imgs[i])

        df_base.loc[:, "filename"] = img_new_name
        df_out = pd.concat([df_out, df_base.copy()], ignore_index=True)

    return df_out


def main(args):
    src_dir = args.src_dir
    dest_dir = args.dest_dir
    csv_input_path = args.csv_input_path
    csv_output_path = args.csv_output_path

    aug_1, aug_2, aug_3 = create_sequence()
    batch_size = args.batch_size

    df = pd.read_csv(csv_input_path).sort_values("filename")
    if args.width is not None and args.height is not None:
        df = df[(df.width == args.width) & (df.height == args.height)]
    df_out = pd.DataFrame(columns=df.columns)
    img_list = df.filename.unique()

    assert df.width.nunique() == 1
    assert df.height.nunique() == 1

    img_shape = (df.loc[0, "height"], df.loc[0, "width"])

    for index in tqdm(range(ceil(img_list.shape[0] / batch_size))):
        imgs_name = img_list[index * batch_size:(index + 1) * batch_size]
        imgs_data = df[df.filename.isin(imgs_name)]

        imgs = [cv2.imread(os.path.join(src_dir, img_name)) for img_name in imgs_name]
        bboxes = get_bboxes(imgs_data, imgs_name, img_shape)

        imgs_aug_1, bboxes_aug_1 = aug_1(images=imgs, bounding_boxes=bboxes)
        imgs_aug_2, bboxes_aug_2 = aug_2(images=imgs, bounding_boxes=bboxes)
        imgs_aug_3, bboxes_aug_3 = aug_3(images=imgs, bounding_boxes=bboxes)

        # Iterate over augmented images of each image.
        for i, img_name in enumerate(imgs_name):
            img_1 = imgs_aug_1[i]
            img_2 = imgs_aug_2[i]
            img_3 = imgs_aug_3[i]

            bboxes_1 = bboxes_aug_1[i].to_xyxy_array(np.int)
            bboxes_2 = bboxes_aug_2[i].to_xyxy_array(np.int)
            bboxes_3 = bboxes_aug_3[i].to_xyxy_array(np.int)

            img_data = imgs_data[imgs_data.filename == img_name]
            df_aug = get_df_aug(img_data, [img_1, img_2, img_3], dest_dir)
            df_aug.loc[:, ["xmin", "ymin", "xmax", "ymax"]] = np.vstack([bboxes_1, bboxes_2, bboxes_3])

            df_out = pd.concat([df_out, df_aug], ignore_index=True)

    df_out.to_csv(csv_output_path, index=False)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('csv_input_path', type=str,
        help='Path to the original csv file.')
    parser.add_argument('src_dir', type=str,
        help='Directory to the original images.')
    parser.add_argument('csv_output_path', type=str,
        help='Path to which the result csv file will be saved.')
    parser.add_argument('dest_dir', type=str,
        help='Directory to which all padded images will be saved.')
    parser.add_argument('--batch_size', type=int, default=32,
        help='Batch size to perform augmentation.')
    parser.add_argument('--width', type=int, default=None,
        help='Width of images to filter and sample from original ones. \
        Default=None means that it will sample all images\
        (and raise Error if all images are not of the same shape).')
    parser.add_argument('--height', type=int, default=None,
        help='Height of images to filter and sample from original ones. \
        Default=None means that it will sample all images\
        (and raise Error if all images are not of the same shape).')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
