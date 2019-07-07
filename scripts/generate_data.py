import sys
import numpy as np
import pandas as pd
import cv2
import os
import argparse
from shutil import rmtree
import time
from tqdm import tqdm

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

import warnings
warnings.filterwarnings("ignore")

def create_sequence():
    sequence = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),
        iaa.Multiply((0.7, 1.3)),
        iaa.Grayscale(alpha=(0.0, 1.0)),
        iaa.AddToHueAndSaturation((-15, 15)),
        iaa.Affine(translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)}),
        iaa.Crop(percent=((0, 0.1), (0, 0.2), (0, 0.1), (0, 0.2))),
        ], random_order=False)
    return sequence


def get_bboxes(df, img_name):
    bboxes = np.asarray(df[df.filename == img_name].iloc[:, 4:])
    return [BoundingBox(*i) for i in bboxes]

# Function to copy data of original image to new image.
def copy_data(orig_data, img_name, bboxes):
    transformed_data = pd.DataFrame(
        [orig_data.iloc[0, :]] * bboxes.shape[0],
        columns=orig_data.columns)

    transformed_data[["xmin", "ymin", "xmax", "ymax"]] = bboxes
    transformed_data.filename = img_name
    return transformed_data

# Function to calculate the fraction of the bounding boxes with relative to the original image.
def cal_bboxes_fraction(orig_area, bboxes):
    bboxes = np.asarray(bboxes, dtype=float)
    bboxes_area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    return bboxes_area / orig_area * 100

# Function to transform all images in src_dir, save transformed images to dest_dir,
# update images data in df and save it to csv_output_dir.


def transform(csv_input_path, src_dir,
              dest_dir, area_threshold,
              csv_output_path=None,
              num_transform=5, test_mode=False,
              keep_orig_img=False):

    df = pd.read_csv(csv_input_path)
    df_drop_duplicates = df.drop_duplicates()
    print(">>> Original df: {} objects".format(df.shape[0]))
    print(">>> Dropped-duplicates df: {} objects".format(df_drop_duplicates.shape[0]))
    df_out = pd.DataFrame(columns=df.columns)
    img_list = sorted(df.filename.unique())

    skip_gen_img = 0

    sequence = create_sequence()

    if test_mode:
        img_list = img_list[:30]
        print(">>> Testing with 30 images.")
    else:
        expected = (num_transform + 1) * df.shape[0] if keep_orig_img else num_transform * df.shape[0]
        print('>>> Number of expected objects: {}'.format(expected))

    with tqdm(img_list, unit="imgs") as t:
        for img in t:
            t.set_postfix(skipped_gen_imgs=skip_gen_img)
            if not img.lower().endswith("jpg"):
                continue

            name, ext = os.path.splitext(img)

            img_data = df_drop_duplicates[df_drop_duplicates.filename == img]
            num_duplicates = int(df[df.filename == img].shape[0] / img_data.shape[0])
            img_shape = (img_data.iloc[0, 2], img_data.iloc[0, 1])
            img_area =  img_shape[0] * img_shape[1]

            bboxes = np.asarray(img_data.iloc[:, 4:], dtype=np.int)
            bboxes_iaa = BoundingBoxesOnImage.from_xyxy_array(bboxes, shape=img_shape)

            suffix = 0
            img_array = cv2.imread(os.path.join(src_dir, img))

            # If keeping original image and the bounding boxes satisfy the threshold condition.
            if keep_orig_img and all(cal_bboxes_fraction(img_area, bboxes) >= area_threshold):
                df_out = df_out.append(img_data, ignore_index=True)
                cv2.imwrite(os.path.join(dest_dir, img), img_array)

            for _ in range(num_transform * num_duplicates):
                transformed_img, transformed_bboxes = sequence(image=img_array, bounding_boxes=bboxes_iaa)

                for transformed_bbox in transformed_bboxes.bounding_boxes:
                    if not transformed_bbox.is_fully_within_image(transformed_img):
                        skip_gen_img += 1
                        break

                else:
                    transformed_bboxes = transformed_bboxes.to_xyxy_array(dtype=np.int)
                    if any(cal_bboxes_fraction(img_area, transformed_bboxes) < area_threshold):
                        skip_gen_img += 1
                        continue

                    transformed_name = name + "_{}".format(suffix) + ext
                    transformed_path = os.path.join(dest_dir, transformed_name)
                    # When the file exists
                    while os.path.isfile(transformed_path):
                        suffix += 1
                        transformed_name = name + "_{}".format(suffix) + ext
                        transformed_path = os.path.join(dest_dir, transformed_name)


                    transformed_data = copy_data(img_data, transformed_name, transformed_bboxes)
                    df_out = df_out.append(transformed_data, ignore_index=True)

                    cv2.imwrite(os.path.join(dest_dir, transformed_name), transformed_img)
                    suffix += 1
    print('''
=================================
Total images processed: {}
Total images generated: {}
Total generated images skipped: {}
=================================

'''.format(len(img_list),
           len(os.listdir(dest_dir)),
           skip_gen_img))

    # Shuffle the DataFrame
    df_out = df_out.sample(frac=1.0).reset_index(drop=True)
    if csv_output_path is not None:
        df_out.to_csv(csv_output_path, index=False)
    return df


def main(args):
    src_dir = args.src_dir
    dest_dir = args.dest_dir
    num_transform = args.num_transform
    test_mode = args.test_mode
    keep_orig_img = args.keep_orig_image
    area_threshold = args.area_threshold

    # Transform data and generate tfrecord files.
    for csv, folder_name in zip(["train_labels.csv"], ["combined"]):
        csv_in = os.path.join(src_dir, csv)
        csv_out = os.path.join(dest_dir, csv)
        src = os.path.join(src_dir, folder_name)
        dest = os.path.join(dest_dir, "train")

        # Remove old generated images
        if os.path.isdir(dest):
            rmtree(dest)
        os.mkdir(dest)

        # Transform images.
        transform(csv_input_path=csv_in,
                  src_dir=src,
                  dest_dir=dest,
                  area_threshold=area_threshold,
                  csv_output_path=csv_out,
                  num_transform=num_transform,
                  test_mode=test_mode,
                  keep_orig_img=keep_orig_img)

        print("Successfully generated images to {} and csv file to {} ...".format(dest, csv_out))

        # Generate tfrecord.
        tf_record_out = os.path.join(dest_dir, "train" + ".record")
        os.system("python generate_tfrecord.py --csv_input={} --image_dir={} --output_path={}".format(csv_out, dest, tf_record_out))

def parse_arguments(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('src_dir', type=str,
        help='Directory to the original dataset, including csv files.')
    parser.add_argument('dest_dir', type=str,
        help='Directory to which the transformed images (including train and \
        test set and their labels, each in a separate subfolder) will be saved.')
    parser.add_argument('num_transform', type=int,
        help='Number of transformed images per image.')
    parser.add_argument('area_threshold', type=float,
        help='Minimum percentage of area allowed of bounding box with relative to the image.')
    parser.add_argument('--test_mode', action="store_true",
        help='Whether to enter test mode.')
    parser.add_argument('--batch_size', type=int, default=128,
        help='Number of images to perform augmentation at a time.')
    parser.add_argument('--keep_orig_image', action="store_true",
        help='Whether to include the original image.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
