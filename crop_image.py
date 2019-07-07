import os
import glob
import pandas as pd
from PIL import Image
import datetime
import sys
import argparse
from tqdm import tqdm


def convert(x):
    return float(x) if "." in x else int(x)


def main(args):
    margin = [args.left, args.top, args.right, args.bottom]
    margin = [convert(i) for i in margin]
    # Check validity
    valid = [isinstance(i, int) for i in margin]
    if 0 < sum(valid) < 4:
        raise ValueError("All the values must be of the same instance (integer or float).")


    img_dir = args.img_dir
    img_list = os.listdir(img_dir)
    dest_dir = args.dest_dir

    for img_name in tqdm(img_list):
        if not img_name.endswith("jpg"):
            continue
        img = Image.open(os.path.join(img_dir, img_name))
        width, height = img.size

        # If percentage
        if sum(valid) == 0:
            margin_img = [margin[0] * width, margin[1] * height, margin[2] * width, margin[3] * height]

        else: margin_img = margin
        img_aug = img.crop((margin_img[0],
                            margin_img[1],
                            width - margin_img[2],
                            height - margin_img[3])
                        ).resize((width, height), Image.ANTIALIAS)

        unique_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
        img_aug.save(os.path.join(dest_dir, unique_id + ".jpg"))


def parse_arguments(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('img_dir', type=str,
        help='Directory to all images.')
    parser.add_argument('dest_dir', type=str,
        help='Directory to which all images will be saved.')

    parser.add_argument('top', type=str,
        help='Percent (float) or Pixel (integer)')
    parser.add_argument('right', type=str,
        help='Percent (float) or Pixel (integer).')
    parser.add_argument('bottom', type=str,
        help='Percent (float) or Pixel (integer).')
    parser.add_argument('left', type=str,
        help='Percent (float) or Pixel (integer).')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
