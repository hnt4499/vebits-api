from tf_utils.bbox_util import BBoxes
import pandas as pd
import argparse
import sys
import os
from tqdm import tqdm

def main(args):
    img_dir = args.img_dir
    dest_dir = img_dir if args.dest_dir is None else args.dest_dir

    df = pd.read_csv(args.csv_path)
    img_list = sorted(df.filename.unique())

    for img_name in tqdm(img_list):
        img_path = os.path.join(img_dir, img_name)
        xml_name = os.path.splitext(img_name)[0] + ".xml"
        xml_path = os.path.join(dest_dir, xml_name)

        bboxes = BBoxes(df=df[df.filename == img_name])
        bboxes.to_xml(img_path, xml_path)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('csv_path', type=str,
        help='Path to the csv file.')
    parser.add_argument('img_dir', type=str,
        help='Directory to images.')
    parser.add_argument('--dest_dir', type=str, default=None,
        help='Directory to which all xml files will be saved. '
             'If not specified, then xml files will be save to `img_dir`')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
