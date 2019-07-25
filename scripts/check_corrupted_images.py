import os
import argparse
import sys

import pandas as pd
from tqdm import tqdm
import tensorflow as tf

DESCRIPTION = """This checks for any corrupted images accidentally produced
during data preparation. This is particularly useful when seeing an error like
[[node ...]]. For best performance, run this script iteratively for about
"""

def main(args):
    # Reading data
    df = pd.read_csv(args.csv_path)
    img_list = sorted(df.filename.unique())[args.start:]
    # Preparing
    img_dir = args.img_dir
    batch_size = args.batch_size
    batch = []

    print(">>> Checking {} images in {}".format(len(img_list), img_dir))
    # Start checking images
    with tf.Graph().as_default():
        init_op = tf.initialize_all_tables()
        with tf.Session() as sess:
            sess.run(init_op)
            for i, img_name in enumerate(tqdm(img_list)):
                img_contents = tf.read_file(os.path.join(img_dir, img_name))
                img = tf.image.decode_jpeg(img_contents, channels=3)
                batch.append(img)
                if (i + 1) % batch_size:
                    try:
                        sess.run(batch)
                    except:
                        if batch_size == 1:
                            print(">>> Found corrupted image: "
                                  "{}".format(img_name))
                        else:
                            print(">>> Found corrupted image(s) at batch "
                                  "{}-{}".format(i - batch_size, i))
                    batch = []


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('csv_path', type=str,
        help='Path to the first inference graph.')
    parser.add_argument('img_dir', type=str,
        help='Path to the label map of the first model.')
    parser.add_argument('--start', type=int, default=0,
        help='Where to start checking.')
    parser.add_argument('--batch_size', type=int, default=128,
        help='How many images to aggregate per batch.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
