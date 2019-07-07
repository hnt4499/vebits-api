import tensorflow as tf
import os
import argparse
import sys
import pandas as pd
from tqdm import tqdm


def main(args):

    df = pd.read_csv(args.csv_path)
    img_list = sorted(df.filename.unique())
    img_errors = []
    img_dir = args.img_dir
    print(">>> Checking {} images in {}".format(img_list.shape[0], img_dir))
    
    with tf.Graph().as_default():
        init_op = tf.initialize_all_tables()
        with tf.Session() as sess:
            sess.run(init_op)
            for img_name in tqdm(img_list):
                img_contents = tf.read_file(os.path.join(img_dir, img_name))
                img = tf.image.decode_jpeg(img_contents, channels=3)
                try:
                    sess.run(img)
                except:
                    img_errors.append(img_name)

    if len(img_errors) == 0:
        print(">>> No corrupted images found.")
    else:
        print(">>> List of corrupted image(s):")
        for img_name in img_errors:
            print(">>> {}".format(img_name))


def parse_arguments(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('csv_path', type=str,
        help='Path to the first inference graph.')
    parser.add_argument('img_dir', type=str,
        help='Path to the label map of the first model.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
