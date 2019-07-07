import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import sys
import argparse


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(float(root.find('size')[0].text)),
                     int(float(root.find('size')[1].text)),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main(args):
    image_path = os.path.join(os.getcwd(), args.src_dir)
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv(args.dest_path, index=None)
    print('Successfully converted xml to csv.')


def parse_arguments(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('src_dir', type=str,
        help='Directory to all the images and their labels.')
    parser.add_argument('dest_path', type=str,
        help='Path to which the csv file will be saved.')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
