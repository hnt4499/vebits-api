
import os
import sys
import argparse
from tqdm import tqdm
from tf_utils.xml_util import change_label_and_save
import glob


def main(args):
    xml_list = []
    xml_src_dir = args.xml_src_dir
    xml_dest_dir = args.xml_dest_dir
    label_src = args.label_src
    label_dest = args.label_dest

    for xml_name in glob.glob("{}/*.xml".format(xml_src_dir)):
        xml_list.append(os.path.split(xml_name)[1])

    for xml_name in tqdm(xml_list):
        xml_src_path = os.path.join(xml_src_dir, xml_name)
        xml_dest_path = os.path.join(xml_dest_dir, xml_name)
        change_label_and_save(xml_src_path=xml_src_path,
                              xml_dest_path=xml_dest_path,
                              label_src=label_src,
                              label_dest=label_dest)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('xml_src_dir', type=str,
        help='Directory to xml files.')
    parser.add_argument('xml_dest_dir', type=str,
        help='Directory to which all modified xml files will be saved.')
    parser.add_argument('label_src', type=str,
        help='Directory to xml files.')
    parser.add_argument('label_dest', type=str,
        help='Directory to which all modified xml files will be saved.')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
