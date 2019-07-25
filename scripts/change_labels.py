import os
import sys
import argparse
import glob

from tqdm import tqdm

from vebits_api.xml_util import change_label_and_save

DESCRIPTION = """This script changes all labels in *.xml files produced by
`labelimg` to a specific label. This is particularly useful when one wants
to collect the dataset for a single class and use a pretrained model to label,
since the model can misclassify sometimes.
"""


def main(args):
    xml_list = []
    # Read arguments
    xml_src_dir = args.xml_src_dir
    xml_dest_dir = args.xml_dest_dir
    label_src = args.label_src
    label_dest = args.label_dest
    # Generate list of xml files
    for xml_name in glob.glob("{}/*.xml".format(xml_src_dir)):
        xml_list.append(os.path.split(xml_name)[1])

    for xml_name in tqdm(xml_list):
        xml_src_path = os.path.join(xml_src_dir, xml_name)
        xml_dest_path = os.path.join(xml_dest_dir, xml_name)
        change_label_and_save(xml_src_path=xml_src_path,
                              xml_dest_path=xml_dest_path,
                              label_src=label_src,
                              label_dest=label_dest)
    print("Successfully change all labels from "
          "{} to {}".format(label_src, label_dest))

def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=DESCRIPTION)

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
