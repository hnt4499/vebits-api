import os
import sys
import argparse
from xml_util import shift_bboxes_and_save

def main(args):
    xml_src_dir = args.xml_src_dir
    xml_dest_dir = args.xml_dest_dir
    x_value = args.x_value
    y_value = args.y_value

    file_list = os.listdir(xml_src_dir)
    for file_name in file_list:
        if not file_name.endswith("xml"):
            continue

        shift_bboxes_and_save(
            xml_src_path=os.path.join(xml_src_dir, file_name),
            xvalue=x_value,
            yvalue=y_value,
            xml_dest_path=os.path.join(xml_dest_dir, file_name))
    print("Done.")


def parse_arguments(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('xml_src_dir', type=str,
        help='Directory containing all xml files.')
    parser.add_argument('xml_dest_dir', type=str,
        help='Directory to which all xml files will be saved.')
    parser.add_argument('x_value', type=int,
        help='Value to shift along the x-axis.')
    parser.add_argument('y_value', type=int,
        help='value to shift along the y-axis.')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
