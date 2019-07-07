import xml.etree.ElementTree as ET
import os

def create_xml_file(img_path, img_width, img_height, bbox_list, xml_path=None):

    # create the file structure
    annotate = ET.Element('annotation')
    folder = ET.SubElement(annotate, 'folder')
    filename = ET.SubElement(annotate, 'filename')
    path = ET.SubElement(annotate, 'path')
    source = ET.SubElement(annotate, 'source')
    database = ET.SubElement(source, 'database')
    size = ET.SubElement(annotate, 'size')
    width = ET.SubElement(size, 'width')
    height = ET.SubElement(size, 'height')
    depth = ET.SubElement(size, 'depth')
    segmented = ET.SubElement(annotate, 'segmented')

    folder_, img_name = os.path.split(img_path)

    folder.text = str(folder_)
    filename.text = str(img_name)
    path.text = str(img_path)
    database.text = 'Unknown'
    width.text = str(img_width)
    height.text = str(img_height)
    depth.text = '3'
    segmented.text = '0'

    # Values in object are dynamic
    for bbox in bbox_list:
        obj = ET.SubElement(annotate, 'object')
        name = ET.SubElement(obj, 'name')
        pose = ET.SubElement(obj, 'pose')
        truncated = ET.SubElement(obj, 'truncated')
        difficult = ET.SubElement(obj, 'difficult')
        bndbox = ET.SubElement(obj, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        ymin = ET.SubElement(bndbox, 'ymin')
        xmax = ET.SubElement(bndbox, 'xmax')
        ymax = ET.SubElement(bndbox, 'ymax')

        name.text = str(bbox.get_label())
        pose.text = 'Unspecified'
        truncated.text = '0'
        difficult.text = '0'

        xmin.text = str(bbox.get_xmin())
        ymin.text = str(bbox.get_ymin())
        xmax.text = str(bbox.get_xmax())
        ymax.text = str(bbox.get_ymax())

    # create a new XML file with the results
    mydata = ET.tostring(annotate, encoding="unicode")

    if xml_path is None:
        xml_path = os.path.splitext(img_path)[0] + ".xml"
    with open(xml_path, "w") as myfile:
        myfile.write(mydata)


def change_label(xml_root, label_src, label_dest):
    for obj in xml_root.findall("object"):
        if obj[0].text == label_src:
            obj[0].text = label_dest
    return xml_root


def change_label_and_save(xml_src_path, label_src, label_dest, xml_dest_path=None):
    xml_tree = ET.parse(xml_src_path)
    change_label(xml_tree.getroot(), label_src, label_dest)
    if xml_dest_path is None:
        xml_tree.write(xml_src_path)
    else:
        xml_tree.write(xml_dest_path)


def prune_label(xml_root, label, num_to_keep):
    count = 0
    obj_list = []
    # Create a list of objects. Deleting objects right from the xml_root
    # might result in different behavior of the genrators.
    for obj in xml_root.findall("object"):
        obj_list.append(obj)

    for obj in obj_list:
        if obj[0].text != label:
            continue
        if count < num_to_keep:
            count += 1
        else:
            xml_root.remove(obj)

    return xml_root


def prune_label_and_save(xml_src_path, label, num_to_keep, xml_dest_path=None):
    xml_tree = ET.parse(xml_src_path)
    prune_label(xml_tree.getroot(), label, num_to_keep)
    if xml_dest_path is None:
        xml_tree.write(xml_src_path)
    else:
        xml_tree.write(xml_dest_path)


def shift(point, value, max_value, min_value=0):
    point_new = int(point) + value
    if point_new > max_value:
        point_new = max_value
    if point_new < min_value:
        point_new = min_value
    return str(point_new)


def shift_bbox(bbox_obj, xvalue, yvalue, img_size):
    for i, point in enumerate(bbox_obj):
        if i % 2 == 0:
            point.text = shift(point.text, xvalue, img_size[1])
        else:
            point.text = shift(point.text, yvalue, img_size[0])


def get_size(root_obj):
    size = root_obj.find("size")
    return (int(size[1].text), int(size[0].text))


def shift_bboxes_and_save(xml_src_path, xvalue, yvalue, xml_dest_path=None):
    xml_tree = ET.parse(xml_src_path)
    xml_root = xml_tree.getroot()
    img_size = get_size(xml_root)

    for obj in xml_root.findall("object"):
        shift_bbox(obj.find("bndbox"), xvalue, yvalue, img_size)

    if xml_dest_path is None:
        xml_tree.write(xml_src_path)
    else:
        xml_tree.write(xml_dest_path)
