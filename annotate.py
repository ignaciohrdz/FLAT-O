# FLATO: Facial Landmark AnnoTation with OpenCV
# Author: Ignacio Hernández Montilla
# https://github.com/ignaciohrdz

import os
import argparse
import random
random.seed(420)

import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from xml.dom import minidom
import xml.etree.ElementTree as ET


face_parts = {'jaw_line': [1, 17],  # 0
              'left_eyebrow': [18, 22],  # 1
              'right_eyebrow': [23, 27],  # 2
              'nasal_bridge': [28, 31],  # 3
              'nostrils': [32, 36],  # 4
              'left_eye_upper': [37, 40],  # 5
              'left_eye_lower': [41, 42],  # 6
              'right_eye_upper': [43, 46],  # 7
              'right_eye_lower': [47, 48],  # 8
              'upper_lip_outer': [49, 55],  # 9
              'lower_lip_outer': [56, 60],  # 10
              'upper_lip_inner': [61, 65],  # 11
              'lower_lip_inner': [66, 68]}  # 12

face_part_keys = list(face_parts.keys())
current_annotations = {0: {k: [] for k in face_part_keys}}
current_face_id = 0
current_face = {0: []}
current_image = ""
current_image_size = []
ratio = 1

# For event handling
current_part = 0
current_part_points_x = []
current_part_points_y = []
hold_click = False

# Input style (freehand/polyline) and curve fitting (on/off)
draw_mode = False
fit_curve = False


def smart_resize(img, new_size=512):
    ratio = new_size / max(img.shape[:2])
    img = cv2.resize(img, None, fx=ratio, fy=ratio)
    return img, ratio


def change_face_part(diff):
    global current_part, face_part_keys
    global current_part_points_x, current_part_points_y

    if diff > 0:
        current_part = min(current_part+diff, len(face_part_keys)-1)
    else:
        current_part = max(current_part+diff, 0)
    current_part_points_x = []
    current_part_points_y = []


def change_face(diff):
    global current_face_id, current_annotations, current_part
    global current_part_points_x, current_part_points_y

    if diff > 0:
        current_face_id = min(current_face_id + diff, len(current_annotations.keys()))
    else:
        current_face_id = max(current_face_id + diff, 0)

    if current_face_id not in current_annotations.keys():
        current_annotations[current_face_id] = {k: [] for k in face_part_keys}

    current_part_points_x = []
    current_part_points_y = []
    current_part = 0


def process_selected_points():
    global current_part_points_x, current_part_points_y, current_annotations, current_face_id
    global face_part_keys, current_part, fit_curve

    part_ref = face_parts[face_part_keys[current_part]]
    steps = part_ref[1] - part_ref[0] + 1

    # Special case for lower eye (left and right) and lower lips
    if current_part in [6, 8, 10, 12]:
        steps += 2

    # Drawing equidistant points along the curve
    # Method explained here:
    # https://stackoverflow.com/questions/51512197/python-equidistant-points-along-a-line-joining-set-of-points
    x_arr = np.array(current_part_points_x)

    if fit_curve:
        # My idea: make the drawn line "softer" by fitting a polynomial regression model
        # https://www.statology.org/curve-fitting-python/
        model = np.poly1d(np.polyfit(current_part_points_x, current_part_points_y, 5))
        y_arr = np.array([model(x_i) for x_i in current_part_points_x])
        y_arr[0] = current_part_points_y[0]
        y_arr[-1] = current_part_points_y[-1]
    else:
        y_arr = np.array(current_part_points_y)

    # Computing total line distance
    diff_x = np.ediff1d(x_arr, to_begin=0)
    diff_y = np.ediff1d(y_arr, to_begin=0)
    distance = np.cumsum(np.sqrt(diff_x ** 2 + diff_y ** 2))
    distance = distance / distance[-1]

    # Obtaining the equidistant points
    fx, fy = interp1d(distance, x_arr), interp1d(distance, y_arr)
    alpha = np.linspace(0, 1, steps)
    x_regular, y_regular = fx(alpha), fy(alpha)

    # Adding the points
    for i, x_i in enumerate(x_regular.tolist()):
        if current_part in [6, 8, 10, 12]:  # lower eyelid (left and right)
            if i not in [0, len(x_regular.tolist())-1]:
                current_annotations[current_face_id][face_part_keys[current_part]].append([int(x_i), int(y_regular[i])])
        else:
            current_annotations[current_face_id][face_part_keys[current_part]].append([int(x_i), int(y_regular[i])])

    current_part_points_x = []
    current_part_points_y = []


def render_image(input_image, font=cv2.FONT_HERSHEY_SIMPLEX):
    global face_part_keys, current_part, current_annotations, current_face, current_face_id
    global draw_mode, fit_curve

    # Adding a black area to the right side or the bottom of the image to show some info
    h, w = input_image.shape[:2]
    black_frame = np.zeros((input_image.shape[0], 500, 3), dtype=np.uint8)
    input_image = np.hstack([input_image, black_frame])
    x = int(1.10 * w)
    y = int(0.05 * h)

    # Showing the current settings
    m = "Freehand" if draw_mode else "Polyline"
    f = "ON" if fit_curve else "OFF"
    cv2.putText(input_image, "Face: " + str(current_face_id), (x, y), font, 0.75, (0, 255, 255), 2)
    cv2.putText(input_image, "Part: " + face_part_keys[current_part], (x, y+30), font, 0.75, (0, 255, 255), 2)
    cv2.putText(input_image, "Mode: " + m, (x, y+60), font, 0.75, (0, 255, 255), 2)
    cv2.putText(input_image, "Fit curve: " + f, (x, y+90), font, 0.75, (0, 255, 255), 2)

    # Drawing each part
    for k, points in current_annotations[current_face_id].items():
        if len(points) > 0:
            for p in points:
                cv2.circle(input_image, (p[0], p[1]), 3, (0, 255, 255), -1)

    # Drawing the currently selected points
    for i, x_i in enumerate(current_part_points_x):
        cv2.circle(input_image, (x_i, current_part_points_y[i]), 3, (255, 255, 0), -1)
        if i > 0:
            cv2.line(input_image, (current_part_points_x[i-1], current_part_points_y[i-1]),
                     (current_part_points_x[i], current_part_points_y[i]),
                     (255, 255, 0), 1)

    # Drawing the bounding box of the face (from the landmarks)
    for f_id, f_box in current_face.items():
        if len(f_box) > 0:
            face_x, face_y, face_w, face_h = f_box
            cv2.rectangle(input_image, (face_x, face_y), (face_x + face_w, face_y + face_h), (0, 0, 255), 2)

    return input_image


def mouse_click(event, x, y, flags, param):
    global current_part_points_x, current_part_points_y
    global show_img, hold_click, draw_mode

    x = min(max(x, 0), param[1])
    y = min(max(y, 0), param[0])

    if event == cv2.EVENT_LBUTTONDOWN:
        if draw_mode:
            hold_click = True
        current_part_points_x.append(x)
        current_part_points_y.append(y)
    elif event == cv2.EVENT_LBUTTONUP and draw_mode:
        hold_click = False

        process_selected_points()
        if check_face_is_complete():
            get_face_bbox_from_points()
        change_face_part(1)

    elif hold_click and draw_mode:
        current_part_points_x.append(x)
        current_part_points_y.append(y)
    else:
        pass


def get_face_bbox_from_points(expand_box=True, expand_pct=0.05):
    global current_annotations, current_face, current_face_id

    all_points = []
    for k, v in current_annotations[current_face_id].items():
        for x, y in v:
            all_points.extend([[x, y]])
    all_points_arr = np.array(all_points, dtype=np.int32)
    bbox = list(cv2.boundingRect(all_points_arr))

    # Expanding the bounding box of the face (optional)
    if expand_box:
        bbox[0] = max(0, bbox[0] - int(bbox[2]*expand_pct))
        bbox[1] = max(0, bbox[1] - int(bbox[3]*expand_pct))
        bbox[2] = int(bbox[2]*(1+2*expand_pct))
        bbox[3] = int(bbox[3]*(1+2*expand_pct))

    current_face[current_face_id] = bbox


def check_face_is_complete():
    global current_annotations, current_face_id
    check = any([len(v) == 0 for _, v in current_annotations[current_face_id].items()])
    return not check


def generate_xml(split_type='training'):
    global args

    root = ET.Element("dataset")
    name_tag = ET.Element("name")
    name_tag.text = os.path.basename(args.img_path)
    root.append(name_tag)

    comment_tag = ET.Element("comment")
    comment_tag.text = "This file contains the annotations of the {} images from {}".format(split_type, args.img_path)
    root.append(comment_tag)

    images_tag = ET.Element("images")
    root.append(images_tag)

    tree = ET.ElementTree(root)

    return tree


def check_image_is_annotated(fname):
    global xml_trees
    check_image = any([t.getroot().find(".//*[@file='{}']".format(fname)) is not None for _, t in xml_trees.items()])
    return check_image


def update_xml(split_name='training'):
    global xml_trees, current_face, current_annotations
    global current_image, current_image_size, ratio

    images_tag = xml_trees[split_name].getroot().find("./images")

    new_image_tag = ET.SubElement(images_tag, "image")
    new_image_tag.set("file", current_image)
    for i, a in enumerate(['height', 'width']):
        new_image_tag.set(a, str(current_image_size[-(i+1)]))

    for face_id, face_xywh in current_face.items():
        if len(face_xywh) > 0:
            box_tag = ET.SubElement(new_image_tag, "box")

            for i, a in enumerate(['left', 'top', 'width', 'height']):
                box_tag.set(a, str(face_xywh[i]))

            # Print all the landmark in XML scaling the coordinates to the original size
            part_count = 0
            for f_part, part_coords in current_annotations[face_id].items():
                for c in part_coords:
                    part_tag = ET.SubElement(box_tag, "part")
                    part_tag.set("name", str(part_count).zfill(2))
                    part_tag.set("x", str(int(c[0]*ratio)))
                    part_tag.set("y", str(int(c[1]*ratio)))
                    part_count += 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img-path', type=str, help="Path to the image folder")
    parser.add_argument('-x', '--xml-path', type=str, help="Path to the folder where XMLs are saved")
    parser.add_argument('-d', '--display-size', type=int, default=896, help="Display size (largest image size)")
    parser.add_argument('-t', '--test-pct', type=int, default=10, help="Percentage of images for the test set (0-100)")
    parser.add_argument('--no-splits', action='store_true', help="Do not make train/test splits and make one single XML")
    args = parser.parse_args()

    # This is for debugging on my machine. Comment these two lines before running on yours
    # args.img_path = "D:/ML Projects/Datasets/Faces_dataset"
    # args.xml_path = "data"
    # args.no_splits = True

    # Checking the XML paths
    # Option 1: Specify a folder for the XMLs
    if (args.xml_path is None) and args.no_splits:
        path_xml = {'training': 'annotations.xml'}
    elif (args.xml_path is not None) and os.path.isdir(args.xml_path) and args.no_splits:
        path_xml = {'training': os.path.join(args.xml_path, 'annotations.xml')}
    elif (args.xml_path is None) and not args.no_splits:
        path_xml = {'training': 'annotations_train.xml',
                    'test': 'annotations_test.xml'}
    elif (args.xml_path is not None) and os.path.isdir(args.xml_path) and not args.no_splits:
        path_xml = {'training': os.path.join(args.xml_path, 'annotations_train.xml'),
                    'test': os.path.join(args.xml_path, 'annotations_test.xml')}
    # Option 2: Specify the path of the XML file, not the folder
    elif os.path.isfile(args.xml_path) and args.no_splits:
        path_xml = {'training': args.xml_path}
        args.xml_path = os.path.dirname(args.xml_path)

    # Creating/reading the XMl file/s
    xml_trees = {}
    for split_name, xml_name in path_xml.items():
        if not os.path.isfile(xml_name):
            xml_trees[split_name] = generate_xml(split_type=split_name)
            with open(xml_name, "w") as xml_file:
                # Source: https://stackoverflow.com/a/28814053/8591713
                xml_str = minidom.parseString(ET.tostring(xml_trees[split_name].getroot())).toprettyxml()
                xml_file.write(xml_str)
        else:
            xml_trees[split_name] = ET.parse(xml_name)

    # Splitting the data into training/validation and saving the split data in a CSV file
    # If the file already exists, we don't have to create the splits again
    if os.path.isfile(os.path.join(args.xml_path, "dataset_info.csv")):
        dataset_info = pd.read_csv(os.path.join(args.xml_path, "dataset_info.csv"))
    else:
        images = sorted(os.listdir(args.img_path))
        dataset_info = pd.DataFrame({'image': images})
        dataset_info['split'] = 'training'
        if not args.no_splits and 0 < args.test_pct < 100:
            n_test = int(len(images) * args.test_pct/100)
            idxs = list(dataset_info.index)
            random.shuffle(idxs)
            train_idx = idxs[:-n_test]
            test_idx = idxs[-n_test:]
            dataset_info.loc[train_idx, 'split'] = 'training'
            dataset_info.loc[test_idx, 'split'] = 'test'
        dataset_info.to_csv(os.path.join(args.xml_path, "dataset_info.csv"), index=False)
    dataset_info = dataset_info.set_index(['image'])

    # Main program
    exit_program = False
    for f in sorted(list(dataset_info.index)):

        # Check that it has not been already included in the XML
        # This is useful when we split annotation into several runs
        if not check_image_is_annotated(f):

            # Load the image
            img = cv2.imread(os.path.join(args.img_path, f))
            current_image_size = list(img.shape[:2])
            img, ratio = smart_resize(img, new_size=args.display_size)
            current_show_size = list(img.shape[:2])
            current_image = f
            image_split = dataset_info.loc[f, 'split']

            win_name = 'FLATO - Annotate facial landmarks'
            cv2.namedWindow(win_name)
            cv2.setMouseCallback(win_name, mouse_click, img.shape[:2])

            while True:
                show_img = render_image(img.copy())
                cv2.imshow(win_name, show_img)
                k = cv2.waitKey(1) & 0xFF

                # Changing face
                if k == ord("a"):  # previous face
                    change_face(-1)
                if k == ord("s"):  # next face (if any)
                    change_face(1)

                # Changing face part
                if k == ord("z"):  # previous part
                    change_face_part(-1)
                if k == ord("x"):  # next part
                    change_face_part(1)

                if k == ord("u"):  # undo current part
                    current_annotations[current_face_id][face_part_keys[current_part]] = []
                    current_part_points_x = []
                    current_part_points_y = []
                if k == ord("r"):  # reset the entire face
                    current_annotations[current_face_id] = {k: [] for k in face_part_keys}
                    current_part = 0
                    current_part_points_x = []
                    current_part_points_y = []

                if k == ord("m"):  # change annotation mode (drawing or clicking lines)
                    draw_mode = not draw_mode
                if k == ord("f"):  # activate/deactivate curve fitting
                    fit_curve = not fit_curve

                if k == ord(" ") and not draw_mode and len(current_part_points_x) > 1:
                    process_selected_points()
                    if check_face_is_complete():
                        get_face_bbox_from_points()
                    change_face_part(1)

                if k == ord("q"):
                    # Exit the annotation
                    exit_program = True
                    break
                if k == ord("n"):
                    # Moving onto the next image
                    break

            # Sources:
            #   - https://www.geeksforgeeks.org/create-xml-documents-using-python/ (see option 2)
            #   - https://stackoverflow.com/a/49473666/8591713
            if not exit_program:
                update_xml(image_split)
                with open(path_xml[image_split], "w") as xml_file:
                    # If we annotate in several runs, doing toprettyxml() will add too many spaces
                    # In order to keep the file readable, I must remove indents, spaces and new lines
                    xml_str = minidom.parseString(ET.tostring(xml_trees[image_split].getroot())).toprettyxml().replace("\n", "").replace("\t", "").replace("  ", "")
                    cleaned_xml = ET.fromstring(xml_str)
                    xml_str = minidom.parseString(ET.tostring(cleaned_xml)).toprettyxml()
                    xml_file.write(xml_str)

            current_part = 0
            current_part_points_x = []
            current_part_points_y = []
            current_annotations = {0: {k: [] for k in face_part_keys}}
            current_face = {0: []}
            current_face_id = 0

            if exit_program:
                break

    cv2.destroyAllWindows()
