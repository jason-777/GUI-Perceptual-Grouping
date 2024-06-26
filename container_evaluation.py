from GUI import GUI
import cv2
import json
import os
from os.path import join as pjoin
import xml.etree.ElementTree as ET  # for xml file
import element.detect_merge.merge as merge
from element.detect_merge.Element import Element
import warnings
import glob

# ------ parse vins ------
# below two path should be set to the corresponding image.
def parse_vins2dict(image_jpg_path: str, image_ant_path: str, specify_cate=None, show=False, write_image=False,
               show_name=""):
    """
    parse vins dataset.
    @param image_jpg_path: origin image file.
    @param image_ant_path: annotation file.
    @param specify_cate: specify component string category list. Vins widget names. If none, detect all.
    @param show: show annotated image.
    @param write_image: # write annotated image, write component file.
    @param show_name: write image name, if empty, then set image file name.
    @return: sComponent List.
    """
    if specify_cate is None:
        specify_cate = []
        detect_all_widget = True
    else:
        detect_all_widget = False
    img_widget_list = []  # store widget appear in image, and appear in widget list.
    # collect annotation from xml file.
    tree = ET.parse(image_ant_path)
    root = tree.getroot()
    for child in root:
        if child.tag == "object":
            # search child tag
            for content in child:
                if content.tag == "name":
                    class_name = content.text  # string
                    if (class_name in specify_cate) or detect_all_widget:
                        for content in child:
                            # The image upper left corner is the origin point.
                            if content.tag == "bndbox":
                                xmin = int(float(content[0].text))  # x is on width axis
                                ymin = int(float(content[1].text))  # y is on height axis
                                xmax = int(float(content[2].text))
                                ymax = int(float(content[3].text))
                                widget_info = [class_name, xmin, ymin, xmax, ymax]
                                img_widget_list.append(widget_info)
                                break  # get widget info, try next widget
                    break  # find widget we want, try next widget

    #  show widget bbox in image, draw rectangle
    org_image = cv2.imread(image_jpg_path)
    for widget in img_widget_list:
        # widget is [class_name, width_min, height_min, width_max, height_max]
        red_color = (0, 0, 255)  # BGR
        cv2.rectangle(org_image, (widget[1], widget[2]), (widget[3], widget[4]), red_color, 2, cv2.LINE_AA)
    # draw label name
    for widget in img_widget_list:
        # class name(that is label name), draw name after bbox is drawn, so the name won't be covered by bbox.
        red_color = (0, 0, 255)  # BGR
        fontFace = cv2.FONT_HERSHEY_COMPLEX
        labelSize = cv2.getTextSize(widget[0], fontFace, 0.5, 1)
        _x2 = widget[1] + labelSize[0][0]  # topright x of text
        _y2 = widget[2] - labelSize[0][1]  # topright y of text
        cv2.rectangle(org_image, (widget[1], widget[2]), (_x2, _y2), red_color, cv2.FILLED)  # fill text background
        cv2.putText(org_image, widget[0], (widget[1], widget[2]), fontFace, 0.5, (0, 0, 0), 1)
    show_name = image_jpg_path.replace('\\', '/').split('/')[-1][:-4]
    if show:
        cv2.imshow(show_name, org_image)
        cv2.waitKey(0)
    if write_image:
        cv2.imwrite(show_name+".png", org_image)
    comp_json, text_json = {"img_shape": [], "compos": []}, {"img_shape": [], "texts": []}
    comp_id, text_id = 1, 0  # Component 0 is background.
    text_json["img_shape"] = comp_json["img_shape"] = [org_image.shape[0], org_image.shape[1], org_image.shape[2]]
    for widget in img_widget_list:
        top, left, bottom, right = widget[2], widget[1], widget[4], widget[3]
        if widget[0] == "Text":
            text_comp = {
                "id": text_id,
                "row_max": bottom,
                "row_min": top,
                "height": bottom-top,
                "content": "fake texts",
                "column_max": right,
                "column_min": left,
                "width": right-left
            }
            text_json["texts"].append(text_comp)
            text_id += 1
        else:
            class_name = "Compo"
            non_text_comp = {
                "id": comp_id,
                "row_max": bottom,
                "column_min": left,
                "class": class_name,
                "height": bottom-top,
                "row_min": top,
                "column_max": right,
                "width": right-left
            }
            comp_json["compos"].append(non_text_comp)
            comp_id += 1
    # if write_image:  # for checking result is the same as main.py result.
        # ocr_root = 'data/output/ocr'
        # text_path = pjoin(ocr_root, show_name+".json")
        # save_text_json(text_path, text_json)
        # nonText_path = pjoin('data/output/ip', show_name+'.json')
        # save_noText_json(nonText_path, comp_json)
    return comp_json, text_json


def save_noText_json(file_path, nonTexts):
    f_out = open(file_path, 'w')
    json.dump(nonTexts, f_out, indent=4)
    f_out.close()


def save_text_json(file_path, texts):
    f_out = open(file_path, 'w')
    json.dump(texts, f_out, indent=4)
    f_out.close()


def my_merge(img_path, compo_json, text_json, merge_root=None, is_paragraph=False, is_remove_bar=True, show=False, wait_key=0):
    # load text and non-text compo
    ele_id = 0
    compos = []
    for compo in compo_json['compos']:
        element = Element(ele_id, (compo['column_min'], compo['row_min'], compo['column_max'], compo['row_max']),
                          compo['class'])
        compos.append(element)
        ele_id += 1
    texts = []
    for text in text_json['texts']:
        element = Element(ele_id, (text['column_min'], text['row_min'], text['column_max'], text['row_max']), 'Text',
                          text_content=text['content'])
        texts.append(element)
        ele_id += 1
    if compo_json['img_shape'] != text_json['img_shape']:
        resize_ratio = compo_json['img_shape'][0] / text_json['img_shape'][0]
        for text in texts:
            text.resize(resize_ratio)

    # check the original detected elements
    img = cv2.imread(img_path)
    img_resize = cv2.resize(img, (compo_json['img_shape'][1], compo_json['img_shape'][0]))
    merge.show_elements(img_resize, texts + compos, show=show, win_name='all elements before merging', wait_key=wait_key)

    # refine elements
    texts = merge.refine_texts(texts, compo_json['img_shape'])
    elements = merge.refine_elements(compos, texts)
    if is_remove_bar:
        elements = merge.remove_top_bar(elements, img_height=compo_json['img_shape'][0])
        elements = merge.remove_bottom_bar(elements, img_height=compo_json['img_shape'][0])
    if is_paragraph:
        elements = merge.merge_text_line_to_paragraph(elements)
    merge.reassign_ids(elements)
    merge.check_containment(elements)  # children or parent or none.
    board = merge.show_elements(img_resize, elements, show=show, win_name='elements after merging', wait_key=wait_key)

    # save all merged elements, clips and blank background
    name = img_path.replace('\\', '/').split('/')[-1][:-4]
    components = merge.save_elements(pjoin(merge_root, name + '.json'), elements, img_resize.shape, False)
    # cv2.imwrite(pjoin(merge_root, name + '.jpg'), board)  # write uied file.
    # print('[Merge Completed] Input: %s Output: %s' % (img_path, pjoin(merge_root, name + '.jpg')))
    return board, components


def detect_listview_gridview_result():
    """detect listview gridview result"""
    in_jpg_folder = "E:\\Container Dataset\\test\\ground_truth\\None\\"
    in_xml_folder = "E:\\VINS Dataset\\All Dataset\\Android\\Annotations\\"
    all_jpg_path = glob.iglob(in_jpg_folder + "*.jpg")
    outputResultDir = "E:\\Container Dataset\\test_result"
    file_amount = 0
    # result_dict = {"listView": 0, "gridView": 0, "None": 0}
    result_dict = {"Group": 0, "None": 0}
    for jpg_file_path in all_jpg_path:
        file_amount += 1
        file_id = jpg_file_path.split("\\")[-1].split(".")[0]
        xml_file_path = in_xml_folder + "\\" + file_id + ".xml"
        gui = GUI(img_file=jpg_file_path)
        cate_list = ["Image", "EditText", "Icon", "TextButton", "CheckBox", "Switch", "Spinner",
                     "Text"]  # vins no radiobutton.
        non_text, text = parse_vins2dict(jpg_file_path, xml_file_path, cate_list, False, False)
        gui.detection_result_img['merge'], gui.compos_json = my_merge(jpg_file_path, non_text, text, "data/output/uied",
                                                                      is_remove_bar=False, is_paragraph=True,
                                                                      show=False)
        gui.img_reshape = gui.compos_json['img_shape']
        gui.recognize_layout(is_save=False)
        if len(gui.lists) >= 1:
            result_dict["Group"] += 1
            print("find group - end")
        else:
            result_dict["None"] += 1
            print("not find - end")
    print("check", file_amount, "files")
    print(result_dict)


if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)  # ignore pandas appends future warning(too much0
    # detect_listview_gridview_result()  # group evaluation
    # input_path = 'data/input/2.jpg'
    output_root = 'data/output'
    jpg_file_path = r"E:\VINS Dataset\All Dataset\Android\JPEGImages\Android_84.jpg"  # list view
    xml_file_path = r"E:\VINS Dataset\All Dataset\Android\Annotations\Android_84.xml"
    # jpg_file_path = r"E:\VINS Dataset\All Dataset\Android\JPEGImages\Android_1.jpg"  # grid view
    # xml_file_path = r"E:\VINS Dataset\All Dataset\Android\Annotations\Android_1.xml"
    # jpg_file_path = r"E:\VINS Dataset\All Dataset\Android\JPEGImages\Android_2.jpg"  # none
    # xml_file_path = r"E:\VINS Dataset\All Dataset\Android\Annotations\Android_2.xml"
    # jpg_file_path = r"E:\VINS Dataset\All Dataset\iphone\JPEGImages\IMG_2907.jpg"  # list view, hard
    # xml_file_path = r"E:\VINS Dataset\All Dataset\iphone\Annotations\IMG_2907.xml"
    gui = GUI(img_file=jpg_file_path, output_dir=output_root)
    cate_list = ["Image", "EditText", "Icon", "TextButton", "CheckBox", "Switch", "Spinner", "Text"]  # vins no radiobutton.
    non_text, text = parse_vins2dict(jpg_file_path, xml_file_path, cate_list, False, False)
    gui.detection_result_img['merge'], gui.compos_json = my_merge(jpg_file_path, non_text, text, "data/output/uied",  is_remove_bar=False, is_paragraph=True, show=False)
    gui.img_reshape = gui.compos_json['img_shape']
    gui.img_resized = cv2.resize(gui.img, (gui.img_reshape[1], gui.img_reshape[0]))
    gui.recognize_layout(is_save=False)
    if len(gui.lists) >= 1:
        print("find group - end")
    else:
        print("not find - end")
    # gui.visualize_layout_recognition()
