import argparse
import os
from pathlib import Path
import pytesseract
from pytesseract import Output
import cv2
import hashlib
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
from distutils.util import strtobool
from itertools import chain
from typesetter import Typesetter
import layoutparser as lp
import functools
import operator
import itertools
import ipdb

import warnings
warnings.filterwarnings("ignore")


def md5_string(string):
    return hashlib.md5(string.encode('utf-8')).hexdigest()


def is_postive_area(area):
    return area.width > 0 and area.height > 0


def map_location(location, from_dpi, to_dpi):
    return {k: int(v * to_dpi / from_dpi) for k, v in location.items()}


def convert_tesseract_data(data):
    length = len(data['level'])
    level2key = {
        i + 1: key
        for i, key in enumerate(
            ['page_num', 'block_num', 'par_num', 'line_num', 'word_num'])
    }

    def get_data(candidate_indexs,
                 level=None,
                 page_num=None,
                 block_num=None,
                 par_num=None,
                 line_num=None,
                 word_num=None):
        result = []
        found_indexs = []
        for i in candidate_indexs:
            if level is not None and data['level'][i] != level:
                continue
            if page_num is not None and data['page_num'][i] != page_num:
                continue
            if block_num is not None and data['block_num'][i] != block_num:
                continue
            if par_num is not None and data['par_num'][i] != par_num:
                continue
            if line_num is not None and data['line_num'][i] != line_num:
                continue
            if word_num is not None and data['word_num'][i] != word_num:
                continue
            found_indexs.append(i)
        for index in found_indexs:
            if level < 5:
                result_data = get_data(
                    list(set(candidate_indexs) - set(found_indexs)),
                    level=level + 1,
                    **{
                        level2key[x]: data[level2key[x]][index]
                        for x in range(1, level + 1)
                    })
            else:
                result_data = data['text'][index]

            result.append({
                'level': level,
                'num': data[level2key[level]][index],
                'location': {
                    'left': data['left'][index],
                    'top': data['top'][index],
                    'width': data['width'][index],
                    'height': data['height'][index],
                },
                'data': result_data
            })
        return result

    result = get_data(level=1, candidate_indexs=range(length))
    return result


LEFT_INTERVAL_COEFFICIENT = 1.1
EXPANDING_STEP = 0.05  # inch

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default='./test/example.pdf')
parser.add_argument('--target', type=str, default='./output.pdf')
parser.add_argument('--detection_dpi', type=int, default=144)
parser.add_argument('--export_dpi', type=int, default=300)
parser.add_argument('--target_paper', type=str, default='pw3')
parser.add_argument('--detector',
                    type=str,
                    nargs='+',
                    default=['general', 'table', 'formula'],
                    choices=['general', 'table', 'formula'])
parser.add_argument('--threshold',
                    type=str,
                    nargs='+',
                    default=['general:0.0', 'table:0.8', 'formula:0.8'])
parser.add_argument('--debug',
                    type=lambda x: bool(strtobool(x)),
                    default=False)
parser.add_argument('--root_temp_dir', type=str, default='./temp')
args = parser.parse_args()

threshold = {k.split(':')[0]: float(k.split(':')[1]) for k in args.threshold}
for k, v in threshold.items():
    assert k in args.detector, f'{k} is not in {args.detector}'
    assert 0 <= v <= 1, f'{v} is not in [0, 1]'

assert os.path.isfile(args.source), 'Source file not found'
source_page_data_pillow = convert_from_path(args.source,
                                            dpi=args.detection_dpi)

detector = {}
if 'general' in args.detector:
    detector['general'] = lp.PaddleDetectionLayoutModel(
        "lp://paddledetection/PubLayNet/ppyolov2_r50vd_dcn_365e",
        label_map={
            0: "Text",
            1: "Title",
            2: "List",
            3: "Table",
            4: "Figure"
        })
if 'table' in args.detector:
    detector['table'] = lp.PaddleDetectionLayoutModel(
        'lp://paddledetection/TableBank/ppyolov2_r50vd_dcn_365e',
        label_map={0: "Table"})
if 'formula' in args.detector:
    detector['formula'] = lp.Detectron2LayoutModel(
        'lp://MFD/faster_rcnn_R_50_FPN_3x/config', label_map={1: "Equation"})

if args.debug:
    hash_value = md5_string(str(args))[:16]
    temp_dir = os.path.join(args.root_temp_dir, hash_value)
    print(f'Temp directory: {temp_dir}')
    Path(temp_dir).mkdir(parents=True, exist_ok=True)

text_block = {'Text', 'Title', 'List'}
non_text_block = {'Table', 'Figure', 'Equation'}

document_data = []
for page_index, image in enumerate(source_page_data_pillow):
    image_temp = image.copy()

    layout = {}
    bg_color = max(image_temp.getcolors(image_temp.width *
                                        image_temp.height))[1]
    for x in set(args.detector) & set(['table', 'formula']):
        layout[x] = lp.Layout([
            e for e in detector[x].detect(image_temp)
            if e.score >= threshold[x]
        ])
        for block in layout[x]:
            image_temp.paste(bg_color, tuple(map(int, block.coordinates)))

    for x in set(args.detector) & set(['general']):
        layout[x] = lp.Layout([
            e for e in detector[x].detect(image_temp)
            if e.score >= threshold[x]
        ])

    blocks = functools.reduce(operator.add, layout.values())

    # Combine overlapped blocks
    while True:
        for first, second in itertools.combinations(range(len(blocks)), 2):
            intersected = blocks[first].intersect(blocks[second])
            if is_postive_area(intersected) and (intersected.area / min(
                    blocks[first].area, blocks[second].area)) > 0.6:
                # inherit the more confident one's properties
                if blocks[first].score >= blocks[second].score:
                    unioned = blocks[first].union(blocks[second])
                else:
                    unioned = blocks[second].union(blocks[first])
                # the deleting order matters...
                del blocks[second]
                del blocks[first]
                blocks.append(unioned)
                break  # continue on `while True`
        else:
            break  # break the whole `while True`

    # Assign the orders
    left_interval = lp.Interval(0,
                                image.width / 2 * LEFT_INTERVAL_COEFFICIENT,
                                axis='x').put_on_canvas(image)
    left_blocks = blocks.filter_by(left_interval, center=True)
    left_blocks.sort(key=lambda b: b.coordinates[1], inplace=True)

    right_blocks = lp.Layout([b for b in blocks if b not in left_blocks])
    right_blocks.sort(key=lambda b: b.coordinates[1], inplace=True)

    blocks = lp.Layout(
        [b.set(id=idx) for idx, b in enumerate(left_blocks + right_blocks)])

    # Expanding the blocks
    expanding_step = EXPANDING_STEP * args.detection_dpi
    for block_index in range(len(blocks)):
        while True:
            for direction_index, direction in enumerate(
                ['left', 'top', 'right', 'bottom']):
                base_coordinates = blocks[block_index].coordinates
                padded = blocks[block_index].pad(**{direction: expanding_step})
                padded_coordinates = padded.coordinates

                # Get (blocks[block_index] XOR padded) coordinates
                # or "padded - blocks[block_index]" coordinates
                not_equal = [
                    (x, y)
                    for x, y in zip(base_coordinates, padded_coordinates)
                    if x != y
                ]
                assert len(not_equal) == 1
                border_coordinates = list(padded_coordinates)
                border_coordinates[direction_index] = not_equal[0][1]
                border_coordinates[(direction_index + 2) % 4] = not_equal[0][0]
                border_coordinates = tuple(border_coordinates)

                # if will overlap with other blocks, stop expanding
                if any([
                        is_postive_area(blocks[j].intersect(
                            lp.Rectangle(*border_coordinates)))
                        for j in range(len(blocks)) if j != block_index
                ]):
                    continue

                border = image.crop(border_coordinates)
                border_colors = border.getcolors(border.width * border.height)
                # if the expanding area is **blank**, stop expanding
                if len(border_colors) == 1 and border_colors[0][1] == bg_color:
                    continue

                blocks[block_index] = padded
                if args.debug:
                    print(
                        f'In page {page_index} padding block {block_index} on {direction}'
                    )

                break  # continue on `while True`

            else:
                break  # break the whole `while True`

    # TODO pdf text to outlines, then copy the vector into new pdf instead of bitmap

    # OCR
    for block in blocks:
        if block.type in text_block:
            data = pytesseract.image_to_data(image.crop(block.coordinates),
                                             output_type=Output.DICT)
            data = convert_tesseract_data(data)
            assert len(data) == 1
            tesseract_page = data[0]

            text = []
            for tesseract_block in tesseract_page['data']:
                for tesseract_par in tesseract_block['data']:
                    for tesseract_line in tesseract_par['data']:
                        top_line = min([
                            tesseract_word['location']['top']
                            for tesseract_word in tesseract_line['data']
                        ])
                        bottom_line = max([
                            tesseract_word['location']['top'] +
                            tesseract_word['location']['height']
                            for tesseract_word in tesseract_line['data']
                        ])
                        for tesseract_word in tesseract_line['data']:
                            if len(tesseract_word['data'].strip()) > 0:
                                text.append({
                                    'location': {
                                        'left':
                                        tesseract_word['location']['left'] +
                                        block.coordinates[0],
                                        'top':
                                        top_line + block.coordinates[1],
                                        'width':
                                        tesseract_word['location']['width'],
                                        'height':
                                        bottom_line - top_line
                                    },
                                    'text': tesseract_word['data']
                                })

            block.set(text=text, inplace=True)

    if args.debug:
        boxed_image = image.copy()
        for block in blocks:
            if block.type in text_block:
                boxed_image = lp.draw_box(
                    boxed_image,
                    lp.Layout([
                        lp.TextBlock(lp.Rectangle(
                            x['location']['left'], x['location']['top'],
                            x['location']['left'] + x['location']['width'],
                            x['location']['top'] + x['location']['height']),
                                     type='Word') for x in block.text
                    ]),
                    color_map={'Word': 'grey'},
                    box_width=1)
        boxed_image = lp.draw_box(
            boxed_image,
            lp.Layout(
                [b.set(id=f'{b.id}/{b.type}/{b.score:.2f}') for b in blocks]),
            show_element_id=True,
            id_font_size=15,
            color_map={
                **{k: 'green'
                   for k in text_block},
                **{k: 'red'
                   for k in non_text_block}
            },
            id_text_background_color='grey',
            id_text_color='white',
            box_width=2)

        boxed_image.save(
            os.path.join(temp_dir, f"page-{page_index:04d}-box.png"))

    for block in blocks:
        if block.type in text_block:
            document_data.append({
                'page_index':
                page_index,
                'is_text_block':
                True,
                'data': [{
                    'location':
                    map_location(x['location'], args.detection_dpi,
                                 args.export_dpi),
                    'text':
                    x['text']
                } for x in block.text]
            })
        else:
            document_data.append({
                'location':
                map_location(
                    {
                        'left': block.coordinates[0],
                        'top': block.coordinates[1],
                        'width': block.width,
                        'height': block.height
                    }, args.detection_dpi, args.export_dpi),
                'page_index':
                page_index,
                'is_text_block':
                False
            })

# for block in sorted_blocks:
#     block_num = block['num']
#     block_location = block['location']
#     if any([
#             is_covered(block_location, non_text_block['location'])
#             for non_text_block in non_text_blocks
#     ]):
#         continue

#     block_area = block_location['width'] * block_location['height']
#     block_word_area = 0
#     for par in block['data']:
#         for line in par['data']:
#             for word in line['data']:
#                 word_location = word['location']
#                 if normal_word_height_limit[0] < word_location[
#                         'height'] < normal_word_height_limit[1]:
#                     if args.debug:
#                         cv2.rectangle(
#                             image,
#                             (word_location['left'], word_location['top']),
#                             (word_location['left'] +
#                              word_location['width'], word_location['top'] +
#                              word_location['height']), (0, 0, 0), 2)
#                     block_word_area += word_location[
#                         'width'] * word_location['height']
#     if args.debug:
#         cv2.putText(
#             image, f'B:{block_num},D:{(block_word_area/block_area):.2f}',
#             (block_location['left'] - 300, block_location['top'] + 50),
#             cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 0, 0), 2, cv2.LINE_AA)

#     def is_text_block():
#         return (block_word_area / block_area) > 0.4

#     if args.debug:
#         overlay = image.copy()
#         cv2.rectangle(
#             overlay,
#             (block_location['left'] - 2, block_location['top'] - 2),
#             (block_location['left'] + block_location['width'] + 2,
#              block_location['top'] + block_location['height'] + 2),
#             (0, 255, 0) if is_text_block() else (0, 0, 255), -1)
#         alpha = 0.1
#         image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

#     if is_text_block():
#         words = {}
#         for par in block['data']:
#             for line in par['data']:
#                 top_line = min(
#                     [word['location']['top'] for word in line['data']])
#                 bottom_line = max([
#                     word['location']['top'] + word['location']['height']
#                     for word in line['data']
#                 ])
#                 for word in line['data']:
#                     word['location']['top'] = top_line
#                     word['location']['height'] = bottom_line - top_line
#                     words[par['num'], line['num'],
#                           word['num']] = (word['location'], word['data'])
#         words = dict(sorted(words.items())).values()
#         words = [{
#             'location': x[0],
#             'text': x[1]
#         } for x in words if len(x[1].strip()) > 0]
#         text_blocks.append({
#             **{k: block[k]
#                for k in ['num', 'location']},
#             'data': words,
#             'is_text_block': True,
#         })
#     else:
#         non_text_blocks.append({
#             **{k: block[k]
#                for k in ['num', 'location']},
#             'is_text_block': False,
#         })

# page_data = sorted([*text_blocks, *non_text_blocks],
#                    key=lambda x: x['num'])
# for block in page_data:
#     block['page_index'] = index
#     del block['num']
# document_data.extend(page_data)

paper2size_mm = {
    'a4': (210, 297),
    'a5': (148, 210),
    'pw3': (90.8, 122.6),
}
if args.target_paper in paper2size_mm:
    paper_size_mm = paper2size_mm[args.target_paper]
else:
    paper_size_mm = list(map(float, args.target_paper.split('x')))
# (width, height)
paper_size_pt = tuple(
    map(lambda x: int(x / 25.4 * args.export_dpi), paper_size_mm))

text_height = int(
    np.median(
        np.array(
            list(
                chain.from_iterable(
                    [[x['location']['height'] for x in block['data']]
                     for block in document_data if block['is_text_block']])))))

source_page_data_opencv = [
    cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
    for page in convert_from_path(args.source, dpi=args.export_dpi)
]
typesetter = Typesetter(*paper_size_pt, text_height, source_page_data_opencv)

for block in document_data:
    typesetter.add_block(block)

exported_pages_opencv = typesetter.export_pages()

exported_pages_pillow = [
    Image.fromarray(cv2.cvtColor(page, cv2.COLOR_BGR2RGB))
    for page in exported_pages_opencv
]
exported_pages_pillow[0].save(args.target,
                              "PDF",
                              resolution=args.export_dpi,
                              save_all=True,
                              append_images=exported_pages_pillow[1:])
