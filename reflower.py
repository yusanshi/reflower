import argparse
import os
import pytesseract
import cv2
import hashlib
import numpy as np
import layoutparser as lp
import functools
import operator
import itertools
import warnings

from pathlib import Path
from pytesseract import Output
from pdf2image import convert_from_path
from PIL import Image
from distutils.util import strtobool
from itertools import chain
from typesetter import Typesetter

warnings.filterwarnings("ignore")


def md5_string(string):
    return hashlib.md5(string.encode('utf-8')).hexdigest()


def is_postive_area(area):
    '''
    Used to check whether two blocks are intersected
    
    Args:
        area: output of `intersect` operation in layout-parser library
    
    Note simply `return area.area > 0` will not work since `area.area` will be positive if both width and height are negative. Tricky :)
    '''
    return area.width > 0 and area.height > 0


def map_location(location, from_dpi, to_dpi):
    '''
    Map the location under one DPI to another DPI.

    Needed since we are using independent DPIs for detection and export.
    '''
    return {k: int(v * to_dpi / from_dpi) for k, v in location.items()}


def convert_tesseract_data(data):
    '''
    Convert raw output of `pytesseract.image_to_data(img, output_type=Output.DICT)`
    to a more intuitive hierarchical format for easier iteration.
    In this way, you can do something like:

    for page in data:
        for block in page['data']:
            for par in block['data']:
                for line in par['data']:
                    for word in line['data']:
                        pass
    '''
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
                # the `data` of the last level is the actual text
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


# For a two-column document, treat a block to be at left side
# if its center is in (0, LEFT_SIDE_COEFFICIENT * page_width)
LEFT_SIDE_COEFFICIENT = 0.55
# The exapnding size for blocks in inch
EXPANDING_STEP = 0.05
# The padding size for word boxes in inch
WORD_BORDER_EXPANDED = 0.01
# In detecting results, for block A and B,
# if `intersect(A, B).area / min(A.area, B.area)`
# is greater than this value, combine them
OVERLAPPING_COMBINATION_THRESHOLD = 0.6
# Some preset paper sizes in mm
PAPER2SIZE = {
    'a4': (210, 297),
    'a5': (148, 210),
    'pw3': (90.8, 122.6),
}
# Block types used with layout-parser.
# The blocks in `TEXT_BLOCK` will be OCRed in word level and reflowed, and those in `NON_TEXT_BLOCK` will simply croped and pasted.
TEXT_BLOCK = {'Text', 'Title', 'List'}
NON_TEXT_BLOCK = {'Table', 'Figure', 'Equation'}

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default='./test/example.pdf')
parser.add_argument('--target', type=str, default='./output.pdf')
parser.add_argument('--detection_dpi', type=int, default=200)
parser.add_argument('--export_dpi', type=int, default=300)
parser.add_argument(
    '--scale',
    type=float,
    default=1.0,
    help=
    'Scale the contents in **physical** size (TODO: not implemented yet, so currently the output contents will be the same physical size with the original)'
)
parser.add_argument(
    '--target_paper',
    type=str,
    default='pw3',
    help=
    f'Specify the target paper. Either provide a key in preset dict ({PAPER2SIZE}) or manually input the size in mm (for example, `--target_paper 90.8x122.6`).'
)
parser.add_argument('--detector',
                    type=str,
                    nargs='+',
                    default=['general', 'table', 'formula'],
                    choices=['general', 'table', 'formula'])
parser.add_argument(
    '--threshold',
    type=str,
    nargs='+',
    default=['general:0.0', 'table:0.8', 'formula:0.8'],
    help=
    'Specify the threshold for each detector. The format is `detector:threshold`. The detected boxes with score lower than the threshold will be ignored.'
)
parser.add_argument(
    '--debug',
    type=lambda x: bool(strtobool(x)),
    default=False,
    help=
    'Set to `True` to print helping information and save intermediate detected images.'
)
parser.add_argument('--root_temp_dir',
                    type=str,
                    default='./temp',
                    help='Location to save intermediate images')
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
    # create the temp directory for saving intermediate images
    hash_value = md5_string(str(args))[:16]
    temp_dir = os.path.join(args.root_temp_dir, hash_value)
    print(f'Temp directory: {temp_dir}')
    Path(temp_dir).mkdir(parents=True, exist_ok=True)

document_data = []
for page_index, image in enumerate(source_page_data_pillow):
    image_temp = image.copy()

    layout = {}
    bg_color = max(image_temp.getcolors(image_temp.width *
                                        image_temp.height))[1]

    # First use a dedicated detector (i.e., table, formula...),
    # then remove the detected area, then use a general purpose detector.
    # In my experiments, this genreates better results than detecting without
    # removing the detected table or formula area
    for x in set(args.detector) & set(['table', 'formula']):
        layout[x] = lp.Layout([
            e for e in detector[x].detect(image_temp)
            if e.score >= threshold[x]
        ])
        for block in layout[x]:
            # Remove them by filling the area with background color
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
                    blocks[first].area,
                    blocks[second].area)) > OVERLAPPING_COMBINATION_THRESHOLD:
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
                                image.width * LEFT_SIDE_COEFFICIENT,
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

                break  # continue on `while True`, note this will recheck expanding in **all four** directions and this is needed. If not recheck on all directions, the expanding results will be not ideal.

            else:
                break  # break the whole `while True`

    # OCR
    for block in blocks:
        if block.type in TEXT_BLOCK:
            data = pytesseract.image_to_data(image.crop(block.coordinates),
                                             lang='eng',
                                             output_type=Output.DICT)
            data = convert_tesseract_data(data)
            assert len(data) == 1
            tesseract_page = data[0]

            text = []
            for tesseract_block in tesseract_page['data']:
                for tesseract_par in tesseract_block['data']:
                    for tesseract_line in tesseract_par['data']:
                        # Later we will use line's top and height as top and height of words in the line,
                        # If we don't do this and directly use words' top and height,
                        # the resulting words in a line will be *ups and downs*
                        line_top = tesseract_line['location']['top']
                        line_height = tesseract_line['location']['height']
                        for tesseract_word in tesseract_line['data']:
                            if len(tesseract_word['data'].strip()) > 0:
                                word_border_expanded = WORD_BORDER_EXPANDED * args.detection_dpi
                                text.append({
                                    'location': {
                                        'left':
                                        tesseract_word['location']['left'] +
                                        block.coordinates[0] -
                                        word_border_expanded,
                                        'top':
                                        line_top + block.coordinates[1] -
                                        word_border_expanded,
                                        'width':
                                        tesseract_word['location']['width'] +
                                        2 * word_border_expanded,
                                        'height':
                                        line_height + 2 * word_border_expanded
                                    },
                                    'text': tesseract_word['data']
                                })

            block.set(text=text, inplace=True)

    if args.debug:
        boxed_image = image.copy()
        for block in blocks:
            if block.type in TEXT_BLOCK:
                # draw the word boxes
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
        # draw the block boxes
        boxed_image = lp.draw_box(
            boxed_image,
            lp.Layout(
                [b.set(id=f'{b.id}/{b.type}/{b.score:.2f}') for b in blocks]),
            show_element_id=True,
            id_font_size=20,
            color_map={
                **{k: 'green'
                   for k in TEXT_BLOCK},
                **{k: 'red'
                   for k in NON_TEXT_BLOCK}
            },
            id_text_background_color='grey',
            id_text_color='white',
            box_width=2)

        boxed_image.save(
            os.path.join(temp_dir, f"page-{page_index:04d}-box.png"))

    for block in blocks:
        if block.type in TEXT_BLOCK:
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

if args.target_paper in PAPER2SIZE:
    paper_size_mm = PAPER2SIZE[args.target_paper]
else:
    paper_size_mm = list(map(float, args.target_paper.split('x')))
# (width, height)
paper_size_pt = tuple(
    map(lambda x: int(x / 25.4 * args.export_dpi), paper_size_mm))

# Use midian text height as base text height
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
