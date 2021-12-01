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


def md5_string(string):
    return hashlib.md5(string.encode('utf-8')).hexdigest()


def is_covered(child, parent):
    bool1 = child['left'] >= parent['left']
    bool2 = child['top'] >= parent['top']
    bool3 = child['left'] + child['width'] <= parent['left'] + parent['width']
    bool4 = child['top'] + child['height'] <= parent['top'] + parent['height']
    return bool1 and bool2 and bool3 and bool4


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


parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default='./test/example.pdf')
parser.add_argument('--target', type=str, default='./output.pdf')
parser.add_argument('--dpi', type=int, default=300)
parser.add_argument('--target_paper', type=str, default='pw3')
parser.add_argument('--debug',
                    type=lambda x: bool(strtobool(x)),
                    default=False)
parser.add_argument('--root_temp_dir', type=str, default='./temp')
args = parser.parse_args()

assert os.path.isfile(args.source), 'Source file not found'
source_page_data_pillow = convert_from_path(args.source, dpi=args.dpi)
source_page_data_opencv = [
    cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
    for page in source_page_data_pillow
]
if args.debug:
    hash_value = md5_string(str(args))[:16]
    temp_dir = os.path.join(args.root_temp_dir, hash_value)
    print(f'Temp directory: {temp_dir}')
    stage_dir = os.path.join(temp_dir, 'stage-1')
    Path(stage_dir).mkdir(parents=True, exist_ok=True)
    for index, page in enumerate(source_page_data_opencv):
        image_path = os.path.join(stage_dir, f"page-{index:04d}.png")
        cv2.imwrite(image_path, page)

block_proportion_threshold = (1 / 50, 8)
document_data = []
for index in range(len(source_page_data_opencv)):
    if args.debug:
        image = source_page_data_opencv[index].copy()
    data = pytesseract.image_to_data(source_page_data_pillow[index],
                                     output_type=Output.DICT)
    word_heights = np.array([
        data['height'][i] for i in range(len(data['level']))
        if data['level'][i] == 5
    ])
    normal_word_height = np.median(word_heights)
    normal_word_height_limit = (normal_word_height / 3, normal_word_height * 3)

    data = convert_tesseract_data(data)
    assert len(data) == 1
    page = data[0]
    page_location = page['location']

    text_blocks = []
    non_text_blocks = []
    sorted_blocks = sorted(page['data'],
                           key=lambda block: block['location']['width'] *
                           block['location']['height'],
                           reverse=True)
    for block in sorted_blocks:
        block_num = block['num']
        block_location = block['location']
        if any([
                is_covered(block_location, non_text_block['location'])
                for non_text_block in non_text_blocks
        ]):
            continue

        block_proportion = block_location['height'] / block_location['width']
        if not (block_proportion_threshold[0] < block_proportion <
                block_proportion_threshold[1]):
            if args.debug:
                print(f'Ignore block with proportion {block_proportion}')
                overlay = image.copy()
                cv2.rectangle(
                    overlay,
                    (block_location['left'] - 2, block_location['top'] - 2),
                    (block_location['left'] + block_location['width'] + 2,
                     block_location['top'] + block_location['height'] + 2),
                    (255, 0, 0), -1)
                alpha = 0.2
                image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

            continue

        block_area = block_location['width'] * block_location['height']
        block_word_area = 0
        for par in block['data']:
            for line in par['data']:
                for word in line['data']:
                    word_location = word['location']
                    if normal_word_height_limit[0] < word_location[
                            'height'] < normal_word_height_limit[1]:
                        if args.debug:
                            cv2.rectangle(
                                image,
                                (word_location['left'], word_location['top']),
                                (word_location['left'] +
                                 word_location['width'], word_location['top'] +
                                 word_location['height']), (0, 0, 0), 2)
                        block_word_area += word_location[
                            'width'] * word_location['height']
        if args.debug:
            cv2.putText(
                image, f'B:{block_num},D:{(block_word_area/block_area):.2f}',
                (block_location['left'] - 300, block_location['top'] + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 0, 0), 2, cv2.LINE_AA)

        def is_text_block():
            return (block_word_area / block_area) > 0.4

        if args.debug:
            overlay = image.copy()
            cv2.rectangle(
                overlay,
                (block_location['left'] - 2, block_location['top'] - 2),
                (block_location['left'] + block_location['width'] + 2,
                 block_location['top'] + block_location['height'] + 2),
                (0, 255, 0) if is_text_block() else (0, 0, 255), -1)
            alpha = 0.1
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        if is_text_block():
            words = {}
            for par in block['data']:
                for line in par['data']:
                    top_line = min(
                        [word['location']['top'] for word in line['data']])
                    bottom_line = max([
                        word['location']['top'] + word['location']['height']
                        for word in line['data']
                    ])
                    for word in line['data']:
                        word['location']['top'] = top_line
                        word['location']['height'] = bottom_line - top_line
                        words[par['num'], line['num'],
                              word['num']] = (word['location'], word['data'])
            words = dict(sorted(words.items())).values()
            words = [{
                'location': x[0],
                'text': x[1]
            } for x in words if len(x[1].strip()) > 0]
            text_blocks.append({
                **{k: block[k]
                   for k in ['num', 'location']},
                'data': words,
                'is_text_block': True,
            })
        else:
            non_text_blocks.append({
                **{k: block[k]
                   for k in ['num', 'location']},
                'is_text_block': False,
            })

    if args.debug:
        stage_dir = os.path.join(temp_dir, 'stage-2')
        Path(stage_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(os.path.join(stage_dir, f"page-{index:04d}.png"), image)

    page_data = sorted([*text_blocks, *non_text_blocks],
                       key=lambda x: x['num'])
    for block in page_data:
        block['page_index'] = index
        del block['num']
    document_data.extend(page_data)

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
paper_size_pt = tuple(map(lambda x: int(x / 25.4 * args.dpi), paper_size_mm))

text_height = int(
    np.median(
        np.array(
            list(
                chain.from_iterable(
                    [[x['location']['height'] for x in block['data']]
                     for block in document_data if block['is_text_block']])))))

typesetter = Typesetter(*paper_size_pt, text_height, source_page_data_opencv)

for block in document_data:
    typesetter.add_block(block)

exported_pages_opencv = typesetter.export_pages()
if args.debug:
    stage_dir = os.path.join(temp_dir, 'stage-3')
    Path(stage_dir).mkdir(parents=True, exist_ok=True)
    for index, page in enumerate(exported_pages_opencv):
        cv2.imwrite(os.path.join(stage_dir, f"page-{index:04d}.png"), page)

exported_pages_pillow = [
    Image.fromarray(cv2.cvtColor(page, cv2.COLOR_BGR2RGB))
    for page in exported_pages_opencv
]
exported_pages_pillow[0].save(args.target,
                              "PDF",
                              resolution=args.dpi,
                              save_all=True,
                              append_images=exported_pages_pillow[1:])
