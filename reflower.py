import argparse
import os
from pathlib import Path
import pytesseract
from pytesseract import Output
import cv2
import hashlib
from pdf2image import convert_from_path
import re
from time import time
import ipdb  # TODO
import numpy as np


def md5_string(string):
    return hashlib.md5(string.encode('utf-8')).hexdigest()


def md5_file(filepath):
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def is_covered(child, parent):
    bool1 = child['left'] >= parent['left']
    bool2 = child['top'] >= parent['top']
    bool3 = child['left'] + child['width'] <= parent['left'] + parent['width']
    bool4 = child['top'] + child['height'] <= parent['top'] + parent['height']
    return bool1 and bool2 and bool3 and bool4


def is_intersected(location1, location2):
    # TODO wrong?
    bool1 = location1['left'] <= location2['left'] + location2['width']
    bool2 = location1['top'] >= location2['top'] + location2['height']
    bool3 = location1['left'] + location1['width'] >= location2['left']
    bool4 = location1['top'] + location1['height'] <= location2['top']
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
                        level2key[l]: data[level2key[l]][index]
                        for l in range(1, level + 1)
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
parser.add_argument('--source',
                    type=str,
                    default='./test/arxiv/2109.05236.pdf')
parser.add_argument('--target', type=str, default='./output.pdf')
parser.add_argument('--processing_dpi', type=int, default=400)
parser.add_argument('--export_dpi', type=int, default=300)
parser.add_argument('--root_temp_dir', type=str, default='./temp')
args = parser.parse_args()

assert args.processing_dpi >= args.export_dpi, 'processing_dpi should be larger than export_dpi'
assert os.path.isfile(args.source), 'Source file not found'

hash_value = md5_string(
    md5_string(args.source) + md5_file(args.source) +
    md5_string(str(args.processing_dpi)))[:16]
temp_dir = os.path.join(args.root_temp_dir, hash_value)
print(f'Temp directory: {temp_dir}')
if not os.path.exists(temp_dir):
    Path(temp_dir).mkdir(parents=True)
    pages = convert_from_path(args.source, dpi=args.processing_dpi)
    for index, page in enumerate(pages):
        image_path = os.path.join(temp_dir, f"page-{index:04d}-stage-1.png")
        page.save(image_path)

page_files = sorted([
    filename for filename in os.listdir(temp_dir)
    if re.search("^page-\d+-stage-1\.png$", filename)
])

document_data = []
for index, filename in enumerate(page_files):
    image_path = os.path.join(temp_dir, filename)
    image = cv2.imread(image_path)
    data = pytesseract.image_to_data(image, output_type=Output.DICT)
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
        block_area = block_location['width'] * block_location['height']
        block_word_area = 0
        for par in block['data']:
            for line in par['data']:
                for word in line['data']:
                    word_location = word['location']
                    if normal_word_height_limit[0] < word_location[
                            'height'] < normal_word_height_limit[1]:
                        cv2.rectangle(
                            image,
                            (word_location['left'], word_location['top']),
                            (word_location['left'] + word_location['width'],
                             word_location['top'] + word_location['height']),
                            (0, 0, 0), 2)
                        block_word_area += word_location[
                            'width'] * word_location['height']

        cv2.putText(image,
                    f'B:{block_num},D:{(block_word_area/block_area):.2f}',
                    (block_location['left'] - 300, block_location['top'] + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 0, 0), 2, cv2.LINE_AA)

        def is_text_block():
            return (block_word_area / block_area) > 0.4

        overlay = image.copy()
        cv2.rectangle(
            overlay, (block_location['left'] - 20, block_location['top'] - 20),
            (block_location['left'] + block_location['width'] + 20,
             block_location['top'] + block_location['height'] + 20),
            (0, 255, 0) if is_text_block() else (0, 0, 255), -1)
        alpha = 0.1
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        if is_text_block():
            words = {}
            for par in block['data']:
                for line in par['data']:
                    for word in line['data']:
                        words[par['num'], line['num'],
                              word['num']] = (word['location'], word['data'])
            words = dict(sorted(words.items())).values()
            words = [{'location': x[0], 'text': x[1]} for x in words]
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

    cv2.imwrite(os.path.join(temp_dir, f"page-{index:04d}-stage-2.png"), image)
    page_data = sorted([*text_blocks, *non_text_blocks],
                       key=lambda x: x['num'])
    for block in page_data:
        block['page_index'] = index
        del block['num']
    document_data.extend(page_data)
    print(f'Finish stage 2 for {index+1} page')

for block in document_data:
    if block['is_text_block']:
        pass
    else:
        print(block)
