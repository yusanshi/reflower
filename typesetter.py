import numpy as np
import cv2


class Typesetter:
    def __init__(self, page_width, page_height, text_height, source_page_data):
        self.page_width = page_width
        self.page_height = page_height
        self.text_height = text_height
        self.line_height = int(1.3 * text_height)
        self.word_space = int(0.4 * text_height)
        self.dash_space = int(0.3 * text_height)
        self.last_word_height = text_height
        self.source_page_data = source_page_data
        self.pages = []
        self.current_page = None
        self.current_curser = None
        self.available_area = [
            (int(0.5 * text_height), int(0.5 * text_height)),
            (page_width - int(0.5 * text_height) - int(0.5 * self.word_space),
             page_height - int(0.5 * text_height))
        ]
        self.take_new_page()

    def add_block(self, block):
        if block['is_text_block']:
            self.add_text_block(block)
        else:
            self.add_non_text_block(block)

    def add_text_block(self, block):
        for word in block['data']:
            new_size = (word['location']['width'], word['location']['height'])
            while True:
                if self.is_fittable(new_size):
                    break
                self.take_new_line(new_size[1])
                if self.is_fittable(new_size):
                    break
                self.take_new_page()
                if self.is_fittable(new_size):
                    break

                raise ValueError

            self.current_page[
                self.current_curser[1]:self.current_curser[1] + new_size[1],
                self.current_curser[0]:self.current_curser[0] +
                new_size[0]] = self.source_page_data[block['page_index']][
                    word['location']['top']:word['location']['top'] +
                    new_size[1],
                    word['location']['left']:word['location']['left'] +
                    new_size[0]]

            if word['text'].endswith('-'):
                word_space = -self.dash_space
            else:
                word_space = self.word_space
            self.current_curser[0] += new_size[0] + word_space
            self.last_word_height = new_size[1]

        self.take_new_line(1.5 * self.text_height)

    def add_non_text_block(self, block):
        if self.current_curser[0] != self.available_area[0][0]:
            self.take_new_line()
        new_size = (block['location']['width'], block['location']['height'])
        source = self.source_page_data[block['page_index']][
            block['location']['top']:block['location']['top'] + new_size[1],
            block['location']['left']:block['location']['left'] + new_size[0]]
        resized = False
        while True:
            if self.is_fittable(new_size):
                break

            resized = True
            available_width = self.available_area[1][0] - self.available_area[
                0][0]
            available_height = self.available_area[1][1] - self.available_area[
                0][1]
            resize_factor_x = new_size[0] / available_width
            resize_factor_y = new_size[1] / available_height
            if resize_factor_x >= resize_factor_y:
                new_size = (available_width,
                            int(new_size[1] / resize_factor_x))
            else:
                new_size = (int(new_size[0] / resize_factor_y),
                            available_height)

            if self.is_fittable(new_size):
                break
            self.take_new_page()
            if self.is_fittable(new_size):
                break

            raise ValueError

        if resized:
            source = cv2.resize(source, new_size)
        self.current_page[self.current_curser[1]:self.current_curser[1] +
                          new_size[1],
                          self.current_curser[0]:self.current_curser[0] +
                          new_size[0]] = source
        self.current_curser[1] += new_size[1]
        self.take_new_line(0.5 * self.text_height)

    def is_fittable(self, new_size):
        return self.current_curser[0] + new_size[0] <= self.available_area[1][
            0] and self.current_curser[1] + new_size[1] <= self.available_area[
                1][1]

    def take_new_page(self):
        if self.current_page is not None:
            self.pages.append(self.current_page)
        self.current_page = 255 * np.ones(
            (self.page_height, self.page_width, 3), np.uint8)
        self.current_curser = list(self.available_area[0])

    def take_new_line(self, current_word_height=None):
        if current_word_height is None:
            coefficient = self.last_word_height / self.text_height
        else:
            coefficient = np.mean((current_word_height,
                                   self.last_word_height)) / self.text_height
        line_height = int(self.line_height * coefficient)
        if self.current_curser[1] + line_height > self.available_area[1][1]:
            self.take_new_page()
        else:
            self.current_curser[0] = self.available_area[0][0]
            self.current_curser[1] += line_height

    def export_pages(self):
        if self.current_page is not None:
            return self.pages + [self.current_page]
        return self.pages
