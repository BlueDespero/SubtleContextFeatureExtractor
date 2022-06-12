import os
import random
from typing import TextIO, Tuple

from data.utils.DataSourceInterface import Translation, DataSource

BIBLE_LINES_PER_PARAGRAPH = 6


def get_book_and_chapter(data_stream: TextIO) -> Tuple[str, int]:
    """
    Get the book and chapter information from the first two lines of the file.

    :param data_stream: Data stream from which the data will be read.
    :return: A pair of book name and chapter number.
    :raise IOError: data_stream was read from before.
    """

    if not data_stream.tell() == 0:
        raise IOError("This data stream was read from. For this function a fresh stream is needed.")

    book, chapter = [next(data_stream) for _ in range(2)]
    book = book.split(" ")[-1]
    chapter = int(chapter.split(" ")[-1].split(".")[0])

    return book, chapter


def get_next_n_lines(data_stream: TextIO, n: int):
    """
    Return n lines (or less if not possible) from the data_stream.

    :param data_stream: Data stream from which lines will be read.
    :param n: Target amount of lines to be read from the data_stream. If reading n lines is not possible,
        fewer lines will be returned.
    :return: List of string lines taken from the data_stream. Up to n entries.
    """
    lines = []
    for _ in range(n):
        if new_line := next(data_stream, False):
            lines.append(new_line)
        else:
            break
    return lines


def initialize_paragraph_mapping():
    mapping = {}
    counter = 0

    for translation_dict in os.listdir("translations"):
        for file in os.listdir(os.path.join("translations", translation_dict)):

            if file.endswith("000_read.txt") or not file.startswith("eng"):
                continue

            with open(os.path.join("translations", translation_dict, file), 'r',
                      encoding="utf-8") as translation_chapter_text:

                book, chapter = get_book_and_chapter(translation_chapter_text)

                position = 0
                no_lines_read = BIBLE_LINES_PER_PARAGRAPH
                while no_lines_read == BIBLE_LINES_PER_PARAGRAPH:
                    no_lines_read = len(get_next_n_lines(translation_chapter_text, BIBLE_LINES_PER_PARAGRAPH))
                    paragraph_identifier_tuple = (book, chapter, position)

                    if paragraph_identifier_tuple not in mapping.values():
                        mapping[counter] = paragraph_identifier_tuple
                        counter += 1

                    position += no_lines_read

    return mapping, {v: k for k, v in mapping.items()}


class BibleTranslation(Translation):
    paragraph_mapping, inverse_paragraph_mapping = initialize_paragraph_mapping()

    def __init__(self, path):
        super().__init__(path)
        self.lines = []
        self.lines_of_paragraph = {}

        # For each book and chapter of this translation load lines and set up a matching showing where each
        # paragraph begins abd ends
        for file in os.listdir(path):
            if file.endswith("000_read.txt") or not file.startswith("eng"):
                continue

            with open(os.path.join(path, file), 'r', encoding="utf-8") as translation_chapter_text:

                book, chapter = get_book_and_chapter(translation_chapter_text)

                position = 0
                no_lines_read = BIBLE_LINES_PER_PARAGRAPH
                while no_lines_read == BIBLE_LINES_PER_PARAGRAPH:
                    lines_read = get_next_n_lines(translation_chapter_text, BIBLE_LINES_PER_PARAGRAPH)
                    no_lines_read = len(lines_read)

                    paragraph_identifier_tuple = (book, chapter, position)
                    id_of_tuple = BibleTranslation.inverse_paragraph_mapping[paragraph_identifier_tuple]

                    self.lines_of_paragraph[id_of_tuple] = (len(self.lines), len(self.lines) + no_lines_read)
                    self.lines += lines_read

                    position += no_lines_read

        self.no_lines = len(self.lines)
        self.no_paragraphs = max(self.lines_of_paragraph.keys())

    def get_line(self, line_id):
        if 0 <= line_id < self.no_lines:
            return self.lines[line_id]
        else:
            raise IndexError

    def get_paragraph(self, paragraph_id):
        if paragraph_id < 0 or paragraph_id > max(BibleTranslation.paragraph_mapping.keys()):
            raise IndexError('Paragraph of of range.')

        if paragraph_id not in self.lines_of_paragraph.keys():
            raise IndexError("%d - This translation doesn't have this paragraph." % paragraph_id)

        start, finish = self.lines_of_paragraph[paragraph_id]
        return "".join(self.lines[start:finish])


class BibleDataSource(DataSource):
    def __init__(self):
        super().__init__()
        self.all_translations_list = os.listdir('translations')
        self.no_translations = len(self.all_translations_list)

    def get_translation(self, translation_id: int) -> BibleTranslation:
        if 0 <= translation_id < self.no_translations:
            chosen_translation_path = self._find_translation(translation_id)
            return BibleTranslation(chosen_translation_path)
        else:
            raise IndexError

    def _find_translation(self, translation_id: int):
        for translation_name in self.all_translations_list:
            if translation_id == int(translation_name.split("-")[0]):
                return os.path.join("translations", translation_name)


if __name__ == "__main__":
    quran_handle = BibleDataSource()
    for translation_index in random.sample(range(quran_handle.no_translations), 4):
        translation = quran_handle.get_translation(translation_index)
        print(translation.get_line(random.choice(range(translation.no_lines))))
        print(translation.get_paragraph(random.choice(list(translation.lines_of_paragraph.keys()))))
