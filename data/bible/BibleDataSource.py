import os
import random
from typing import TextIO, Tuple

from data.utils.DataSourceInterface import Translation, DataSource
from definitions import ROOT_DIR

BIBLE_LINES_PER_PARAGRAPH = 6
TRANSLATIONS_PATH = os.path.join(ROOT_DIR, "data/bible/translations")


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


def initialize_paragraph_mapping() -> Tuple[dict, dict]:
    """
    Function analyzes all the translations of the bible we've collected and creates a mapping of paragraph
    id to a specific section. Function aims to create a mapping providing the same section for each translation.

    :return: Tuple of dicts - one mapping paragraph ids to book and chapter pair, other one is the inverse of this mapping
    :rtype: tuple[dict, dict]
    """
    mapping = {}
    counter = 0

    for translation_dict in os.listdir(TRANSLATIONS_PATH):
        for file in os.listdir(os.path.join(TRANSLATIONS_PATH, translation_dict)):

            if file.endswith("000_read.txt") or not file.startswith("eng"):
                continue

            with open(os.path.join(TRANSLATIONS_PATH, translation_dict, file), 'r',
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
    """
    Object allowing for interaction with a translation of the Bible.
    Translation is defined by a path given during initialization.

    :ivar lines: Collection of all the lines of the translation.
    :type lines: list[str]
    :ivar no_lines: Number of lines in this translation
    :ivar no_paragraphs: Index of the maximum paragraph. No paragraph has higher id, but some with the lower ids might be missing from some translations.
    """
    paragraph_mapping, inverse_paragraph_mapping = initialize_paragraph_mapping()

    def __init__(self, path):
        """
        :param path: Path to the directory containing the translation.
        :type path: str
        """
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
        """
        Get a single line from the translation.

        :param line_id: The line of the translation which shall be returned
        :type line_id: int
        :return: The text of the line matching the line_id
        :rtype: str
        :raise IndexError: When line_id requested is not in the collection
        """
        if 0 <= line_id < self.no_lines:
            return self.lines[line_id]
        else:
            raise IndexError

    def get_paragraph(self, paragraph_id: int) -> str:
        """
        Get a greater section of a text, big enough to recognize the style of the translation author.
        It should be used to get the data for the neural net we are training.

        :param paragraph_id: identifier of the paragraph which shall be returned.
        :type paragraph_id: int
        :return: The text of a chosen paragraph. We aim to make them be around 4 lines.
        :rtype: str
        :raise IndexError: when paragraph_id requested is not in the collection
        """
        if paragraph_id < 0 or paragraph_id > max(BibleTranslation.paragraph_mapping.keys()):
            raise IndexError('Paragraph of of range.')

        if paragraph_id not in self.lines_of_paragraph.keys():
            raise IndexError("%d - This translation doesn't have this paragraph." % paragraph_id)

        start, finish = self.lines_of_paragraph[paragraph_id]
        return "".join(self.lines[start:finish])


class BibleDataSource(DataSource):
    """
    Handler for the Bible translations in the database.
    Allows for easy fetching of the translations with the get_translation method.

    :ivar no_translations: Number of available translations. Translation max index of the translation is one less than this.
    """

    def __init__(self):
        super().__init__()
        self._all_translations_list = os.listdir(TRANSLATIONS_PATH)
        self.no_translations = len(self._all_translations_list)

    def get_translation(self, translation_id: int) -> BibleTranslation:
        """
        Allows you to pick a Bible translation from the translation database.

        :param translation_id: Identifier of the Bible translation.
        :return: BibleTranslation object for translation identified by translation_id.
        :raise IndexError: translation_id doesn't match any translation in the database.
        """
        if 0 <= translation_id < self.no_translations:
            chosen_translation_path = self._find_translation(translation_id)
            return BibleTranslation(chosen_translation_path)
        else:
            raise IndexError

    def _find_translation(self, translation_id: int):
        for translation_name in self._all_translations_list:
            if translation_id == int(translation_name.split("-")[0]):
                return os.path.join(TRANSLATIONS_PATH, translation_name)


if __name__ == "__main__":
    quran_handle = BibleDataSource()
    for translation_index in random.sample(range(quran_handle.no_translations), 4):
        translation = quran_handle.get_translation(translation_index)
        print(translation.get_line(random.choice(range(translation.no_lines))))
        print(translation.get_paragraph(random.choice(list(translation.lines_of_paragraph.keys()))))
