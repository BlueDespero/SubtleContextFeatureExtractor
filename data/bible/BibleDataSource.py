import os
import pickle
import random
from typing import TextIO, Tuple

from data.utils.DataSourceBase import BaseDataSource, BaseMetadata, BaseTranslation
from definitions import ROOT_DIR

BIBLE_LINES_PER_PARAGRAPH = 6
TRANSLATIONS_PATH = os.path.join(ROOT_DIR, 'data', 'bible', 'translations')


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


def load_precomputed_files() -> Tuple[dict, set]:
    try:
        with open(os.path.join(ROOT_DIR, 'data', 'bible', 'preprocessed_data', 'files_already_mapped.pickle'),
                  'rb') as translations_already_mapped, open(
            os.path.join(ROOT_DIR, 'data', 'bible', 'preprocessed_data', 'precomputed_mapping.pickle'),
            'rb') as precomputed_mapping:
            mapping = pickle.load(precomputed_mapping)
            handled_files = pickle.load(translations_already_mapped)
            return mapping, handled_files
    except FileNotFoundError:
        return {}, set()


def save_computed_files(mapping: dict, handled_files: set):
    with open(os.path.join(ROOT_DIR, 'data', 'bible', 'preprocessed_data', 'files_already_mapped.pickle'),
              'wb') as translations_already_mapped, open(
        os.path.join(ROOT_DIR, 'data', 'bible', 'preprocessed_data', 'precomputed_mapping.pickle'),
        'wb') as precomputed_mapping:
        pickle.dump(handled_files, translations_already_mapped)
        pickle.dump(mapping, precomputed_mapping)


def initialize_paragraph_mapping() -> Tuple[dict, dict]:
    """
    Function analyzes all the translations of the bible we've collected and creates a mapping of paragraph
    id to a specific section. Function aims to create a mapping providing the same section for each translation.

    :return: Tuple of dicts - one mapping paragraph ids to book and chapter pair, other one is the inverse of this mapping
    :rtype: tuple[dict, dict]
    """
    counter = 0
    files_changed = False
    mapping, handled_files = load_precomputed_files()

    for translation_dict in os.listdir(TRANSLATIONS_PATH):
        for file in os.listdir(os.path.join(TRANSLATIONS_PATH, translation_dict)):
            file_under_consideration = os.path.join(TRANSLATIONS_PATH, translation_dict, file)

            if file.endswith("000_read.txt") or not file.startswith("eng"):
                continue

            if file_under_consideration in handled_files:
                continue

            with open(file_under_consideration, 'r',
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

            handled_files.add(file_under_consideration)
            files_changed = True

    if files_changed:
        save_computed_files(mapping, handled_files)

    return mapping, {v: k for k, v in mapping.items()}


class BibleTranslation(BaseTranslation):
    """
    Object allowing for interaction with a translation of the Bible.
    Translation is defined by a path given during initialization.

    :ivar lines: Collection of all the lines of the translation.
    :type lines: list[str]
    :ivar no_lines: Number of lines in this translation
    :ivar no_paragraphs: Index of the maximum paragraph. No paragraph has higher id, but some with the lower ids might be missing from some translations.
    """
    paragraph_mapping, inverse_paragraph_mapping = initialize_paragraph_mapping()

    def __init__(self, path, translation_id, embedding):
        """
        :param path: Path to the directory containing the translation.
        :type path: str
        """
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
        super().__init__(path, translation_id, embedding)


class BibleMetadata(BaseMetadata):

    def __init__(self, data_source):
        path_to_save_file = os.path.join(ROOT_DIR, 'data', 'bible', 'preprocessed_data', 'metadata.pickle')
        super().__init__(data_source, path_to_save_file)


class BibleDataSource(BaseDataSource):
    """
    Handler for the Bible translations in the database.
    Allows for easy fetching of the translations with the get_translation method.

    :ivar no_translations: Number of available translations. Translation max index of the translation is one less than this.
    """

    def __init__(self):
        super().__init__()
        self._all_translations_list = os.listdir(TRANSLATIONS_PATH)
        self.no_translations = len(self._all_translations_list)

    def get_translation(self, translation_id: int, embedding=None) -> BibleTranslation:
        """
        Allows you to pick a Bible translation from the translation database.

        :param translation_id: Identifier of the Bible translation.
        :return: BibleTranslation object for translation identified by translation_id.
        :raise IndexError: translation_id doesn't match any translation in the database.
        """
        if 0 <= translation_id < self.no_translations:
            chosen_translation_path = self._find_translation(translation_id)
            return BibleTranslation(chosen_translation_path, translation_id, embedding)
        else:
            raise IndexError

    def _find_translation(self, translation_id: int):
        for translation_name in self._all_translations_list:
            if translation_id == int(translation_name.split("-")[0]):
                return os.path.join(TRANSLATIONS_PATH, translation_name)

    def get_metadata(self) -> BibleMetadata:
        return BibleMetadata(data_source=self)


if __name__ == "__main__":
    quran_handle = BibleDataSource()
    for translation_index in random.sample(range(quran_handle.no_translations), 4):
        translation = quran_handle.get_translation(translation_index)
        print(translation.get_line(random.choice(range(translation.no_lines))))
        print(translation.get_paragraph(random.choice(list(translation.lines_of_paragraph.keys()))))
