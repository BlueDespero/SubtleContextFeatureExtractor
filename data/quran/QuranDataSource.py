import ast
import os
import random

from data.utils.DataSourceInterface import Translation, DataSource, Metadata
from definitions import ROOT_DIR

QURAN_LINES_PER_PARAGRAPH = 6
TRANSLATIONS_PATH = os.path.join(ROOT_DIR, "data", "quran", "translations")


class QuranTranslation(Translation):

    def __init__(self, path, translation_id):
        super().__init__(path, translation_id)
        self.lines_of_paragraph = {}

        with open(path, 'r', encoding="utf-8") as translation_text:
            all_lines = translation_text.readlines()
            metadata_separator_line = all_lines.index("{\n")
            self.lines = all_lines[:metadata_separator_line]
            self.metadata = ast.literal_eval("".join(all_lines[metadata_separator_line:]))

        self.lines_of_paragraph = {i: (start, start + QURAN_LINES_PER_PARAGRAPH) for i, start in
                                   enumerate(range(0, len(self.lines) - QURAN_LINES_PER_PARAGRAPH,
                                                   QURAN_LINES_PER_PARAGRAPH))}

        self.no_lines = len(self.lines)
        self.no_paragraphs = max(self.lines_of_paragraph.keys())

    def get_line(self, line_id):
        if 0 <= line_id < self.no_lines:
            return self.lines[line_id]
        else:
            raise IndexError

    def get_paragraph(self, paragraph_id):
        if paragraph_id not in self.lines_of_paragraph.keys():
            raise IndexError("%d - This translation doesn't have this paragraph." % paragraph_id)

        start, finish = self.lines_of_paragraph[paragraph_id]
        return "".join(self.lines[start:finish])


class QuranMetadata(Metadata):

    def __init__(self, data_source):
        path_to_save_file = os.path.join(ROOT_DIR, 'data', 'quran', 'utils', 'metadata.pickle')
        super().__init__(data_source, path_to_save_file)


class QuranDataSource(DataSource):
    def __init__(self):
        super().__init__()
        self.all_translations_list = os.listdir('translations')
        self.no_translations = len(self.all_translations_list)

    def get_translation(self, translation_id: int) -> QuranTranslation:
        if 0 <= translation_id < self.no_translations:
            chosen_translation_path = self._find_translation(translation_id)
            return QuranTranslation(chosen_translation_path, translation_id)
        else:
            raise IndexError

    def _find_translation(self, translation_id: int):
        for translation_name in self.all_translations_list:
            if translation_id == int(translation_name.split("-")[0]):
                return os.path.join("translations", translation_name)

    def get_metadata(self) -> QuranMetadata:
        return QuranMetadata(data_source=self)


if __name__ == "__main__":
    quran_handle = QuranDataSource()
    for translation_index in random.sample(range(quran_handle.no_translations), 4):
        translation = quran_handle.get_translation(translation_index)
        print(translation.get_line(random.choice(range(translation.no_lines))))
        print(translation.get_paragraph(random.choice(range(translation.no_paragraphs))))
