import ast
import os

from data.utils.DataSourceInterface import Translation, DataSource


class QuranTranslation(Translation):
    def get_line(self, line_id):
        if 0 <= line_id < self.no_lines:
            return self.lines[line_id]
        else:
            raise IndexError

    def get_paragraph(self, paragraph_id):
        return self.get_line(paragraph_id)

    def __init__(self, path):
        super().__init__(path)
        with open(path, 'r') as translation:
            all_lines = translation.readlines()
            metadata_separator_line = all_lines.index("{")
            self.lines = all_lines[:metadata_separator_line]
            self.metadata = ast.literal_eval("".join(all_lines[metadata_separator_line:]))
        self.no_lines = len(self.lines)
        self.no_paragraphs = self.no_lines


class QuranDataSource(DataSource):
    def __init__(self):
        super().__init__()
        self.all_translations_list = os.listdir('translations')
        self.no_translations = len(self.all_translations_list)

    def get_translation(self, translation_id: int) -> QuranTranslation:
        if 0 <= translation_id < self.no_translations:
            chosen_translation_path = self._find_translation(translation_id)
            return QuranTranslation(chosen_translation_path)
        else:
            raise IndexError

    def _find_translation(self, translation_id: int):
        for translation_name in self.all_translations_list:
            if translation_id == int(translation_name.split("-")[0]):
                return os.path.join("translations", translation_name)
