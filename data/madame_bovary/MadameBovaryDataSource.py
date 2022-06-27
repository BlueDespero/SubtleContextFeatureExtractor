import os
import pickle
from collections import defaultdict

from data.utils.DataSourceInterface import Translation, DataSource, Metadata
from data_preprocessing.paragraph_matching.matching_algorithm import matching_two_translations
from definitions import ROOT_DIR

CENTRAL_MADAME_BOVARY_TRANSLATION = os.path.join(ROOT_DIR, 'data', 'madame_bovary', 'translations',
                                                 '000-Madame_Bovary_Charlotte_Woolley_Underwood.txt')
DEFAULT_PARAGRAPH_LENGTH = 6
MAX_PARAGRAPH_LENGTH = 20


class MadameBovaryTranslation(Translation):

    def __init__(self, path, translation_id):
        super().__init__(path, translation_id)
        self.lines_of_paragraph = {}
        self.lines = None

        with open(path, 'r', encoding="utf-8") as translation_text:
            self.lines = translation_text.readlines()

        self._map_paragraphs()

        self.no_lines = len(self.lines)
        self.no_paragraphs = max(self.lines_of_paragraph.keys())

    def get_line(self, line_id):
        if 0 <= line_id < self.no_lines:
            return self.lines[line_id]
        else:
            raise IndexError("Line id must be between 0 and {no_lines}. Is {l_id}".format(no_lines=self.no_lines,
                                                                                          l_id=line_id))

    def get_paragraph(self, paragraph_id):
        if paragraph_id in self.lines_of_paragraph.keys():
            start, finish = self.lines_of_paragraph[paragraph_id]
            return "".join(self.lines[start:finish])
        else:
            raise IndexError("Paragraph with id {p_id} doesn't"
                             " occur in translation {t_id}".format(p_id=paragraph_id, t_id=self.translation_id))

    def _get_central_translation(self):
        return MadameBovaryTranslation(CENTRAL_MADAME_BOVARY_TRANSLATION, 0)

    def _map_paragraphs(self):
        if self.path == CENTRAL_MADAME_BOVARY_TRANSLATION:
            self._solo_paragraph_mapping()
        else:
            self._central_paragraph_mapping()

    def _default_paragraph_mapping(self):
        return {i: (start, start + DEFAULT_PARAGRAPH_LENGTH) for i, start in
                enumerate(range(0, len(self.lines) - DEFAULT_PARAGRAPH_LENGTH,
                                DEFAULT_PARAGRAPH_LENGTH))}

    def _solo_paragraph_mapping(self):
        try:
            self._load_precomputed_mapping()
        except FileNotFoundError:
            self.lines_of_paragraph = self._default_paragraph_mapping()
            self._save_precomputed_mapping()

    def _central_paragraph_mapping(self):
        try:
            self._load_precomputed_mapping()
        except FileNotFoundError:
            temp_paragraph_mapping = self._default_paragraph_mapping()
            central_translation = self._get_central_translation()

            temp_paragraphs = [self.lines[start:finish] for start, finish in temp_paragraph_mapping.values()]
            central_paragraphs = [central_translation.get_paragraph(i) for i in
                                  range(central_translation.no_paragraphs)]

            matching = matching_two_translations(temp_paragraphs, central_paragraphs)

            created_mapping = defaultdict(list)
            for temp, central in matching:
                created_mapping[central] += list(temp_paragraph_mapping[temp])

            for paragraph_id, indices in created_mapping.items():
                start = min(indices)
                finish = max(indices)
                if finish - start <= MAX_PARAGRAPH_LENGTH:
                    self.lines_of_paragraph[paragraph_id] = (start, finish)
            self._save_precomputed_mapping()

    def _load_precomputed_mapping(self):
        path_to_mapping = os.path.join(ROOT_DIR, 'data', 'madame_bovary', 'utils',
                                       'translation_{}_mapping.pickle'.format(self.translation_id))
        with open(path_to_mapping, 'rb') as mapping_file:
            self.lines_of_paragraph = pickle.load(mapping_file)

    def _save_precomputed_mapping(self):
        path_to_mapping = os.path.join(ROOT_DIR, 'data', 'madame_bovary', 'utils',
                                       'translation_{}_mapping.pickle'.format(self.translation_id))
        with open(path_to_mapping, 'wb') as mapping_file:
            pickle.dump(self.lines_of_paragraph, mapping_file)


class MadameBovaryMetadata(Metadata):

    def __init__(self, data_source):
        path_to_save_file = os.path.join(ROOT_DIR, 'data', 'madame_bovary', 'utils', 'metadata.pickle')
        super().__init__(data_source, path_to_save_file)


class MadameBovaryDataSource(DataSource):
    def __init__(self):
        super().__init__()
        self.all_translations_list = os.listdir(os.path.join(ROOT_DIR, "data", "madame_bovary", "translations"))
        self.no_translations = len(self.all_translations_list)

    def get_translation(self, translation_id: int) -> MadameBovaryTranslation:
        if 0 <= translation_id < self.no_translations:
            chosen_translation_path = self._find_translation(translation_id)
            return MadameBovaryTranslation(chosen_translation_path, translation_id)
        else:
            raise IndexError

    def _find_translation(self, translation_id: int):
        for translation_name in self.all_translations_list:
            if translation_id == int(translation_name.split("-")[0]):
                return os.path.join(ROOT_DIR, "data", "madame_bovary", "translations", translation_name)

    def get_metadata(self) -> MadameBovaryMetadata:
        return MadameBovaryMetadata(data_source=self)
