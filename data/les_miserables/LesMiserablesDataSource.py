import os

from data.utils.DataSourceBase import BaseDataSource, BaseMetadata, BaseMatchingTranslation
from definitions import ROOT_DIR

CENTRAL_LES_MISERABLES_TRANSLATION = os.path.join(ROOT_DIR, 'data', 'les_miserables', 'translations',
                                                  '000-les_miserables_Bigelow_Smith_and_Company.txt')


class LesMiserablesTranslation(BaseMatchingTranslation):
    def __init__(self, path, translation_id, embedding):
        self.path_to_mapping = os.path.join(ROOT_DIR, 'data', 'les_miserables', 'preprocessed_data',
                                            'translation_{}_mapping.pickle'.format(translation_id))
        super().__init__(path, translation_id, embedding)

    def _get_central_translation(self):
        return LesMiserablesTranslation(CENTRAL_LES_MISERABLES_TRANSLATION, 0, None)

    def _map_paragraphs(self):
        if self.path == CENTRAL_LES_MISERABLES_TRANSLATION:
            self._solo_paragraph_mapping()
        else:
            self._central_paragraph_mapping()


class LesMiserablesMetadata(BaseMetadata):

    def __init__(self, data_source):
        path_to_save_file = os.path.join(ROOT_DIR, 'data', 'les_miserables', 'preprocessed_data', 'metadata.pickle')
        super().__init__(data_source, path_to_save_file)


class LesMiserablesDataSource(BaseDataSource):
    def __init__(self):
        super().__init__()
        self.all_translations_list = os.listdir(os.path.join(ROOT_DIR, "data", "les_miserables", "translations"))
        self.no_translations = len(self.all_translations_list)

    def get_translation(self, translation_id: int, embedding=None) -> LesMiserablesTranslation:
        if 0 <= translation_id < self.no_translations:
            chosen_translation_path = self._find_translation(translation_id)
            return LesMiserablesTranslation(chosen_translation_path, translation_id, embedding)
        else:
            raise IndexError

    def _find_translation(self, translation_id: int):
        for translation_name in self.all_translations_list:
            if translation_id == int(translation_name.split("-")[0]):
                return os.path.join(ROOT_DIR, "data", "les_miserables", "translations", translation_name)

    def get_metadata(self) -> LesMiserablesMetadata:
        return LesMiserablesMetadata(data_source=self)
