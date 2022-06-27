import os

from data.utils.DataSourceBase import BaseDataSource, BaseMetadata, BaseMatchingTranslation
from definitions import ROOT_DIR

CENTRAL_MADAME_BOVARY_TRANSLATION = os.path.join(ROOT_DIR, 'data', 'madame_bovary', 'translations',
                                                 '000-Madame_Bovary_Charlotte_Woolley_Underwood.txt')


class MadameBovaryTranslation(BaseMatchingTranslation):

    def __init__(self, path, translation_id, embedding):
        self.path_to_mapping = os.path.join(ROOT_DIR, 'data', 'madame_bovary', 'preprocessed_data',
                                            'translation_{}_mapping.pickle'.format(translation_id))
        super().__init__(path, translation_id, embedding)

    def _get_central_translation(self):
        return MadameBovaryTranslation(CENTRAL_MADAME_BOVARY_TRANSLATION, 0, None)

    def _map_paragraphs(self):
        if self.path == CENTRAL_MADAME_BOVARY_TRANSLATION:
            self._solo_paragraph_mapping()
        else:
            self._central_paragraph_mapping()


class MadameBovaryMetadata(BaseMetadata):

    def __init__(self, data_source):
        path_to_save_file = os.path.join(ROOT_DIR, 'data', 'madame_bovary', 'preprocessed_data', 'metadata.pickle')
        super().__init__(data_source, path_to_save_file)


class MadameBovaryDataSource(BaseDataSource):
    def __init__(self):
        super().__init__()
        self.all_translations_list = os.listdir(os.path.join(ROOT_DIR, "data", "madame_bovary", "translations"))
        self.no_translations = len(self.all_translations_list)

    def get_translation(self, translation_id: int, embedding=None) -> MadameBovaryTranslation:
        if 0 <= translation_id < self.no_translations:
            chosen_translation_path = self._find_translation(translation_id)
            return MadameBovaryTranslation(chosen_translation_path, translation_id, embedding)
        else:
            raise IndexError

    def _find_translation(self, translation_id: int):
        for translation_name in self.all_translations_list:
            if translation_id == int(translation_name.split("-")[0]):
                return os.path.join(ROOT_DIR, "data", "madame_bovary", "translations", translation_name)

    def get_metadata(self) -> MadameBovaryMetadata:
        return MadameBovaryMetadata(data_source=self)


if __name__ == '__main__':
    madame_hook = MadameBovaryDataSource()

    for i in range(madame_hook.no_translations):
        translation = madame_hook.get_translation(i)
        print("\n\nTranslation :" + translation.path)
        for j in range(3):
            try:
                print("Paragraph {}".format(j))
                print(translation.get_paragraph(j))
            except IndexError as e:
                print(e)
