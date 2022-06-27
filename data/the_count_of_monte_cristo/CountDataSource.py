import os

from data.utils.DataSourceBase import BaseDataSource, BaseMetadata, BaseMatchingTranslation
from definitions import ROOT_DIR

CENTRAL_COUNT_TRANSLATION = os.path.join(ROOT_DIR, 'data', 'the_count_of_monte_cristo', 'translations',
                                         '000-The_Count_of_Monte_Cristo_Donohue_and_Company.txt')
DEFAULT_PARAGRAPH_LENGTH = 6
MAX_PARAGRAPH_LENGTH = 20


class CountTranslation(BaseMatchingTranslation):
    def __init__(self, path, translation_id, embedding):
        self.path_to_mapping = os.path.join(ROOT_DIR, 'data', 'the_count_of_monte_cristo', 'preprocessed_data',
                                            'translation_{}_mapping.pickle'.format(translation_id))
        super().__init__(path, translation_id, embedding)

    def _get_central_translation(self):
        return CountTranslation(CENTRAL_COUNT_TRANSLATION, 0, None)

    def _map_paragraphs(self):
        if self.path == CENTRAL_COUNT_TRANSLATION:
            self._solo_paragraph_mapping()
        else:
            self._central_paragraph_mapping()


class CountMetadata(BaseMetadata):

    def __init__(self, data_source):
        path_to_save_file = os.path.join(ROOT_DIR, 'data', 'the_count_of_monte_cristo', 'utils', 'metadata.pickle')
        super().__init__(data_source, path_to_save_file)


class CountDataSource(BaseDataSource):

    def get_metadata(self) -> CountMetadata:
        return CountMetadata(data_source=self)


if __name__ == '__main__':
    count_hook = CountDataSource()

    for i in range(count_hook.no_translations):
        translation = count_hook.get_translation(i)
        print("\n\nTranslation :" + translation.path)
        for j in range(3):
            try:
                print("Paragraph {}".format(j))
                print(translation.get_paragraph(j))
            except IndexError as e:
                print(e)
