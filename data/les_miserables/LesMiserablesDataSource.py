import os

from data.utils.DataSourceBase import BaseDataSource, BaseMetadata, BaseTranslation
from definitions import ROOT_DIR


class LesMiserablesTranslation(BaseTranslation):
    pass


class LesMiserablesMetadata(BaseMetadata):

    def __init__(self, data_source):
        path_to_save_file = os.path.join(ROOT_DIR, 'data', 'les_miserables', 'utils', 'metadata.pickle')
        super().__init__(data_source, path_to_save_file)


class LesMiserablesDataSource(BaseDataSource):
    def get_metadata(self) -> LesMiserablesMetadata:
        return LesMiserablesMetadata(data_source=self)
