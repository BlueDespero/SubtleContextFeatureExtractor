import os

from data.utils.DataSourceInterface import Translation, DataSource, Metadata
from definitions import ROOT_DIR


class LesMiserablesTranslation(Translation):
    pass


class LesMiserablesMetadata(Metadata):

    def __init__(self, data_source):
        path_to_save_file = os.path.join(ROOT_DIR, 'data', 'les_miserables', 'utils', 'metadata.pickle')
        super().__init__(data_source, path_to_save_file)


class LesMiserablesDataSource(DataSource):
    def get_metadata(self) -> LesMiserablesMetadata:
        return LesMiserablesMetadata(data_source=self)
