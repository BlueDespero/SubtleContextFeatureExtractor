import os

from data.utils.DataSourceInterface import Translation, DataSource, Metadata
from definitions import ROOT_DIR


class CountTranslation(Translation):
    pass


class CountMetadata(Metadata):

    def __init__(self, data_source):
        path_to_save_file = os.path.join(ROOT_DIR, 'data', 'the_count_of_monte_cristo', 'utils', 'metadata.pickle')
        super().__init__(data_source, path_to_save_file)


class CountDataSource(DataSource):

    def get_metadata(self) -> CountMetadata:
        return CountMetadata(data_source=self)
