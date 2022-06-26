import os

from data.utils.DataSourceInterface import Translation, DataSource, Metadata
from definitions import ROOT_DIR


class MadameBovaryTranslation(Translation):
    pass


class MadameBovaryMetadata(Metadata):

    def __init__(self, data_source):
        path_to_save_file = os.path.join(ROOT_DIR, 'data', 'madame_bovary', 'utils', 'metadata.pickle')
        super().__init__(data_source, path_to_save_file)


class MadameBovaryDataSource(DataSource):
    pass
