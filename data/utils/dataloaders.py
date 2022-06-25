from typing import List

from data.The_Count_of_Monte_Cristo.CountDataSource import CountDataSource
from data.bible.BibleDataSource import BibleDataSource
from data.les_miserables.LesMiserablesDataSource import LesMiserablesDataSource
from data.madame_bovary.MadameBovaryDataSource import MadameBovaryDataSource
from data.quran.QuranDataSource import QuranDataSource
from data.utils.DataSourceInterface import DataSource


def get_data_source_object_from_name(book_name: str) -> DataSource:
    match book_name.lower():
        case 'quran':
            return QuranDataSource()
        case 'bible':
            return BibleDataSource()
        case 'les miserables':
            return LesMiserablesDataSource()
        case 'madame bovary':
            return MadameBovaryDataSource()
        case 'the count of monte cristo':
            return CountDataSource()
        case _:
            raise KeyError("{} - no such book in the database".format(book_name))


class Dataloader:
    def __init__(self,
                 book_name: str,
                 book_translations: List[int],
                 batch_size: int,
                 shuffle=True,
                 device='cpu') -> None:
        self.book_name = book_name
        self.book_translations = book_translations
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

        self._data_source = get_data_source_object_from_name(self.book_name)

    def __len__(self) -> int:
        pass

    def __getitem__(self, item):
        pass


def create_data_loaders(book_name, translations_count):
    pass
