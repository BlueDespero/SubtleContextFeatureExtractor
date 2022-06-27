import random
from time import time
from typing import List

from tqdm import tqdm

from data.bible.BibleDataSource import BibleDataSource
from data.les_miserables.LesMiserablesDataSource import LesMiserablesDataSource
from data.madame_bovary.MadameBovaryDataSource import MadameBovaryDataSource
from data.quran.QuranDataSource import QuranDataSource
from data.the_count_of_monte_cristo.CountDataSource import CountDataSource
from data.utils.DataSourceInterface import DataSourceInterface, TranslationInterface
from data.utils.embeddings import NAME_TO_EMBEDDING, Embedding

DATASOURCE_MAPPING = {
    'quran': QuranDataSource,
    'bible': BibleDataSource,
    'les miserables': LesMiserablesDataSource,
    'madame bovary': MadameBovaryDataSource,
    'the count of monte cristo': CountDataSource
}


def get_data_source_object_from_name(book_name: str) -> DataSourceInterface:
    try:
        datasource = DATASOURCE_MAPPING[book_name.lower()]
        return datasource()
    except KeyError:
        raise KeyError("{name} - no such book in the database. "
                       "Available picks: {list}".format(name=book_name, list=DATASOURCE_MAPPING.keys()))


class Dataloader:
    def __init__(self,
                 book_name: str,
                 book_translations: List[int],
                 batch_size: int,
                 paragraphs: List[int],
                 embedding: str = 'bert',
                 shuffle: bool = True,
                 device: str = 'cpu') -> None:
        self.book_name = book_name
        self.book_translations = book_translations
        self.batch_size = batch_size
        self.paragraphs = paragraphs
        self.embedding = embedding
        self.shuffle = shuffle
        self.device = device

        self._data_source = get_data_source_object_from_name(self.book_name)
        self._translations = self._load_translations()
        self._len = self._measure_length()
        self._metadata = self._data_source.get_metadata()
        self._embedder = self._get_embedder()
        self._paragraph_order = self._get_order()

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, item):
        this_batch = []
        initial_id = item * self.batch_size

        for paragraph_id in self._paragraph_order[initial_id: initial_id + self.batch_size]:
            paragraphs, labels = self._get_paragraphs_from_translations(paragraph_id)
            paragraphs_embeddings = [[embed, label] for embed, label in zip(paragraphs, labels)]
            this_batch.append(paragraphs_embeddings)
        return this_batch

    def _load_translations(self) -> List[TranslationInterface]:
        prepared_translations = []
        for translation_id in self.book_translations:
            prepared_translations.append(self._data_source.get_translation(translation_id, embedding=self.embedding))
        return prepared_translations

    def _measure_length(self) -> int:
        max_paragraphs = max([translation.no_paragraphs for translation in self._translations])
        return max_paragraphs / self.batch_size

    def _get_embedder(self) -> Embedding:
        return NAME_TO_EMBEDDING[self.embedding.lower()](device=self.device, metadata=self._metadata)

    def _get_paragraphs_from_translations(self, paragraph_id):
        paragraphs = []
        labels = []
        for t in self._translations:
            try:
                paragraph = t.get_paragraph(paragraph_id)
            except IndexError:
                paragraph = "..."
            paragraphs.append(paragraph)
            labels.append(t.translation_id)
        return paragraphs, labels

    def _get_order(self):
        order = self.paragraphs
        if self.shuffle:
            random.shuffle(order)
        return order


def create_data_loaders(book_name: str,
                        translations: List[int],
                        training_proportion: float,
                        testing_proportion: float,
                        validation_proportion: float,
                        batch_size: int,
                        embedding: str = 'bert',
                        shuffle: bool = True,
                        device: str = 'cpu'
                        ):
    ds: DataSourceInterface
    ds = DATASOURCE_MAPPING[book_name]()
    metadata = ds.get_metadata()

    paragraphs_ids = list(range(metadata.max_number_of_paragraphs))
    random.shuffle(paragraphs_ids)

    assert (training_proportion + testing_proportion + validation_proportion) == 1.0

    training_part = int(metadata.max_number_of_paragraphs * training_proportion)
    testing_part = int(metadata.max_number_of_paragraphs * testing_proportion)
    validation_part = int(metadata.max_number_of_paragraphs * validation_proportion)

    return {'train': Dataloader(
        book_name,
        translations,
        batch_size,
        paragraphs_ids[:training_part],
        embedding=embedding,
        shuffle=shuffle,
        device=device
    ),
        'validation': Dataloader(
            book_name,
            translations,
            batch_size,
            paragraphs_ids[training_part:training_part + testing_part],
            embedding=embedding,
            shuffle=shuffle,
            device=device
        ),
        'test': Dataloader(
            book_name,
            translations,
            batch_size,
            paragraphs_ids[-validation_part:],
            embedding=embedding,
            shuffle=shuffle,
            device=device
        )
    }


if __name__ == "__main__":
    test_loaders = create_data_loaders('madame bovary',
                                       translations=[0, 1, 2, 3],
                                       training_proportion=0.7,
                                       testing_proportion=0.15,
                                       validation_proportion=0.15,
                                       batch_size=64,
                                       embedding='bert',
                                       shuffle=True,
                                       device='cpu')
    print(test_loaders)

    print('Loop')
    t_0 = time()
    for batch in tqdm(test_loaders['train']):
        pass
    t_1 = time()
    print(t_1 - t_0)
