import random
from typing import List

from data.bible.BibleDataSource import BibleDataSource
from data.les_miserables.LesMiserablesDataSource import LesMiserablesDataSource
from data.madame_bovary.MadameBovaryDataSource import MadameBovaryDataSource
from data.quran.QuranDataSource import QuranDataSource
from data.the_count_of_monte_cristo.CountDataSource import CountDataSource
from data.utils.DataSourceInterface import DataSource, Translation
from data.utils.embeddings import NAME_TO_EMBEDDING, Embedding

DATASOURCE_MAPPING = {
    'quran': QuranDataSource,
    'bible': BibleDataSource,
    'les miserables': LesMiserablesDataSource,
    'madame bovary': MadameBovaryDataSource,
    'the count of monte cristo': CountDataSource
}


def get_data_source_object_from_name(book_name: str) -> DataSource:
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
                 embedding: str = 'bert',
                 shuffle: bool = True,
                 device: str = 'cpu') -> None:
        self.book_name = book_name
        self.book_translations = book_translations
        self.batch_size = batch_size
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
        batch = []
        initial_id = item * self.batch_size

        for paragraph_id in self._paragraph_order[initial_id: initial_id + self.batch_size]:
            paragraphs, labels = self._get_paragraphs_from_translations(paragraph_id)
            paragraphs_embeddings = self._embedder.encode_batch(paragraphs)
            paragraphs_embeddings = [[embed, label] for embed, label in zip(paragraphs_embeddings, labels)]
            batch.append(paragraphs_embeddings)
        return batch

    def _load_translations(self) -> List[Translation]:
        prepared_translations = []
        for translation_id in self.book_translations:
            prepared_translations.append(self._data_source.get_translation(translation_id))
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
        order = list(range(self._metadata.max_number_of_paragraphs))
        if self.shuffle:
            random.shuffle(order)
        return order


def create_data_loaders(book_name: str,
                        training_translations: List[int],
                        testing_translations: List[int],
                        validation_translations: List[int],
                        batch_size: int,
                        embedding: str = 'bert',
                        shuffle: bool = True,
                        device: str = 'cpu'
                        ):
    return {'train': Dataloader(
        book_name,
        training_translations,
        batch_size,
        embedding,
        shuffle,
        device
    ),
        'validation': Dataloader(
            book_name,
            testing_translations,
            batch_size,
            embedding,
            shuffle,
            device
        ),
        'test': Dataloader(
            book_name,
            validation_translations,
            batch_size,
            embedding,
            shuffle,
            device
        )
    }


if __name__ == "__main__":
    test_loaders = create_data_loaders('bible', [1, 2, 3], [4, 5], [6, 7], batch_size=124, embedding='none',
                                       shuffle=False)
    print(test_loaders)
