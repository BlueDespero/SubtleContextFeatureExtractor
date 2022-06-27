import random
from typing import List, Dict

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


class DataLoaderIterator:
    def __init__(self, dataloader):
        self._dataloader: Dataloader
        self._dataloader = dataloader
        # member variable to keep track of current index
        self._index = 0

    def __next__(self):
        if self._index < self._dataloader._len:
            if self._index < len(self._team._juniorMembers):  # Check if junior members are fully iterated or not
                result = (self._team._juniorMembers[self._index], 'junior')
            else:
                result = (self._team._seniorMembers[self._index - len(self._team._juniorMembers)], 'senior')
            self._index += 1
            return result
        # End of Iteration
        raise StopIteration


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
        self._index = 0

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

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        this_batch = []
        initial_id = self._index * self.batch_size

        if self._index <= self._len:
            for paragraph_id in self._paragraph_order[initial_id: initial_id + self.batch_size]:
                paragraphs, labels = self._get_paragraphs_from_translations(paragraph_id)
                paragraphs_embeddings = [[embed, label] for embed, label in zip(paragraphs, labels)]
                this_batch.append(paragraphs_embeddings)
            self._index += 1
            return this_batch

        raise StopIteration

    def _load_translations(self) -> List[TranslationInterface]:
        prepared_translations = []
        for translation_id in self.book_translations:
            prepared_translations.append(self._data_source.get_translation(translation_id, embedding=self.embedding))
        return prepared_translations

    def _measure_length(self) -> int:
        return int(len(self.paragraphs) / self.batch_size)

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
                        embedding: str = None,
                        shuffle: bool = True,
                        device: str = 'cpu'
                        ) -> Dict[str, Dataloader]:
    """
    Function fetching three Dataloaders - for training, testing and validation.

    :param book_name: The name of the book the translations should be taken from.
        Available options are: 'bible', 'quran', 'les miserables', 'madame bovary', 'the count of monte cristo
    :param translations: List of indices specifying which translations of the book from 'book_name' should be used.
    :param training_proportion: Sets what proportion of all the translations specified should be used for training.
        All the proportions must add up to 1.
    :param testing_proportion: Sets what proportion of all the translations specified should be used for testing.
        All the proportions must add up to 1.
    :param validation_proportion: Sets what proportion of all the translations specified should be used for validation.
        All the proportions must add up to 1.
    :param batch_size: Size of the batches which Dataloaders will be providing.
    :param embedding: Can be set to 'bert' or 'simple'.
        If 'bert' is chosen, paragraphs will be represented with tensor embeddings (size 1024) generated by BertModel.
        If 'simple' is chosen, sentences are represented with IntTensors. Values of those tensors are indices
        representing the words. If left empty or None is passed, Dataloaders will yield string paragraphs, without
        embedding.
    :param shuffle: If true, order of the paragraphs is randomized.
    :param device: Dataloaders yield batches of tensors. Those tensors will be moved to the device set here.
    :return: Dict with three keys: 'training', 'testing' and 'validation'. Values of the dict are the Dataloaders
        matching the purpose described by the key.
    :rtype: Dict[str, Dataloader]
    """
    ds: DataSourceInterface
    ds = DATASOURCE_MAPPING[book_name]()
    metadata = ds.get_metadata()

    paragraphs_ids = list(range(metadata.max_number_of_paragraphs))
    random.shuffle(paragraphs_ids)

    assert (training_proportion + testing_proportion + validation_proportion) == 1.0

    training_part = int(metadata.max_number_of_paragraphs * training_proportion)
    testing_part = int(metadata.max_number_of_paragraphs * testing_proportion)
    validation_part = int(metadata.max_number_of_paragraphs * validation_proportion)

    return {'training': Dataloader(
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
