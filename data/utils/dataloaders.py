from typing import List


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

    def __len__(self) -> int:
        pass

    def __getitem__(self, item):
        pass


def create_data_loaders(book_name, translations_count):
    pass
