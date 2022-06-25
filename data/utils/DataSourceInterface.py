import pickle
from collections import defaultdict

import nltk
import numpy as np
from nltk.corpus import stopwords


class Translation:
    def __init__(self, path, translation_id):
        self.no_lines = None
        self.no_paragraphs = None
        self.translation_id = translation_id

    def get_line(self, line_id):
        raise NotImplemented

    def get_paragraph(self, paragraph_id):
        raise NotImplemented


def _prepare_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')


class Metadata:
    def __init__(self, data_source, path_to_save_file=None):
        self.data_source: DataSource
        self.data_source = data_source
        self.longest_paragraph_length = -np.inf
        self.max_number_of_paragraphs = -np.inf
        self.path_to_save_file = path_to_save_file

        self.words_to_idx: dict
        self.words_to_idx = None

        self.idx_to_words: dict
        self.idx_to_words = None

        self.no_unique_words = None

        try:
            self._load_metadata()
        except FileNotFoundError:
            self._get_metadata()

    def __str__(self):
        return str(
            {
                "longest_paragraph_length": self.longest_paragraph_length,
                "max_number_of_paragraphs": self.max_number_of_paragraphs,
                "no_unique_words": self.no_unique_words
            }
        )

    def _load_metadata(self):
        with open(self.path_to_save_file, 'rb') as metadata_pickle:
            metadata_dict = pickle.load(metadata_pickle)
        self.max_number_of_paragraphs = metadata_dict['max_number_of_paragraphs']
        self.longest_paragraph_length = metadata_dict['longest_paragraph_length']

        self.words_to_idx = defaultdict(lambda: 0)
        self.words_to_idx.update(metadata_dict['words_to_idx'])
        self.no_unique_words = len(self.words_to_idx.keys())
        self.idx_to_words = defaultdict(lambda: "<UNK>")
        self.idx_to_words.update({
            v: k for k, v in self.words_to_idx.keys()
        })

    def _get_metadata(self):
        _prepare_nltk()
        self.words_to_idx = defaultdict(lambda: 0)
        self.words_to_idx.update({
            "<UNK>": 0,
            "<PAD>": 1
        })
        word_idx_counter = len(self.words_to_idx.keys())

        for translation_id in range(self.data_source.no_translations):
            translation = self.data_source.get_translation(translation_id)

            if translation.no_paragraphs > self.max_number_of_paragraphs:
                self.max_number_of_paragraphs = translation.no_paragraphs

            for paragraph_id in range(translation.no_paragraphs):
                try:
                    paragraph = translation.get_paragraph(paragraph_id)
                except IndexError:
                    continue

                if (paragraph_text_len := len(paragraph)) > self.longest_paragraph_length:
                    self.longest_paragraph_length = paragraph_text_len

                interesting_words = self._purge_words(paragraph)

                for word in interesting_words:
                    self.words_to_idx[word] = word_idx_counter
                    word_idx_counter += 1

        self.no_unique_words = len(self.words_to_idx.keys())
        self.idx_to_words = defaultdict(lambda: "<UNK>")
        self.idx_to_words.update({
            v: k for k, v in self.words_to_idx.keys()
        })

        with open(self.path_to_save_file, 'wb') as metadata_pickle:
            pickle.dump({
                "longest_paragraph_length": self.longest_paragraph_length,
                "max_number_of_paragraphs": self.max_number_of_paragraphs,
                "words_to_idx": self.words_to_idx
            }, metadata_pickle)

    def _purge_words(self, paragraph):
        list_of_tokens = nltk.word_tokenize(paragraph)
        return [word for word in list_of_tokens
                if len(word) > 2
                and word not in stopwords.words('english')
                and word not in self.words_to_idx.keys()]


class DataSource:

    def __init__(self):
        self.no_translations = None

    def get_translation(self, translation_id) -> Translation:
        raise NotImplemented

    def get_metadata(self) -> Metadata:
        raise NotImplemented
