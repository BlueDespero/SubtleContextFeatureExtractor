import os
import pickle
from collections import defaultdict

import nltk
import numpy as np
from nltk.corpus import stopwords
from tqdm import trange

from data.utils.DataSourceInterface import DataSourceInterface, MetadataInterface, TranslationInterface
from data.utils.embeddings import NAME_TO_EMBEDDING, Embedding
from data.utils.utils import prepare_nltk


class BaseTranslation(TranslationInterface):
    def __init__(self, path, translation_id, embedding):
        super().__init__(path, translation_id, embedding)

        if not hasattr(self, 'lines'):
            self.lines = []
        if not hasattr(self, 'lines_of_paragraph'):
            self.lines_of_paragraph = {}

        self.no_lines = len(self.lines)
        self.no_paragraphs = max(self.lines_of_paragraph.keys())

        self.path_to_embedding_file = os.path.join(os.path.dirname(os.path.dirname(self.path)),
                                                   'preprocessed_data',
                                                   'translation_{t_id}_paragraphs_embedded_{e_name}.pickle'.format(
                                                       t_id=translation_id,
                                                       e_name=embedding))
        self.embedder: Embedding
        self.embedder = None
        self.embedded_paragraphs = {}
        if self.embedding:
            self._prepare_embedder()
            self._prepare_embedded_paragraphs()

    def get_line(self, line_id):
        """
        Get a single line from the translation.

        :param line_id: The line of the translation which shall be returned
        :type line_id: int
        :return: The text of the line matching the line_id
        :rtype: str
        :raise IndexError: When line_id requested is not in the collection
        """

        if 0 <= line_id < self.no_lines:
            return self.lines[line_id]
        else:
            raise IndexError("Line id must be between 0 and {no_lines}. Is {l_id}".format(no_lines=self.no_lines,
                                                                                          l_id=line_id))

    def get_paragraph(self, paragraph_id):
        """
        Get a greater section of a text, big enough to recognize the style of the translation author.
        It should be used to get the data for the neural net we are training.

        :param paragraph_id: identifier of the paragraph which shall be returned.
        :type paragraph_id: int
        :return: The text of a chosen paragraph. We aim to make them be around 4 lines.
        :rtype: str
        :raise IndexError: when paragraph_id requested is not in the collection
        """
        if self.embedding:
            return self._get_embedded_paragraph(paragraph_id)
        else:
            return self._get_pure_paragraph(paragraph_id)

    def _get_embedded_paragraph(self, paragraph_id):
        if paragraph_id in self.embedded_paragraphs.keys():
            return self.embedded_paragraphs[paragraph_id]
        else:
            self.embedder.get_filler_embedding()

    def _get_pure_paragraph(self, paragraph_id):
        if paragraph_id not in self.lines_of_paragraph.keys():
            raise IndexError("Paragraph with id {p_id} doesn't"
                             " occur in translation {t_id}".format(p_id=paragraph_id, t_id=self.translation_id))

        start, finish = self.lines_of_paragraph[paragraph_id]
        return "".join(self.lines[start:finish])

    def _prepare_embedded_paragraphs(self):
        try:
            self._load_embedded_paragraphs()
        except FileNotFoundError:
            self._create_embedded_paragraphs()

    def _load_embedded_paragraphs(self):

        with open(self.path_to_embedding_file, 'rb') as embedding_file:
            self.embedded_paragraphs = pickle.load(embedding_file)

    def _create_embedded_paragraphs(self):
        for i in range(self.no_paragraphs):
            try:
                self.embedded_paragraphs[i] = self.embedder.encode(self._get_pure_paragraph(i))
            except IndexError:
                self.embedded_paragraphs[i] = self.embedder.get_filler_embedding()

        self._save_embedding()

    def _prepare_embedder(self):
        self.embedder = NAME_TO_EMBEDDING[self.embedding]()

    def _save_embedding(self):
        with open(self.path_to_embedding_file, 'wb') as embedding_file:
            pickle.dump(self.embedded_paragraphs, embedding_file)


class BaseMetadata(MetadataInterface):
    def __init__(self, data_source, path_to_save_file=None):
        super().__init__(data_source, path_to_save_file)

        self.longest_paragraph_length = -np.inf
        self.max_number_of_paragraphs = -np.inf

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
            v: k for k, v in self.words_to_idx.items()
        })

    def _get_metadata(self):
        prepare_nltk()
        self.words_to_idx = defaultdict(lambda: 0)
        self.words_to_idx.update({
            "<UNK>": 0,
            "<PAD>": 1
        })
        word_idx_counter = len(self.words_to_idx.keys())

        for translation_id in trange(self.data_source.no_translations):
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

        with open(self.path_to_save_file, 'wb') as metadata_pickle:
            pickle.dump({
                "longest_paragraph_length": self.longest_paragraph_length,
                "max_number_of_paragraphs": self.max_number_of_paragraphs,
                "words_to_idx": {k: v for k, v in self.words_to_idx.items()}
            }, metadata_pickle)

        self.no_unique_words = len(self.words_to_idx.keys())
        self.idx_to_words = defaultdict(lambda: "<UNK>")
        self.idx_to_words.update({
            v: k for k, v in self.words_to_idx.items()
        })

    def _purge_words(self, paragraph):
        list_of_tokens = nltk.word_tokenize(paragraph)
        return [word for word in list_of_tokens
                if len(word) > 2
                and word not in stopwords.words('english')
                and word not in self.words_to_idx.keys()]


class BaseDataSource(DataSourceInterface):

    def __init__(self):
        super().__init__()
