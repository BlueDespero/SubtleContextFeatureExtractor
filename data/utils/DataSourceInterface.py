import pickle

import numpy as np


class Translation:
    def __init__(self, path, translation_id):
        self.no_lines = None
        self.no_paragraphs = None
        self.translation_id = translation_id

    def get_line(self, line_id):
        raise NotImplemented

    def get_paragraph(self, paragraph_id):
        raise NotImplemented


class Metadata:
    def __init__(self, data_source, path_to_save_file=None):
        self.data_source: DataSource
        self.data_source = data_source
        self.longest_paragraph_length = -np.inf
        self.max_number_of_paragraphs = -np.inf
        self.path_to_save_file = path_to_save_file

        try:
            self._load_metadata()
        except FileNotFoundError:
            self._get_metadata()

    def __str__(self):
        return str(
            {
                "longest_paragraph_length": self.longest_paragraph_length,
                "max_number_of_paragraphs": self.max_number_of_paragraphs
            }
        )

    def _load_metadata(self):
        with open(self.path_to_save_file, 'rb') as metadata_pickle:
            metadata_dict = pickle.load(metadata_pickle)
        self.max_number_of_paragraphs = metadata_dict['max_number_of_paragraphs']
        self.longest_paragraph_length = metadata_dict['longest_paragraph_length']

    def _get_metadata(self):

        for translation_id in range(self.data_source.no_translations):
            translation = self.data_source.get_translation(translation_id)

            if translation.no_paragraphs > self.max_number_of_paragraphs:
                self.max_number_of_paragraphs = translation.no_paragraphs

            for paragraph_id in range(translation.no_paragraphs):
                paragraph = translation.get_paragraph(paragraph_id)
                try:
                    if (paragraph_text_len := len(paragraph)) > self.longest_paragraph_length:
                        self.longest_paragraph_length = paragraph_text_len

                except IndexError:
                    pass

        with open(self.path_to_save_file, 'wb') as metadata_pickle:
            pickle.dump({
                "longest_paragraph_length": self.longest_paragraph_length,
                "max_number_of_paragraphs": self.max_number_of_paragraphs
            }, metadata_pickle)


class DataSource:

    def __init__(self):
        self.no_translations = None

    def get_translation(self, translation_id) -> Translation:
        raise NotImplemented

    def get_metadata(self) -> Metadata:
        raise NotImplemented
