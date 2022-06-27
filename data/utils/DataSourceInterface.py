class TranslationInterface:
    def __init__(self, path, translation_id, embedding):
        self.no_paragraphs = None
        self.path = path
        self.translation_id = translation_id
        self.embedding = embedding

    def get_line(self, line_id):
        raise NotImplemented

    def get_paragraph(self, paragraph_id):
        raise NotImplemented


class MetadataInterface:
    def __init__(self, data_source, path_to_save_file=None):
        self.max_number_of_paragraphs = None
        self.longest_paragraph_length = None
        self.words_to_idx = {}
        self.data_source: DataSourceInterface
        self.data_source = data_source
        self.path_to_save_file = path_to_save_file

    def __str__(self):
        pass


class DataSourceInterface:

    def __init__(self):
        self.no_translations = None

    def get_translation(self, translation_id, embedding=None) -> TranslationInterface:
        raise NotImplemented

    def get_metadata(self) -> MetadataInterface:
        raise NotImplemented
