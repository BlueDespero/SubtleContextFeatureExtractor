class Translation:
    def __init__(self, path):
        self.no_lines = None
        self.no_paragraphs = None

    def get_line(self, line_id):
        raise NotImplemented

    def get_paragraph(self, paragraph_id):
        raise NotImplemented


class DataSource:

    def __init__(self):
        self.no_translations = None

    def get_translation(self, translation_id) -> Translation:
        raise NotImplemented
