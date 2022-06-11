from data.utils.DataSourceInterface import Translation, DataSource


class QuranTranslation(Translation):
    def get_line(self, line_id):
        pass

    def get_paragraph(self, paragraph_id):
        pass

    def __init__(self):
        super().__init__()


class QuranDataSource(DataSource):
    def __init__(self):
        super().__init__()

    def get_translation(self, translation_id) -> QuranTranslation:
        pass
