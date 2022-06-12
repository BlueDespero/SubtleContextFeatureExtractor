import os
import random

from data.utils.DataSourceInterface import Translation, DataSource


def initialize_paragraph_mapping():
    mapping = {}
    counter = 0

    for translation_dict in os.listdir("translations"):
        for file in os.listdir(os.path.join("translations", translation_dict)):
            if file.endswith("000_read.txt") or not file.startswith("eng"):
                continue

            with open(os.path.join("translations", translation_dict, file), 'r',
                      encoding="utf-8") as translation_chapter_text:

                book, chapter = [next(translation_chapter_text) for _ in range(2)]
                book = book.split(" ")[-1]
                chapter = int(chapter.split(" ")[-1].split(".")[0])

                if (book, chapter) not in mapping.values():
                    mapping[counter] = (book, chapter)
                    counter += 1

    return mapping, {v: k for k, v in mapping.items()}


class BibleTranslation(Translation):
    paragraph_mapping, inverse_paragraph_mapping = initialize_paragraph_mapping()

    def __init__(self, path):
        super().__init__(path)
        self.lines = []
        self.lines_of_paragraph = {}

        for file in os.listdir(path):
            if file.endswith("000_read.txt") or not file.startswith("eng"):
                continue

            with open(os.path.join(path, file), 'r', encoding="utf-8") as translation_chapter_text:

                book, chapter = [next(translation_chapter_text) for _ in range(2)]
                book = book.split(" ")[-1]
                chapter = int(chapter.split(" ")[-1].split(".")[0])

                all_chapter_lines = translation_chapter_text.readlines()

                paragraph_id = BibleTranslation.inverse_paragraph_mapping[(book, chapter)]
                self.lines_of_paragraph[paragraph_id] = (len(self.lines), len(self.lines) + len(all_chapter_lines) - 1)

                self.lines += all_chapter_lines

        self.no_lines = len(self.lines)
        self.no_paragraphs = max(self.lines_of_paragraph.keys())

    def get_line(self, line_id):
        if 0 <= line_id < self.no_lines:
            return self.lines[line_id]
        else:
            raise IndexError

    def get_paragraph(self, paragraph_id):
        if paragraph_id < 0 or paragraph_id > max(BibleTranslation.paragraph_mapping.keys()):
            raise IndexError('Paragraph of of range.')

        if paragraph_id not in self.lines_of_paragraph.keys():
            raise IndexError("%d - This translation doesn't have this paragraph." % paragraph_id)

        start, finish = self.lines_of_paragraph[paragraph_id]
        return "".join(self.lines[start:finish])


class BibleDataSource(DataSource):
    def __init__(self):
        super().__init__()
        self.all_translations_list = os.listdir('translations')
        self.no_translations = len(self.all_translations_list)

    def get_translation(self, translation_id: int) -> BibleTranslation:
        if 0 <= translation_id < self.no_translations:
            chosen_translation_path = self._find_translation(translation_id)
            return BibleTranslation(chosen_translation_path)
        else:
            raise IndexError

    def _find_translation(self, translation_id: int):
        for translation_name in self.all_translations_list:
            if translation_id == int(translation_name.split("-")[0]):
                return os.path.join("translations", translation_name)


if __name__ == "__main__":
    quran_handle = BibleDataSource()
    for translation_index in random.sample(range(quran_handle.no_translations), 4):
        translation = quran_handle.get_translation(translation_index)
        print(translation.get_line(random.choice(range(translation.no_lines))))
        print(translation.get_paragraph(random.choice(list(translation.lines_of_paragraph.keys()))))
