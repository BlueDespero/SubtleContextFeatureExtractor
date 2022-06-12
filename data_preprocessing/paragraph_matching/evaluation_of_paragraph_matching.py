from paragraph_embedding import similarity_multiset_comparison
from data_analysis import get_bible_alternative_translations
import numpy as np
from tqdm.auto import tqdm
from plotting import similarities_plot


def similarity_metric_evaluation_based_on_bible(similarity_metric, metric_name, text_length='short',
                                                number_of_joined_verses=5):
    if text_length == 'short':
        # Based on verses of single chapter
        translations = get_bible_alternative_translations(limit=2, chapter='03')
        list_of_paragraphs_1 = []
        list_of_paragraphs_2 = []
        for i in sorted(translations.keys()):
            if len(translations[i]) == 2:
                list_of_paragraphs_1.append(translations[i][0])
                list_of_paragraphs_2.append(translations[i][1])
            elif len(translations[i]) == 1:
                list_of_paragraphs_1.append(translations[i][0])
    elif text_length == 'long':
        # Based on groups of verses in a single book
        list_of_paragraphs_1 = []
        list_of_paragraphs_2 = []
        for chapter in range(1, 21):
            translations = get_bible_alternative_translations(limit=2, chapter=str(chapter).zfill(2))
            temp_1 = []
            temp_2 = []
            for i in sorted(translations.keys()):
                if len(translations[i]) == 2:
                    temp_1.append(translations[i][0])
                    temp_2.append(translations[i][1])
                if i % number_of_joined_verses == 0:
                    list_of_paragraphs_1.append(' '.join(temp_1))
                    list_of_paragraphs_2.append(' '.join(temp_2))
                    temp_1 = []
                    temp_2 = []
            if i % number_of_joined_verses != 0:
                list_of_paragraphs_1.append(' '.join(temp_1))
                list_of_paragraphs_2.append(' '.join(temp_2))

    matching = np.zeros((len(list_of_paragraphs_1), len(list_of_paragraphs_2))).astype(np.float32)
    for y, paragraph_1 in enumerate(tqdm(list_of_paragraphs_1)):
        for x, paragraph_2 in enumerate(list_of_paragraphs_2):
            matching[y, x] = similarity_metric(paragraph_1, paragraph_2)

    similarities_plot(matching, metric_name)


if __name__ == '__main__':
    similarity_metric_evaluation_based_on_bible(similarity_multiset_comparison, 'Multiset comparison', 'long')
