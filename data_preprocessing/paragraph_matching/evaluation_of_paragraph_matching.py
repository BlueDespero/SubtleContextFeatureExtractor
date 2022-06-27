import random
from typing import Tuple

import numpy as np
from tqdm.auto import tqdm

from data.bible.BibleDataSource import BibleDataSource, BibleTranslation
from data_preprocessing.paragraph_matching.tools.paragraph_embedding import jaccard_similarity
from data_preprocessing.paragraph_matching.tools.plotting import similarities_plot


def similarity_metric_evaluation_based_on_bible(similarity_metric, metric_name: str,
                                                chosen_translations: Tuple[int, int] = None):
    """
        Bible is a good benchmark for paragraph-matching algorithms as it is perfectly aligned
        This method uses it to check visually the performance of similarity metrics
        Perfect metric should return diagonal matching on final plot, every difference is
        sign of some mismatch

        :param function similarity_metric: function that takes two texts (paragraphs) and returns its similarity score
        :param string metric_name: Name of metric used in plot title
        :param chosen_translations: Specify which Bible translations should be used for evaluation. If None,
            two random translations will be chosen.
    """

    translation_1: BibleTranslation
    translation_2: BibleTranslation

    translation_1, translation_2 = [BibleDataSource().get_translation(i) for i in (
        chosen_translations if chosen_translations is not None else random.sample(
            range(BibleDataSource().no_translations), 2))]

    paragraphs_of_translation_1, paragraphs_of_translation_2 = [], []
    for paragraph_id in range(translation_1.no_paragraphs):
        try:
            new_paragraph_translation_1 = translation_1.get_paragraph(paragraph_id)
            new_paragraph_translation_2 = translation_2.get_paragraph(paragraph_id)
            paragraphs_of_translation_1.append(new_paragraph_translation_1)
            paragraphs_of_translation_2.append(new_paragraph_translation_2)
        except IndexError:
            pass

    matching = np.zeros((len(paragraphs_of_translation_1), len(paragraphs_of_translation_2))).astype(np.float32)
    for y, paragraph_1 in enumerate(tqdm(paragraphs_of_translation_1)):
        for x, paragraph_2 in enumerate(paragraphs_of_translation_2):
            matching[y, x] = similarity_metric(paragraph_1, paragraph_2)

    similarities_plot(matching, metric_name)


if __name__ == '__main__':
    similarity_metric_evaluation_based_on_bible(jaccard_similarity, 'Multiset comparison')
