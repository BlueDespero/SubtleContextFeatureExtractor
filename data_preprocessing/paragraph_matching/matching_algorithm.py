import os.path
import pickle
from datetime import datetime
from os import listdir

import numpy as np
from tqdm.auto import tqdm

from definitions import ROOT_DIR
from paragraph_embedding import Word2vec_matching, jaccard_similarity
from plotting import similarities_plot

LOOK_BACK_AMOUNT = 20
LOOK_AHEAD_AMOUNT = 40


def matching_two_translations(list_of_paragraphs_1, list_of_paragraphs_2, fast_mode=True, plot_results=False,
                              pickle_result=False):
    """
        :param list list_of_paragraphs_1: List of paragraphs from the first translation
        :param list list_of_paragraphs_2: List of paragraphs from the second translation
        :param bool fast_mode: Changes similarity metric from simple and fast to more sophisticated and accurate
        :param bool plot_results: Switches plotting functionality
        :param bool pickle_result: If True, list of a tuples of a corresponding paragraphs will be saved as pickle file
        :return: List of a tuples of a corresponding paragraphs
        :rtype: list
    """
    similarity = jaccard_similarity if fast_mode else Word2vec_matching().similarity_combined
    matching_matrix = np.zeros((len(list_of_paragraphs_1), len(list_of_paragraphs_2))).astype(np.float32)

    moving_average = [0] * 20
    for y, paragraph_1 in enumerate(tqdm(list_of_paragraphs_1)):
        center = int(np.median(moving_average))

        for i, paragraph_2 in enumerate(list_of_paragraphs_2[
                                        max(0, center - LOOK_BACK_AMOUNT):
                                        min(center + LOOK_AHEAD_AMOUNT, len(list_of_paragraphs_2) - 1)]):
            x = i + max(0, center - LOOK_BACK_AMOUNT)
            matching_matrix[y, x] = similarity(paragraph_1, paragraph_2) + 0.3

        moving_average.append(np.argmax(matching_matrix[y]))
        moving_average = moving_average[1:]

    results = []
    for y, row in enumerate(matching_matrix):
        if np.min(row) != np.max(row):
            results.append((y, np.argmax(row)))

    matches = [(y, x) for y, x in results]
    if pickle_result:
        pickle.dump(matches, open(datetime.now().strftime("%d_%m_%Y__%H_%M_%S.pickle"), 'wb'))

    if plot_results:
        similarities_plot(matching_matrix, 'Multiset comparison')
    return matches


def matching(translations, fast_mode=True, pickle_result=False):
    """
        :param list translations: List of different book translations - list of lists of strings (paragraphs)
        :param bool fast_mode: Changes similarity metric from simple and fast to more sophisticated and accurate
        :param bool pickle_result: If True, list of a tuples of a corresponding paragraphs will be saved as pickle file
        :return: List of a tuples of a corresponding paragraphs
        :rtype: list
    """
    similarity = jaccard_similarity if fast_mode else Word2vec_matching().similarity_combined
    translations.sort(key=len, reverse=True)
    central_translation, translations = translations[0], translations[1:]
    matching = {central_translation_paragraph_id: [] for central_translation_paragraph_id in
                range(len(central_translation))}
    for translation_id, translation in enumerate(translations):
        print(f"Currently working on translation number {translation_id}.")
        matching_matrix = np.zeros((len(central_translation), len(translation))).astype(np.float32)
        moving_average = [0 for _ in range(20)]
        for y, paragraph_1 in enumerate(tqdm(central_translation)):
            center = np.median(moving_average)
            for x, paragraph_2 in enumerate(translation):
                if center - 45 < x < center + 55:
                    matching[y, x] = similarity(paragraph_1, paragraph_2) + 0.3
            moving_average.append(np.argmax(matching_matrix[y]))
            moving_average = moving_average[1:]

        for y, row in enumerate(matching_matrix):
            if np.min(row) != np.max(row):
                matching[y].append(np.argmax(row))
            else:
                matching[y].append(None)
    matches = []
    for k, v in matching.items():
        temp = [central_translation[k]]
        for paragraph_id, translation in zip(v, translations):
            if paragraph_id is not None:
                temp.append(translation[paragraph_id])
            else:
                temp.append("")
        matches.append(temp)
    if pickle_result:
        pickle.dump(matches, open(datetime.now().strftime("%d_%m_%Y__%H_%M_%S.pickle"), 'wb'))
    return matches


if __name__ == '__main__':
    def get_translation(file):
        return [line for line in file.read().split('\n\n') if len(line.split()) > 3]


    path = os.path.join(ROOT_DIR, 'data','madame_bovary','translations')
    translations = [get_translation(open(os.path.join(path, file_name), 'r', encoding='UTF-8'))[:20] for file_name in
                    listdir(path)]

    matching_result = matching_two_translations(translations[0], translations[1], fast_mode=True, plot_results=False,
                              pickle_result=False)
    print(matching_result)
    pass
    # matching(translations, fast_mode=True, pickle_result=True)
