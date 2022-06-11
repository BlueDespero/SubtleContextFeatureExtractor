from paragraph_embedding import Word2vec_matching, similarity_3
from data_analysis import get_bible_alternative_translations
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


def bible_evaluation_1(similarity_metric):
    # Based on verses of single chapter
    translations = get_bible_alternative_translations(limit=2, chapter='03')
    t1 = []
    t2 = []
    for i in sorted(translations.keys()):
        if len(translations[i]) == 2:
            t1.append(translations[i][0])
            t2.append(translations[i][1])
        elif len(translations[i]) == 1:
            t1.append(translations[i][0])
    results = []

    for y, paragraph_1 in enumerate(tqdm(t1)):
        temp_result = []
        for x, paragraph_2 in enumerate(t2):
            temp_result.append(similarity_metric(paragraph_1, paragraph_2))
        results.append(temp_result)

    print(results)
    plt.imshow(np.array(results))
    plt.title('Word2vec, CBOW - paragraph similarity')
    plt.xlabel('Number of a paragraph in second translation')
    plt.ylabel('Number of a paragraph in first translation')
    plt.show()


def bible_evaluation_2(similarity_metric, number_of_joined_verses=5, continuous=True):
    # Based on a single book
    paragraph_1 = []
    paragraph_2 = []
    for chapter in range(1, 21):
        translations = get_bible_alternative_translations(limit=2, chapter=str(chapter).zfill(2))
        temp_1 = []
        temp_2 = []
        for i in sorted(translations.keys()):
            if len(translations[i]) == 2:
                temp_1.append(translations[i][0])
                temp_2.append(translations[i][1])
            if i % number_of_joined_verses == 0:
                paragraph_1.append(' '.join(temp_1))
                paragraph_2.append(' '.join(temp_2))
                temp_1 = []
                temp_2 = []
        if i % number_of_joined_verses != 0:
            paragraph_1.append(' '.join(temp_1))
            paragraph_2.append(' '.join(temp_2))
    results = []
    for y, p_1 in enumerate(tqdm(paragraph_1)):
        temp_result = []
        for x, p_2 in enumerate(paragraph_2):
            temp_result.append(similarity_metric(p_1, p_2))
        if continuous:
            results.append(temp_result)
        elif max(temp_result)==min(temp_result):
            results.append([0 for _ in range(len(temp_result))])
        else:
            results.append([1 if element == max(temp_result) else 0 for element in temp_result])

    plt.imshow(np.array(results))
    plt.title('Combined methods - paragraph similarity (continuous)')
    plt.xlabel('Number of a paragraph in second translation')
    plt.ylabel('Number of a paragraph in first translation')
    plt.show()


if __name__ == '__main__':
    w2v = Word2vec_matching()
    # bible_evaluation_2(w2v.similarity_2, continuous=False)

    bible_evaluation_2(w2v.similarity_4, continuous=True)
