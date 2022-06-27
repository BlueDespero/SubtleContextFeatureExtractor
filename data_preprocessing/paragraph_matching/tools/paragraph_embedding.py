import logging
import sys

import gensim.downloader
import numpy as np
from gensim.corpora.wikicorpus import tokenize
from multiset import Multiset
from nltk.corpus import stopwords

from data.utils.utils import prepare_nltk

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)


class Word2vec_matching:
    def __init__(self):
        # Initialization of gensim model might take few seconds, however it should be executed once before process of
        # calculating similarities
        self.model = gensim.downloader.load('word2vec-google-news-300')
        prepare_nltk()

    def similarity_simple(self, paragraph_1, paragraph_2):
        # This function computes similarity as cosine of two paragraph embeddings
        # Paragraph embedding is sum of embeddings of its words (except stopwords)
        def paragraph_vector(paragraph):
            list_of_tokens = tokenize(paragraph)
            return sum([self.model[word] for word in list_of_tokens
                        if
                        word in self.model.key_to_index and len(word) > 3 and word not in stopwords.words('english')] +
                       [np.zeros(300).astype(np.float32)])

        v_1 = paragraph_vector(paragraph_1)
        v_2 = paragraph_vector(paragraph_2)
        normalization = (np.linalg.norm(v_1) * np.linalg.norm(v_2))
        if normalization != 0.0:
            return np.dot(v_1, v_2) / normalization
        else:
            return 0.0

    def similarity_combined(self, paragraph_1, paragraph_2):
        value_1 = self.similarity_simple(paragraph_1, paragraph_2)
        value_2 = jaccard_similarity(paragraph_1, paragraph_2)
        return value_1 * value_2


def jaccard_similarity(paragraph_1: str, paragraph_2: str) -> float:
    """
    Calculate Jaccard similarity coefficient of two paragraphs.
    English stopwords and very short words are ommited.
    :param paragraph_1: First string paragraph for comparison.
    :param paragraph_2: Second string paragraph for comparison.
    :return: similarity coefficient of two paragraphs
    """

    def paragraph_preprocessing(paragraph):
        list_of_tokens = tokenize(paragraph)
        return [word for word in list_of_tokens if len(word) > 3 and word not in stopwords.words('english')]

    p_1 = paragraph_preprocessing(paragraph_1)
    p_2 = paragraph_preprocessing(paragraph_2)

    m_1 = Multiset(p_1)
    m_2 = Multiset(p_2)

    union = Multiset.union(m_1, m_2)
    intersection = Multiset.intersection(m_1, m_2)

    if len(union) == 0:
        return 0.0
    else:
        return len(intersection) / len(union)
