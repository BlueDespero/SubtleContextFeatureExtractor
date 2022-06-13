import logging
import sys
from collections import defaultdict as dd

import gensim.downloader
import nltk
import numpy as np
from gensim.corpora.wikicorpus import tokenize
from nltk.corpus import stopwords

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)


class Word2vec_matching:
    def __init__(self):
        # Initialization of gensim model might take few seconds, however it should be executed once before process of
        # calculating similarities
        self.model = gensim.downloader.load('word2vec-google-news-300')
        try:
            if "the" in stopwords.words('english'):
                logging.info("NLTK - English stopwords downloaded.")
        except LookupError:
            logging.error("NLTK - English stopwords missing. Trying to download...")
            nltk.download('stopwords')

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
        value_2 = similarity_multiset_comparison(paragraph_1, paragraph_2)
        return value_1 * value_2


def similarity_multiset_comparison(paragraph_1, paragraph_2):
    # In this function paragraphs are treated as multi-sets of words (except stopwords)
    # Similarity score is size of multi-sets intersection divided by multi-sets sum
    def paragraph_preprocessing(paragraph):
        list_of_tokens = tokenize(paragraph)
        return [word for word in list_of_tokens if len(word) > 3 and word not in stopwords.words('english')]

    p_1 = paragraph_preprocessing(paragraph_1)
    p_2 = paragraph_preprocessing(paragraph_2)
    d_1 = dd(int)
    d_2 = dd(int)

    for element in p_1:
        d_1[element] += 1
    for element in p_2:
        d_2[element] += 1
    union = 0
    visited = set()
    for key in d_1:
        if key not in visited:
            visited.add(key)
            union += max(d_1[key], d_2[key])

    for key in d_2:
        if key not in visited:
            visited.add(key)
            union += max(d_1[key], d_2[key])

    intersection = len([value for value in p_1 if value in p_2])

    if union == 0:
        return 0.0
    else:
        return intersection / union
