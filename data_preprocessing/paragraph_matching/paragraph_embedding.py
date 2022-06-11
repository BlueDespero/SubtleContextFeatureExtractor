import gensim.downloader
from gensim.corpora.wikicorpus import WikiCorpus, tokenize
import numpy as np


class Word2vec_matching:
    def __init__(self):
        self.model = gensim.downloader.load('word2vec-google-news-300')

    def similarity_1(self, paragraph_1, paragraph_2):
        # paragraph embedding is sum of word (from w2v) embeddings
        def paragraph_vector(paragraph):
            list_of_tokens = tokenize(paragraph)
            return sum([self.model[word] for word in list_of_tokens[:100] if word in self.model.key_to_index])

        v_1 = paragraph_vector(paragraph_1)
        v_2 = paragraph_vector(paragraph_2)
        return np.dot(v_1, v_2) / (np.linalg.norm(v_1) * np.linalg.norm(v_2))

    def similarity_2(self, paragraph_1, paragraph_2):
        stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
                     "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its",
                     "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
                     "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has",
                     "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or",
                     "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between",
                     "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in",
                     "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when",
                     "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some",
                     "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can",
                     "will", "just", "don", "should", "now"]
        def paragraph_vector(paragraph):
            list_of_tokens = tokenize(paragraph)
            return sum([self.model[word] for word in list_of_tokens[:100]
                        if word in self.model.key_to_index and len(word)>3 and word not in stopwords])

        v_1 = paragraph_vector(paragraph_1)
        v_2 = paragraph_vector(paragraph_2)
        return np.dot(v_1, v_2) / (np.linalg.norm(v_1) * np.linalg.norm(v_2))
