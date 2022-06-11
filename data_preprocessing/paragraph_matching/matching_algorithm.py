from paragraph_embedding import Word2vec_matching
import numpy as np

def matching(list_of_paragraphs_1,list_of_paragraphs_2,aligned=True):
    """
        Function returns list of tuples (matched paragraphs in initial order)
        Provide information if two list of paragraphs are more or less 'aligned'
    """

    w2v = Word2vec_matching()
    matching = np.zeros((len(list_of_paragraphs_1),len(list_of_paragraphs_2))).astype(np.float32)
    for y, paragraph_1 in enumerate(list_of_paragraphs_1):
        for x, paragraph_2 in enumerate(list_of_paragraphs_2):
            matching[y,x] = w2v.similarity_4(paragraph_1, paragraph_2)

