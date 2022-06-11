from paragraph_embedding import Word2vec_matching, similarity_3
from os import listdir
from tqdm.auto import tqdm
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

def matching_two_translations(list_of_paragraphs_1,list_of_paragraphs_2,fast_mode=True,plot_results=False,pickle_result=False):
    """
        Function returns list of tuples (matched paragraphs in initial order)
        alternative to fast_mode is a more accurate model
    """
    if not fast_mode:
        w2v = Word2vec_matching()
    matching = np.zeros((len(list_of_paragraphs_1),len(list_of_paragraphs_2))).astype(np.float32)

    moving_average = [0 for _ in range(10)]
    for y, paragraph_1 in enumerate(tqdm(list_of_paragraphs_1)):
        center = np.median(moving_average)
        for x, paragraph_2 in enumerate(list_of_paragraphs_2):
            if center-15 < x < center+25:
                if fast_mode:
                    matching[y,x] = similarity_3(paragraph_1, paragraph_2) + 0.3
                else:
                    matching[y,x] = w2v.similarity_4(paragraph_1, paragraph_2) + 0.3
        moving_average.append(np.argmax(matching[y]))
        moving_average = moving_average[1:]

    results = []
    for y, row in enumerate(matching):
        if np.min(row) != np.max(row):
            results.append((y,np.argmax(row)))

    matches = [(list_of_paragraphs_1[y],list_of_paragraphs_1[x]) for y,x in results]
    if pickle_result:
        pickle.dump(matches,open(datetime.now().strftime("%d_%m_%Y__%H_%M_%S.pickle"),'wb'))

    if plot_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,5))
        ax1.imshow(matching)
        zeros = np.zeros_like(matching)
        zeros[tuple(np.array(results).T)]=1
        ax2.imshow(zeros)
        plt.suptitle("Paragraph similarity")
        ax1.set_title('Continuous')
        ax2.set_title('Discrete')
        ax1.set_xlabel('Number of a paragraph in second translation')
        ax1.set_ylabel('Number of a paragraph in first translation')
        ax2.set_xlabel('Number of a paragraph in second translation')
        plt.show()
    return matches



if __name__ == '__main__':
    def get_translation(file):
        return [line for line in file.read().split('\n\n') if len(line.split())>3]
    path = r'../../data/madame_bovary/'
    file_names = [f for f in listdir(path)]
    translations = [get_translation(open(path+file_name,'r',encoding='UTF-8')) for file_name in file_names]

    matching_two_translations(translations[0],translations[1],plot_results=True,pickle_result=True)
