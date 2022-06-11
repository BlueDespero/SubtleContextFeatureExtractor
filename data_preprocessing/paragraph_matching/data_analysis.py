from os import listdir
from os.path import isfile, join
from collections import defaultdict as dd
import os
import numpy as np

def get_bible_alternative_translations(book_name='002_GEN',limit=None,**kwargs):
    path_to_bible = r'../../data/bible/translations/'
    file_names = [f for f in listdir(path_to_bible) if not isfile(join(path_to_bible, f))]
    chapter = kwargs.get('chapter','02')
    translations = dd(list)
    max_length = 0
    for file_num, file_name in enumerate(file_names):
        path = os.path.join(path_to_bible,
                            f'{file_name}/',
                            f'{file_name[:-10]}_{book_name}_{chapter}_read.txt')
        try:
            with open(path,'r',encoding='UTF-8') as file:
                for i, line in enumerate(file):
                    if i<=max_length:
                        translations[i+1].append(line.rstrip())
                    else:
                        max_length = i
                        for _ in range(file_num):
                            translations[i + 1].append("")
                        translations[i + 1].append(line.rstrip())
                if i < max_length:
                    for j in range(i,max_length):
                        translations[j + 1].append("")

        except:
            if limit is not None:
                limit += 1
        if file_num+1 >= limit:
            break
    return translations
