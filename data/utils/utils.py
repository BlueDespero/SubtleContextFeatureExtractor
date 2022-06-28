import nltk


def prepare_nltk():
    '''
    Function makes sure you have punkt and stopwords nltk packages downloaded.
    If any of them is missing, it downloads it.

    :return:
    '''
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
