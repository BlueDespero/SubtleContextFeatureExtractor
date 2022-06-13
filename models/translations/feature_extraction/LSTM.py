import torch
import torch.nn as nn
import gensim.downloader
from nltk.corpus import stopwords
from gensim.corpora.wikicorpus import tokenize

def embedding_single(w2v, paragraph):
    list_of_tokens = tokenize(paragraph)
    embeddings = []

    for word in list_of_tokens:
        if word in w2v.key_to_index and len(word) > 3 and word not in stopwords.words('english'):
            emb = w2v[word]
            emb.flags['WRITEABLE'] = True
            embeddings.append(torch.from_numpy(emb.T))
    return embeddings


def embedding(w2v, list_of_paragraphs, embedding_size):
    def pad(expected_length, lst):
        return [torch.zeros(embedding_size) for _ in range(expected_length - len(lst))] + lst

    list_of_paragraph_embeddings = [embedding_single(w2v, paragraph) for paragraph in list_of_paragraphs]
    expected_length = max(map(len, list_of_paragraph_embeddings))
    list_of_paragraph_embeddings_padded = [torch.vstack(pad(expected_length, paragraph_embeddings)) for
                                           paragraph_embeddings in list_of_paragraph_embeddings]
    return torch.stack(list_of_paragraph_embeddings_padded)
class LSTM(nn.Module):
    def __init__(self,dimension=128):
        super(LSTM, self).__init__()

        w2v = gensim.downloader.load('glove-wiki-gigaword-200')
        # w2v = gensim.downloader.load('word2vec-google-news-300')
        embedding_size = w2v.vector_size

        self.embedding = lambda list_of_paragraphs: embedding(w2v,list_of_paragraphs,embedding_size)

        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=300,
                            hidden_size=dimension,
                            num_layers=1,
                            batch_first=True,
                            dropout=0.5)


    def forward(self, paragraph: str):
        x = self.fc1(paragraph)
        return x
