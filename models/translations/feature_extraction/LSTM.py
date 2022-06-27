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
    def __init__(self, dimension=128, gensim_model='glove-wiki-gigaword-50'):
        """
            :param int dimension: Dimension of a output vector.
            :param str gensim_model: Name of pretrained gensim model, suggested:
                - 'glove-wiki-gigaword-50'
                - 'glove-wiki-gigaword-200'
                - 'word2vec-google-news-300'
        """
        super(LSTM, self).__init__()

        # w2v = gensim.downloader.load(gensim_model)
        # embedding_size = w2v.vector_size

        # self.embedding = lambda list_of_paragraphs: embedding(w2v, list_of_paragraphs, embedding_size)

        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=1,
                            hidden_size=dimension,
                            num_layers=2,
                            batch_first=True)

        self.drop = nn.Dropout(p=0.5)

    def forward(self, paragraph):
        # embeds = self.embedding(paragraph)
        lstm_out, _ = self.lstm(paragraph) # batch_size, input_size, output_size
        lstm_out = self.drop(lstm_out)
        return torch.amax(lstm_out, dim=1)


if __name__ == '__main__':
    paragraphs = ["""
    We were in study hall when the
    Headmaster entered, followed by a “new boy” 
    dressed in ordinary clothes and by a classmate who was 
    carrying a large desk. Those who were asleep woke up, and every¬ 
    one stood up as if taken imawares while at work. 
    """, """The Headmaster made us a sign to be seated; then, turning 
    toward the master in charge of study hall: 
    """, """“Monsieur Roger,” he said in a low tone, “here is a student whom 
    1 am putting in your charge. He is entering the fifth form. If his 
    work and his conduct warrant it, he will be promoted to the upper 
    forms, as befits his age.”"""]

    lstm = LSTM()
    print(repr(lstm(paragraphs)))
