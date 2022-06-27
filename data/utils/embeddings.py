from typing import List

import nltk
import torch
from transformers import BertTokenizer, BertModel

from data.bible.BibleDataSource import BibleDataSource
from data.utils.DataSourceInterface import Metadata, prepare_nltk


class Embedding:
    def __init__(self, target_length: int = None, device: str = 'cpu', metadata: Metadata = None):
        self.metadata = metadata
        self.target_length = target_length
        self.device = device

    def encode(self, sentence: str):
        raise NotImplemented

    def encode_batch(self, list_of_sentences: List[str]):
        raise NotImplemented


class BertEmbedding(Embedding):
    model_name = "bert-large-cased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    def _encode(self, input_ids):
        with torch.no_grad():
            output = BertEmbedding.model(
                input_ids
            )
            sequence_output, pooled_output = output[:2]
            return pooled_output[0]

    def encode(self, sentence: str):
        encoded = BertEmbedding.tokenizer.batch_encode_plus(
            [sentence],
            padding='longest',
            return_tensors="pt",
        )
        input_id = encoded["input_ids"]

        return self._encode(input_id)

    def encode_batch(self, list_of_sentences: List[str]):
        encoded = BertEmbedding.tokenizer.batch_encode_plus(
            list_of_sentences,
            padding='longest',
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"]
        result = []
        for input_id in input_ids:
            encode_input = torch.reshape(input_id, (1, -1))
            result.append(self._encode(encode_input))

        return result


class IdentityEmbedding(Embedding):

    def __init__(self, target_length: int = None, device: str = 'cpu', metadata: Metadata = None):
        super().__init__(target_length, device, metadata)
        if self.metadata is None:
            raise AttributeError("For IdentityEmbedding ('none' embedding option) metadata needs to be specified.")
        prepare_nltk()

    def encode(self, sentence: str):
        list_of_tokens = nltk.word_tokenize(sentence)
        embedding = [self.metadata.words_to_idx[word] for word in list_of_tokens]
        embedding = embedding + [0] * (self.metadata.longest_paragraph_length - len(embedding))
        embedding = torch.IntTensor(embedding, device=self.device)
        return embedding

    def encode_batch(self, list_of_sentences: List[str]):
        return [self.encode(sentence) for sentence in list_of_sentences]


NAME_TO_EMBEDDING = {
    'bert': BertEmbedding,
    'none': IdentityEmbedding
}

if __name__ == "__main__":
    sentence1 = "There is a cat playing with a ball"
    sentence2 = "Blood for the blood god, skulls for the skull throne."
    embedder = BertEmbedding()

    embedding1 = embedder.encode(
        "In the very beginning, long, long ago, God made the earth below and the heavens above. At the time when God "
        "first started creating the world, "
        "it was all watery. Water covered everything and that was all. There was no dry land yet, only water, "
        "and it was dark. It was dark just like a cave is dark inside at night. But the spirit of God was moving over "
        "the water. "
        "Then God spoke. “Let there be light,” he said. And then the light came out. "
        "God looked and saw that the light was good. Then he separated the light from the darkness. "
        "He called the light “Day” and the darkness “Night.” Then night came. That was the first night. Then a new "
        "day dawned and it was morning. "
    )
    embedding2 = embedder.encode_batch([sentence1, sentence2])
    embedding3 = embedder.encode_batch([sentence2, sentence1])

    print(embedding1)
    print(embedding2)
    print(embedding3)

    bible_hook = BibleDataSource()
    bible_translation = bible_hook.get_translation(1)
    bible_metadata = bible_hook.get_metadata()

    embedder = IdentityEmbedding(metadata=bible_metadata)
    embedding1 = embedder.encode(
        bible_translation.get_line(4)
    )
    embedding2 = embedder.encode_batch([bible_translation.get_line(1), bible_translation.get_line(2)])
    embedding3 = embedder.encode_batch([bible_translation.get_line(2), bible_translation.get_line(1)])

    print(embedding1)
    print(embedding2)
    print(embedding3)
