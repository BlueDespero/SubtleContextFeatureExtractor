from typing import List

import nltk
import torch
from transformers import BertTokenizer, BertModel

from data.utils.DataSourceInterface import MetadataInterface
from data.utils.utils import prepare_nltk


class Embedding:
    def __init__(self, target_length: int = None, device: str = 'cpu', metadata: MetadataInterface = None):
        self.metadata = metadata
        self.target_length = target_length
        self.device = device

    def encode(self, sentence: str):
        raise NotImplemented

    def encode_batch(self, list_of_sentences: List[str]):
        raise NotImplemented

    def get_filler_embedding(self):
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
            return list(pooled_output)

    def encode(self, sentence: str) -> torch.Tensor:
        encoded = BertEmbedding.tokenizer.batch_encode_plus(
            [sentence],
            padding='longest',
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        input_id = encoded["input_ids"]

        return self._encode(input_id)[0]

    def encode_batch(self, list_of_sentences: List[str]) -> List[torch.Tensor]:
        encoded = BertEmbedding.tokenizer.batch_encode_plus(
            list_of_sentences,
            padding='longest',
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        input_ids = encoded["input_ids"]
        encoded = self._encode(input_ids)
        return encoded

    def get_filler_embedding(self):
        return self.encode('<FILL>')


class IdentityEmbedding(Embedding):

    def __init__(self, target_length: int = None, device: str = 'cpu', metadata: MetadataInterface = None):
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

    def get_filler_embedding(self):
        return [0] * self.metadata.longest_paragraph_length


NAME_TO_EMBEDDING = {
    'bert': BertEmbedding,
    'none': IdentityEmbedding
}

