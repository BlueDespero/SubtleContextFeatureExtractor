from typing import List

from transformers import BertTokenizer, BertModel
import numpy as np
import torch


class Embedding:
    def encode(self, sentence: str):
        raise NotImplemented

    def encode_batch(self, list_of_sentences: List[str]):
        raise NotImplemented


class BertEmbedding(Embedding):
    model_name = "bert-large-cased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    def encode(self, sentence: str):
        encoded = BertEmbedding.tokenizer.batch_encode_plus(
            [sentence],
            max_length=512,
            pad_to_max_length=False,
            return_tensors="pt",
        )

        input_ids = np.array(encoded["input_ids"], dtype="int32")
        output = BertEmbedding.model(
            input_ids
        )
        sequence_output, pooled_output = output[:2]
        return pooled_output[0]

    def encode_batch(self, list_of_sentences: List[str]):
        encoded = BertEmbedding.tokenizer.batch_encode_plus(
            list_of_sentences,
            max_length=512,
            pad_to_max_length=False,
            return_tensors="pt",
        )

        with torch.no_grad():
            # get the model embeddings
            embeds = model(**encodings)
            input_ids = np.array(encoded["input_ids"], dtype="int32")
            output = BertEmbedding.model(
                input_ids
            )
            sequence_output, pooled_output = output[:2]
            return pooled_output[0]


class IdentityEmbedding(Embedding):

    def encode(self, sentence: str):
        pass


NAME_TO_EMBEDDING = {
    'bert': BertEmbedding,
    'none': IdentityEmbedding
}

if __name__ == "__main__":
    sentence1 = "There is a cat playing with a ball"
    embedder = BertEmbedding()
    embedding = embedder.encode(sentence1)
    print(embedding)
