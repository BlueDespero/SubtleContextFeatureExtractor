from typing import List

import torch
from transformers import BertTokenizer, BertModel


class Embedding:
    def __init__(self, target_length: int = None, device: str = None):
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

    def encode(self, sentence: str):
        encoded = BertEmbedding.tokenizer.batch_encode_plus(
            [sentence],
            padding='longest',
            return_tensors="pt",
        )

        with torch.no_grad():
            input_ids = encoded["input_ids"]
            output = BertEmbedding.model(
                input_ids
            )
            sequence_output, pooled_output = output[:2]
            return pooled_output[0]

    def encode_batch(self, list_of_sentences: List[str]):
        encoded = BertEmbedding.tokenizer.batch_encode_plus(
            list_of_sentences,
            padding='longest',
            return_tensors="pt",
        )

        with torch.no_grad():
            input_ids = encoded["input_ids"]

            return [
                BertEmbedding.model.encode(input_id)[2][0] for input_id in input_ids
            ]


class IdentityEmbedding(Embedding):

    def encode(self, sentence: str):
        raise NotImplemented


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
