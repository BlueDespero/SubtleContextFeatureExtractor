from data.bible.BibleDataSource import BibleDataSource
from data.utils.embeddings import IdentityEmbedding, BertEmbedding

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
