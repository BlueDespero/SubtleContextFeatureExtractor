import torch.nn as nn
from models.translations.classification.MLP import MLP
from models.translations.feature_extraction.LSTM import LSTM


class MyEnsemble(nn.Module):
    def __init__(self, feature_extractor, classifier):
        super(MyEnsemble, self).__init__()
        self.Feature_extractor = feature_extractor
        self.Classifier = classifier

    def forward(self, list_of_paragraphs):
        paragraph_embeddings = self.Feature_extractor(list_of_paragraphs)
        predictions = self.Classifier(paragraph_embeddings)
        return predictions


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

    feature_extractor = LSTM()
    classifier = MLP(4)
    model = MyEnsemble(feature_extractor, classifier)

    predictions = model(paragraphs)
    print(predictions)
    print(predictions.shape)
