import torch.nn as nn
from models.translations.classification.MLP import MLP
from models.translations.feature_extraction.LSTM import LSTM


class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB

    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x1)
        return x2

if __name__ == '__main__':
    feature_extractor = MLP()
    classifier = LSTM()
    model = MyEnsemble(feature_extractor, classifier)