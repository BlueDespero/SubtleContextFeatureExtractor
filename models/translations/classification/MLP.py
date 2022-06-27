import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, output_size, dimension=128):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(nn.Linear(dimension, dimension),
                                    nn.ReLU(),
                                    nn.Linear(dimension, dimension),
                                    nn.ReLU(),
                                    nn.Linear(dimension, output_size))
        #                           nn.Softmax(dim=0)

    def forward(self, paragraph_embeddings):
        """
            :param torch.Tensor paragraph_embeddings: Two dimensional tensor (shape = (batch_size, dimension)) produced
                by a feature extractor
            :return: Predictions of given text author - vector of probabilities
            :rtype: torch.Tensor
        """
        return self.layers.forward(paragraph_embeddings)


if __name__ == '__main__':
    pass
