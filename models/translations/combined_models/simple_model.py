import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.nn.functional import nll_loss
from models.translations.classification.MLP import MLP
from models.translations.feature_extraction.LSTM import LSTM
from time import time
from models.data_loaders import InMemDataLoader


class MyEnsemble(nn.Module):
    def __init__(self, feature_extractor, classifier):
        super(MyEnsemble, self).__init__()
        self.Feature_extractor = feature_extractor
        self.Classifier = classifier

    def forward(self, list_of_paragraphs):
        paragraph_embeddings = self.Feature_extractor(list_of_paragraphs)
        predictions = self.Classifier(paragraph_embeddings)
        return predictions

    @staticmethod
    def loss(predictions, targets):
        return nll_loss(predictions, targets)

def plot_history(history):
    """Helper to plot the trainig progress over time."""
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 2, 1)
    train_loss = np.array(history["train_losses"])
    plt.semilogy(np.arange(train_loss.shape[0]), train_loss, label="batch train loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    train_errs = np.array(history["train_errs"])
    plt.plot(np.arange(train_errs.shape[0]), train_errs, label="batch train error rate")
    val_errs = np.array(history["val_errs"])
    plt.plot(val_errs[:, 0], val_errs[:, 1], label="validation error rate", color="r")
    plt.ylim(0, 0.20)
    plt.legend()

def compute_error_rate(model, data_loader, device="cpu"):
    """Evaluate model on all samples from the data loader.
    """
    # Put the model in eval mode, and move to the evaluation device.
    model.eval()
    model.to(device)
    if isinstance(data_loader, InMemDataLoader):
        data_loader.to(device)

    num_errs = 0.0
    num_examples = 0
    # we don't need gradient during eval!
    with torch.no_grad():
        for x, y in data_loader:
            # x = x.to(device)
            # y = y.to(device)
            outputs = model.forward(x)
            _, predictions = outputs.data.max(dim=1)
            num_errs += (predictions != y.data).sum().item()
            num_examples += y.size(0)
    return num_errs / num_examples


def training(model,
             data_loaders,
             max_num_epochs=10,
             log_every=100,
             learning_rate=0.05,
             device="cpu"):
    model.train()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {"train_losses": [], "train_errs": [], "val_errs": []}
    best_val_err = np.inf
    best_params = None
    epoch = 0
    iter_ = 0

    try:
        tstart = time()
        while epoch < max_num_epochs:
            epoch += 1
            for batch, target in data_loaders['train']:
                iter_ += 1
                optimizer.zero_grad()
                predictions = model(batch)
                loss = model.loss(predictions, target)
                loss.backward()

                _, predictions = predictions.max(dim=1)
                batch_err_rate = (predictions != target).sum().item() / predictions.size(0)

                history["train_losses"].append(loss.item())
                history["train_errs"].append(batch_err_rate)

                optimizer.step()

                if iter_ % log_every == 0:
                    print(
                        "Minibatch {0: >6}  | loss {1: >5.2f} | err rate {2: >5.2f}%, steps/s {3: >5.2f}".format(
                            iter_,
                            loss.item(),
                            batch_err_rate * 100.0,
                            iter_ / (time() - tstart),
                        )
                    )
                    tstart = time()
            val_err_rate = compute_error_rate(model, data_loaders["valid"], device)
            history["val_errs"].append((iter_, val_err_rate))

            if val_err_rate < best_val_err:
                # Adjust num of epochs
                best_epoch = epoch
                best_val_err = val_err_rate
                best_params = [p.detach().cpu() for p in model.parameters()]
            m = "After epoch {0: >2} | valid err rate: {1: >5.2f}%".format(epoch, val_err_rate * 100.0)
            print("{0}\n{1}\n{0}".format("-" * len(m), m))
    except KeyboardInterrupt:
        if best_params is not None:
            print("\nLoading best params on validation set (epoch %d)\n" % (best_epoch))
            with torch.no_grad():
                for param, best_param in zip(model.parameters(), best_params):
                    param[...] = best_param
        test_err_rate = compute_error_rate(model, data_loaders["test"])
        m = (
            f"Test error rate: {test_err_rate * 100.0:.3f}%, "
            f"training took {time.time() - tstart:.0f}s."
        )
        print("{0}\n{1}\n{0}".format("-" * len(m), m))
        plot_history(history)


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
    targets = torch.tensor([0, 1, 3])

    feature_extractor = LSTM()
    classifier = MLP(4)
    model = MyEnsemble(feature_extractor, classifier)
    data_loader = {'train':[(paragraphs,targets)],
                   'valid':[(paragraphs,targets)],
                   'test':[(paragraphs,targets)]}
    training(model,data_loader,log_every=2)
