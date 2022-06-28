import torch
from models.translations.classification.MLP import MLP
from models.translations.feature_extraction.LSTM import LSTM
from data.utils.dataloaders import create_data_loaders
from models.translations.combined_models.combined_model import training, MyEnsemble
from utils import shuffling


def hypothesis_data_loader(restriction=3):
    load = lambda name: create_data_loaders(name,
                                            translations=[0],
                                            training_proportion=0.7,
                                            testing_proportion=0.15,
                                            validation_proportion=0.15,
                                            batch_size=50,
                                            embedding='bert',
                                            shuffle=True,
                                            device='cpu')
    books = ['earthsea', 'harry potter', 'i robot', 'the witcher']
    train = []
    validation = []
    test = []
    for author_id, book_name in enumerate(books):
        dl = load(book_name)
        for set_object, set_name in zip([train, validation, test], ['train', 'validation', 'test']):
            i = 0
            for batch in dl[set_name]:
                set_object.append([[[sample[0][0], author_id]] for sample in batch])
                i += 1
                if set_name == 'train' and i > restriction:
                    break

    dl = {'train': train, 'validation': validation, 'test': test}
    return shuffling(dl)


def testing_hypothesis_1(log_every=20, learning_rate=0.001, max_num_epochs=6):
    print("Loading data...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    data_loaders = [hypothesis_data_loader(restriction=2),
                    hypothesis_data_loader(restriction=3),
                    hypothesis_data_loader(restriction=4),
                    hypothesis_data_loader(restriction=5),
                    hypothesis_data_loader(restriction=6),
                    hypothesis_data_loader(restriction=7)]

    print('Testing pre-trained model')
    for i, data_loader in enumerate(data_loaders):
        print(i)
        feature_extractor = LSTM()
        feature_extractor.load_state_dict(torch.load(r'..\translations\trained_models\FE_bert_4.model'))
        feature_extractor.to(device)
        classifier = MLP(4)
        model = MyEnsemble(feature_extractor, classifier)
        training(model, data_loader, log_every=log_every, learning_rate=learning_rate, device=device,
                 max_num_epochs=max_num_epochs, save=False)

    print('Testing randomly initialized model')
    for i, data_loader in enumerate(data_loaders):
        print(i)
        feature_extractor = LSTM()
        feature_extractor.to(device)
        classifier = MLP(4)
        model = MyEnsemble(feature_extractor, classifier)
        training(model, data_loader, log_every=log_every, learning_rate=learning_rate, device=device,
                 max_num_epochs=max_num_epochs, save=False)
