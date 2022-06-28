import torch
from models.translations.classification.MLP import MLP
from models.translations.feature_extraction.LSTM import LSTM
from data.utils.dataloaders import create_data_loaders
from models.translations.combined_models.simple_model import training, MyEnsemble

if __name__ == '__main__':
    print("Loading data...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    # device = 'cpu'
    l = [0, 1, 2, 3]
    init = True
    for name in ['the count of monte cristo', 'madame bovary', 'les miserables']:
        print(name)
        data_loader = create_data_loaders(name,
                                          translations=l,
                                          training_proportion=0.7,
                                          testing_proportion=0.15,
                                          validation_proportion=0.15,
                                          batch_size=32,
                                          embedding='bert',
                                          shuffle=True,
                                          device='cpu')
        print('Model initialization...')
        if init:
            feature_extractor = LSTM()
            feature_extractor.load_state_dict(torch.load(r'..\trained_models\FE_bert_4.model'))
            feature_extractor.to(device)
            init = False
        else:
            feature_extractor = LSTM()
            feature_extractor.load_state_dict(torch.load(r'..\trained_models\FE_bert.model'))
            feature_extractor.to(device)
        classifier = MLP(len(l))
        model = MyEnsemble(feature_extractor, classifier)

        print('Training...')
        training(model, data_loader, log_every=3, learning_rate=0.01, device=device, max_num_epochs=20)
        torch.save(feature_extractor.state_dict(), r'..\trained_models\FE_bert.model')
