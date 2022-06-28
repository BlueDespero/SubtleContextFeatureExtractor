from time import time

from tqdm import tqdm

from data.utils.dataloaders import create_data_loaders

if __name__ == '__main__':
    for book_name in ['earthsea', 'harry potter', 'i robot', 'the witcher']:
        test_loaders = create_data_loaders(book_name,
                                           translations=[0],
                                           training_proportion=0.7,
                                           testing_proportion=0.15,
                                           validation_proportion=0.15,
                                           batch_size=64,
                                           embedding='bert',
                                           shuffle=True,
                                           device='cpu')

        print(test_loaders)

        print('Loop')
        t_0 = time()
        for batch in tqdm(test_loaders['train']):
            print(batch)
        t_1 = time()
        print(t_1 - t_0)
