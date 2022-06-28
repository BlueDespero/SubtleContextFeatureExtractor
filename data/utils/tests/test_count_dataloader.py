from time import time

from tqdm import tqdm

from data.utils.dataloaders import create_data_loaders

test_loaders = create_data_loaders('the count of monte cristo',
                                   translations=[0, 1, 2, 3],
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
    pass
t_1 = time()
print(t_1 - t_0)
