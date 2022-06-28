import random


def shuffling(dl):
    train_len = len(dl['train'])
    valid_len = len(dl['validation'])
    total = dl['train'] + dl['validation'] + dl['test']
    random.shuffle(total)
    new_train = []
    new_valid = []
    new_test = []
    for i, sample in enumerate(total):
        if i < train_len:
            new_train.append(sample)
        elif i < train_len + valid_len:
            new_valid.append(sample)
        else:
            new_test.append(sample)
    return {'train': new_train,
            'validation': new_valid,
            'test': new_test}
