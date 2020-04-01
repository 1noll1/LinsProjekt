import pandas as pd
import torch
import numpy as np
from train import trained_batches
from GRU_model import GRUclassifier
from torch import optim
from torch import nn
from OrtLoader import OrtLoader
from torch.utils.data import Dataset, DataLoader  
import pickle
import argparse
from read_data import read_data
from collections.abc import Iterable

def get_vocab(sammansattningar):
    vocab = set()
    for w in sammansattningar:
        vocab.update(w)
    vocab_size = len(vocab)
    print('Vocab size:', vocab_size)
    return vocab, vocab_size

def flatten(l):
    # this function was written by user Cristian on stackoverflow.
    # https://stackoverflow.com/a/2158532
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

def load_compounds():
    with open('sammansattningar.pkl', 'rb') as f:
        orter = pickle.load(f)

    for k in orter:
        # remove spaces from the compounds
        orter[k] = [filter(None, s.split(' ')) for s in orter[k]]
        orter[k] = list(map(lambda x: [str(c) for c in x], orter[k]))
        orter[k] = list(flatten(orter[k]))

    sammansattningar = orter.values()

    return sammansattningar, orter

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a place name classifier.')
    parser.add_argument('--modelfile', type=str, default="fastText_trained_model",
                    help='The name of the file you wish to save the trained model to.')
    parser.add_argument('--dataset', type=str, default='fastText_dataset',
                    help='Path you wish to save the dataset to.')
    parser.add_argument('--pretrained', type=bool, default=False,
                    help='Whether or not to use pretrained fastText weights.')
    parser.add_argument('--epochs', type=int, default=20,
                    help='Number of epochs you wish to train the model for.')

    args = parser.parse_args()

    dev = torch.device("cuda:{}".format(hash('gusstrlip') % 4) if torch.cuda.is_available() else "cpu")
    np.random.seed(42)

    _, _, total, smaort_train, tatort_train = read_data()
    sammansattningar, orter = load_compounds()
    vocab, vocab_size = get_vocab(sammansattningar)
    max_len = max(map(lambda x: len(x), orter.values()))

    dataset = OrtLoader(smaort_train, tatort_train, vocab, max_len, dev, orter=orter)
    train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0)

    if args.pretrained == True:
        print('Loading fastText vectors')
        with open('fasttext_vectors.pkl', 'rb') as f:
            fasttext = pickle.load(f)
        int2vec = {dataset.char2int[i]: list(fasttext[i]) for i in dataset.char2int}

        for b in int2vec:
            # we just need to change the dtype as some of the vectors come from strings:
            if type(int2vec[b][0]) == np.str_:
                int2vec[b] = np.array([float(a) for a in int2vec[b]])

        weights = torch.FloatTensor([i for i in list(int2vec.values())])

        print('weights:', weights.shape)
        model = GRUclassifier(vocab_size, len(dataset.X_tensors[0]), 50, 1, dev, weights)

    if args.pretrained == False:
        model = GRUclassifier(vocab_size, len(dataset.X_tensors[0]), 50, 1, dev, None)

    datapath = 'datasets/' + args.dataset
    print('Saving dataset to {}'.format(datapath))
    with open(datapath, 'wb') as f:
        pickle.dump(dataset, f)

    trained_model = trained_batches(model, args.epochs, dev, train_loader=train_loader)

    filepath = 'trained_models/' + args.modelfile
    print('Saving model to {}'.format(filepath))
    torch.save(trained_model, filepath)
