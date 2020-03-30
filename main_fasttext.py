import pandas as pd
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from train import trained_batches
from GRU_model import GRUclassifier
from torch import optim
from torch import nn
from torch.utils.data import Dataset, DataLoader  
import csv
import ast
import pickle
from read_data import read_data
from sklearn.model_selection import train_test_split
import argparse
from OrtLoader import OrtLoader

dev = torch.device("cuda:{}".format(hash('gusstrlip') % 4) if torch.cuda.is_available() else "cpu")

def get_vocab(sammansattningar):
    vocab = set()
    for w in sammansattningar:
        vocab.update(w)
    vocab_size = len(vocab)
    print('Vocab size:', vocab_size)
    return vocab, vocab_size

def load_vectors():
    vectors = pd.read_csv('fastText_vectors.csv', names=['Word', 'Vector'])

    with open('sammansattningar.pkl', 'rb') as f:
        orter = pickle.load(f)

    for k in orter:
        # remove spaces
        orter[k] = [filter(None, s.split(' ')) for s in orter[k]]
        orter[k] = list(map(lambda x: [str(c) for c in x], orter[k]))[0]

    sammansattningar = orter.values()

    return sammansattningar, orter

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a place name classifier.')
    parser.add_argument('--modelfile', type=str, default="fastText_trained_model",
                    help='The name of the file you wish to save the trained model to.')
    parser.add_argument('--dataset', type=str, default='fastText_dataset',
                    help='Path you wish to save the dataset to.')
    parser.add_argument('--pretrained', type=bool, default=False,
                    help='Whether or not to use pretrained weights.')

    args = parser.parse_args()

    dev = torch.device("cuda:{}".format(hash('gusstrlip') % 4) if torch.cuda.is_available() else "cpu")

    _, _, total, smaort_train, tatort_train = read_data()
    sammansattningar, orter = load_vectors()
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
        model = GRUclassifier(vocab_size, len(dataset.X_tensors[0]), 50, 2, dev, weights)

    if args.pretrained == False:
        model = GRUclassifier(vocab_size, len(dataset.X_tensors[0]), 50, 2, dev, args.pretrained)

    datapath = 'datasets/' + args.dataset
    print('Saving dataset to {}'.format(datapath))
    with open(datapath, 'wb') as f:
        pickle.dump(dataset, f)

    trained_model = trained_batches(model, 20, dev, train_loader=train_loader, loss_mode=1)

    filepath = 'trained_models/' + args.modelfile
    #print('Saving model to {}'.format(filename))
    print('Saving model to {}'.format(filepath))
    #torch.save(trained_model, filename)
    torch.save(trained_model, filepath)

    # filename = args.modelfile
    # print('Saving model to {}'.format(filename))
    # torch.save(trained_model, filename)
