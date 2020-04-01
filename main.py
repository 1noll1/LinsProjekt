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

def get_vocab(dataframe, column_name):
    vocab = set()
    for idx, row in dataframe.iterrows():
        s = row[column_name]
        vocab.update(s)
    vocab_size = len(vocab)
    print('Vocab size:', vocab_size)
    return vocab, vocab_size

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a place name classifier.')
    parser.add_argument('--modelfile', type=str, default="trained_model",
                    help='The name of the file you wish to save the trained model to.')
    parser.add_argument('--dataset', type=str, default='default_dataset',
                    help='Path you wish to save the dataset to.')
    parser.add_argument('--pretrained', type=bool, default=False,
                    help='Whether or not to use pretrained weights.')

    args = parser.parse_args()

    dev = torch.device("cuda:{}".format(hash('gusstrlip') % 4) if torch.cuda.is_available() else "cpu")
    
    smaort_excel, tatort_excel, concat_df, smaort_train, tatort_train = read_data()
    vocab, vocab_size = get_vocab(concat_df, 'Distriktsnamn')
    max_len = max(map(lambda x: len(x), concat_df['Distriktsnamn']))

    dataset = OrtLoader(smaort_train, tatort_train, vocab, max_len, dev, orter=None)
    train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0)

    datapath = 'datasets/' + args.dataset
    print('Saving dataset to {}'.format(datapath))
    with open(datapath, 'wb') as f:
        pickle.dump(dataset, f)

    model = GRUclassifier(vocab_size, len(dataset.X_tensors[0]), 50, 1, dev, args.pretrained)
    trained_model = trained_batches(model, 20, dev, train_loader=train_loader)
    filepath = 'trained_models/' + args.modelfile
    print('Saving model to {}'.format(filepath))
    torch.save(trained_model, filepath)
