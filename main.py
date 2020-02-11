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
import pickle

def read_data():
    data = pd.read_excel('smaorter-2015_ver2.xlsx', skiprows=list(range(9)), usecols=list(range(4, 10)))
    data2 = pd.read_excel('tatorter-2015.xlsx', skiprows=list(range(10)), usecols=list(range(4, 20)))

    data.drop_duplicates(subset='Distriktsnamn', inplace=True)

    smaort = list(data.Distriktsnamn.unique())
    tatort = list(data2.Tätortsbeteckning.unique())

    sma_by = list(filter(lambda x: 'by' in x, smaort))
    sma_stad = list(filter(lambda x: 'stad' in x, smaort))

    print('Antal byar i grupp småort: {}'.format(len(sma_by)))
    print('Antal städer i grupp småort: {}'.format(len(sma_stad)))
    
    tat_by = list(filter(lambda x: 'by' in x, tatort))
    tat_stad = list(filter(lambda x: 'stad' in x, tatort))

    print('Antal byar i grupp tätort: {}'.format(len(tat_by)))
    print('Antal städer i grupp tätort: {}'.format(len(tat_stad)))
    
    with open('smaort.pkl', 'wb') as f:
            pickle.dump(smaort, f)

    with open('tatort.pkl', 'wb') as f:
            pickle.dump(tatort, f)

    return data, data2, smaort, tatort

def get_vocab(dataframes, column_names):
    vocab = set()
    for column_name, dataframe in zip(column_names, dataframes):
        for idx, row in dataframe.iterrows():
            s = row[column_name]
            vocab.update(s)
    vocab_size = len(vocab)
    print('Vocab size:', vocab_size)
    return vocab, vocab_size

class OrtLoader():
    def __init__(self, smaort, tatort, vocab):

        self.vocab = vocab
        char2int = {t:n for n, t in enumerate(vocab)}
        total = smaort + tatort
        self.len = len(total)

        def ort2int(total):   
            encoded = []         
            for i, ort in enumerate(total):
                try:
                    intort = [char2int[t] for t in ort]
                    #total[i] = np.array(intort)
                    encoded.append(torch.Tensor(intort))
                except:
                    pass
            return encoded

        x_padded = pad_sequence(ort2int(total), batch_first=True, padding_value=0)
        x_padded = x_padded.type('torch.LongTensor')
        self.input_size = len(x_padded[0])

        self.X_tensors = x_padded
        self.y_tensors = torch.LongTensor([0 if ort in smaort else 1 for ort in total])

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        return self.X_tensors[i].to(dev), self.y_tensors[i].to(dev)

def train_model():
    model = GRUclassifier(vocab_size, len(dataset.X_tensors[0]), 50, 2, dev, pretrained=False)
    trained_model = trained_batches(model, 20, dev, train_loader=train_loader, loss_mode=1)
    print('Saving trained model to trained_batch32_epoch20')
    #torch.save(trained_model, 'trained_batch32_epoch20')

if __name__ == '__main__':
    dev = torch.device("cuda:{}".format(hash('gusstrlip') % 4) if torch.cuda.is_available() else "cpu")
    data, data2, smaort, tatort = read_data()
    vocab, vocab_size = get_vocab([data, data2], ['Distriktsnamn', 'Tätortsbeteckning'])

    dataset = OrtLoader(smaort, tatort, vocab)
    train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0)

    print('Saving dataset to dataset.pkl')
    with open('dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)

    model = GRUclassifier(vocab_size, len(dataset.X_tensors[0]), 50, 2, dev, pretrained=False)
    trained_model = trained_batches(model, 20, dev, train_loader=train_loader, loss_mode=1)
    filename = 'trained_batch32_epoch20'
    print('Saving model to {}'.format(filename))
    torch.save(trained_model, filename)
