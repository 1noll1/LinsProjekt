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
from sklearn.model_selection import train_test_split
import argparse

def read_data():
    print('Importing Excel data')
    smaort_excel = pd.read_excel('smaorter-2015_ver2.xlsx', skiprows=list(range(9)), usecols=list(range(4, 10)))
    tatort_excel = pd.read_excel('tatorter-2015.xlsx', skiprows=list(range(10)), usecols=list(range(4, 20)))

    smaort_excel['Label'] = ['smaort' for item in smaort_excel.iterrows()]
    tatort_excel['Label'] = ['tatort' for item in tatort_excel.iterrows()]

    tatort_excel.rename(columns = {'Tätortsbeteckning': 'Distriktsnamn'}, inplace=True)

    concat_df = pd.concat([smaort_excel, tatort_excel])

    concat_df.drop_duplicates(subset='Distriktsnamn', inplace=True)
    concat_df = concat_df[['Distriktsnamn', 'Label']]

    max_len = concat_df.Distriktsnamn.map(len).max()
    print('Max len:', max_len)

    train_df, test_df = train_test_split(concat_df.sample(frac=1), test_size=0.2)

    smaort = list(smaort_excel.Distriktsnamn.unique())
    tatort = list(tatort_excel.Distriktsnamn.unique())

    smaort_train = [o for o in smaort if (train_df['Distriktsnamn']==o).any()]
    tatort_train = [t for t in tatort if (train_df['Distriktsnamn']==t).any()]

    smaort_test = [o for o in smaort if (test_df['Distriktsnamn']==o).any()]
    tatort_test = [t for t in tatort if (test_df['Distriktsnamn']==t).any()]

    sma_by = list(filter(lambda x: 'by' in x, smaort))
    sma_stad = list(filter(lambda x: 'stad' in x, smaort))

    print('Antal byar i grupp småort: {}'.format(len(sma_by)))
    print('Antal städer i grupp småort: {}'.format(len(sma_stad)))
    
    tat_by = list(filter(lambda x: 'by' in x, tatort))
    tat_stad = list(filter(lambda x: 'stad' in x, tatort))

    print('Antal byar i grupp tätort: {}'.format(len(tat_by)))
    print('Antal städer i grupp tätort: {}'.format(len(tat_stad)))
    
    with open('smaort_test.pkl', 'wb') as f:
            pickle.dump(smaort_test, f)

    with open('tatort_test.pkl', 'wb') as f:
            pickle.dump(tatort_test, f)

    return smaort_excel, tatort_excel, concat_df, smaort_train, tatort_train, max_len

def get_vocab(dataframe, column_name):
    vocab = set()
    for idx, row in dataframe.iterrows():
        s = row[column_name]
        vocab.update(s)
    vocab_size = len(vocab)
    print('Vocab size:', vocab_size)
    return vocab, vocab_size

class OrtLoader():
    def __init__(self, smaort, tatort, vocab, max_len):
        self.max_len = max_len
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
                    encoded.append(intort)
                    #encoded.append(torch.Tensor(intort))
                except:
                    pass
            return encoded

        encoded = ort2int(total)

        x_padded = []
        for ort in encoded:
            seq = np.zeros(self.max_len)
            i = 0
            for _int in ort:
               #seq[i] = _int[0]
               seq[i] = _int
               i += 1
               #seq = torch.Tensor(seq)
               x_padded.append(np.copy(seq))

        #x_padded = pad_sequence(ort2int(total), batch_first=True, padding_value=0)
        #x_padded = x_padded.type('torch.LongTensor')
        self.input_size = len(x_padded[0])

        self.X_tensors = [torch.LongTensor(seq) for seq in x_padded]
        #self.X_tensors = x_padded
        self.y_tensors = torch.LongTensor([0 if ort in smaort else 1 for ort in total])

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        return self.X_tensors[i].to(dev), self.y_tensors[i].to(dev)

def train_model(pretrained):
    model = GRUclassifier(vocab_size, len(dataset.X_tensors[0]), 50, 2, dev, pretrained=False)
    trained_model = trained_batches(model, 20, dev, train_loader=train_loader, loss_mode=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a place name classifier.')
    parser.add_argument('--modelfile', type=str, default="trained_model",
                    help='The name of the file you wish to save the trained model to.')
    parser.add_argument('--pretrained', type=bool, default=False,
                    help='Whether or not to use pretrained weights.')

    args = parser.parse_args()

    dev = torch.device("cuda:{}".format(hash('gusstrlip') % 4) if torch.cuda.is_available() else "cpu")
    smaort_excel, tatort_excel, concat_df, smaort_train, tatort_train, max_len = read_data()
    vocab, vocab_size = get_vocab(concat_df, 'Distriktsnamn')

    dataset = OrtLoader(smaort_train, tatort_train, vocab, max_len)
    train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0)

    print('Saving dataset to dataset.pkl')
    with open('dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)

    model = GRUclassifier(vocab_size, len(dataset.X_tensors[0]), 50, 2, dev, args.pretrained)
    trained_model = trained_batches(model, 20, dev, train_loader=train_loader, loss_mode=1)
    filename = args.modelfile
    print('Saving model to {}'.format(filename))
    torch.save(trained_model, filename)
