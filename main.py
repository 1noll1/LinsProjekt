import pandas as pd
import torch
import numpy as np
from train import trained_batches
from GRU_model import GRUclassifier
from torch import optim
from torch import nn
from OrtLoader import OrtLoader
from torch.utils.data import Dataset, DataLoader  
import csv
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

# class OrtLoader():
#     def __init__(self, smaort, tatort, vocab, max_len):
#         self.max_len = max_len
#         self.vocab = vocab
#         char2int = {t:n for n, t in enumerate(vocab)}
#         total = smaort + tatort
#         self.len = len(total)

#         def ort2int(total):   
#             encoded = []         
#             for i, ort in enumerate(total):
#                 try:
#                     intort = [char2int[t] for t in ort]
#                     encoded.append(intort)
#                 except:
#                     pass
#             return encoded

#         encoded = ort2int(total)

#         x_padded = []
#         for ort in encoded:
#             seq = np.zeros(self.max_len)
#             i = 0
#             for _int in ort:
#                seq[i] = _int
#                i += 1
#                x_padded.append(np.copy(seq))

#         self.input_size = len(x_padded[0])

#         self.X_tensors = [torch.LongTensor(seq) for seq in x_padded]
#         self.y_tensors = torch.LongTensor([0 if ort in smaort else 1 for ort in total])

#     def __len__(self):
#         return self.len

#     def __getitem__(self, i):
#         return self.X_tensors[i].to(dev), self.y_tensors[i].to(dev)

def train_model(pretrained):
    model = GRUclassifier(vocab_size, len(dataset.X_tensors[0]), 50, 2, dev, pretrained=False)
    trained_model = trained_batches(model, 20, dev, train_loader=train_loader, loss_mode=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a place name classifier.')
    parser.add_argument('--modelfile', type=str, default="trained_model",
                    help='The name of the file you wish to save the trained model to.')
    parser.add_argument('--dataset', type=str, default='default_dataset',
                    help='The name of the file you wish to save the dataset to.')
    parser.add_argument('--pretrained', type=bool, default=False,
                    help='Whether or not to use pretrained weights.')

    args = parser.parse_args()

    dev = torch.device("cuda:{}".format(hash('gusstrlip') % 4) if torch.cuda.is_available() else "cpu")
    smaort_excel, tatort_excel, concat_df, smaort_train, tatort_train = read_data()
    vocab, vocab_size = get_vocab(concat_df, 'Distriktsnamn')
    max_len = max(map(lambda x: len(x), concat_df['Distriktsnamn']))
    print('Max len:', max_len)

    dataset = OrtLoader(smaort_train, tatort_train, vocab, max_len, dev, sammansattningar=None)
    train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0)

    print('Saving dataset to {}'.format(args.dataset))
    with open(args.dataset, 'wb') as f:
        pickle.dump(dataset, f)

    model = GRUclassifier(vocab_size, len(dataset.X_tensors[0]), 50, 2, dev, args.pretrained)
    trained_model = trained_batches(model, 20, dev, train_loader=train_loader, loss_mode=1)
    filename = args.modelfile
    print('Saving model to {}'.format(filename))
    torch.save(trained_model, filename)
