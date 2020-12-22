import argparse
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

from GRU_model import GRUclassifier
from OrtLoader import OrtLoader
from pad_collate import pad_collate
from read_data import read_data
from train import trained_batches


def get_vocab(dataframe, column_name):
    v = set()
    for idx, row in dataframe.iterrows():
        s = row[column_name]
        v.update(s)
    v_size = len(v)
    print('Vocab size:', v_size)
    return v, v_size


def load_args():
    parser = argparse.ArgumentParser(description='Train a place name classifier.')
    parser.add_argument('--modelfile', type=str, default="trained_model",
                        help='The name of the file you wish to save the trained model to.')
    parser.add_argument('--dataset', type=str, default='default_dataset',
                        help='Path you wish to save the dataset to.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs you wish to train the model for.')

    args = parser.parse_args()
    return args


def main():
    args = load_args()
    dev = torch.device("cuda:{}".format(hash('gusstrlip') % 4) if torch.cuda.is_available() else "cpu")
    np.random.seed(42)

    smaort_excel, tatort_excel, concat_df, smaort_train, tatort_train = read_data()
    vocab, vocab_size = get_vocab(concat_df, 'Distriktsnamn')
    max_len = max(map(lambda x: len(x), concat_df['Distriktsnamn']))

    dataset = OrtLoader(smaort_train, tatort_train, vocab, max_len, dev, orter=None)
    train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=pad_collate)

    datapath = 'datasets/' + args.dataset
    print('Saving dataset to {}'.format(datapath))
    with open(datapath, 'wb') as f:
        pickle.dump(dataset, f)

    model = GRUclassifier(vocab_size, max_len, 50, 1, dev, None)
    trained_model = trained_batches(model, args.epochs, dev, train_loader=train_loader)
    filepath = 'trained_models/' + args.modelfile
    print('Saving model to {}'.format(filepath))
    torch.save(trained_model, filepath)


if __name__ == '__main__':
    main()

