import io
import pandas as pd
import ast
#import csv
import numpy as np
import pickle
from tqdm import tqdm
import os
import argparse

def fetch_oov(vocab, vectors):
    print('Writing out-of-vocabulary words to OOV_words.txt')
    with open('OOV_words.txt', 'w') as f:
        for word in vocab:
            if word not in vectors:
                f.write(word + '\n')

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    oov = []

    print('Looking for fasttext vectors in vocabulary...')
    for line in tqdm(fin):
        tokens = line.rstrip().split(' ')
        if tokens[0] not in vocab:
            continue
        else:
            data[tokens[0]] = list(map(float, tokens[1:]))
    print('Done.')

    print('Fasttext vectors make out {} percent of the vocabulary.'.format(len(data)/len(vocab) * 100))

    return data

def add_oov(oovfile):
    oov = {}
    with open(oovfile, 'r') as f:
        for line in f:
            line = line.split(' ')
            w = line[0]
            # slicing excludes the \n at the end
            v = line[1:-1]
            vec = {w: np.array(v)}
            oov.update(vec)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a place name classifier.')
    parser.add_argument('--fastbin', type=str, default='cc.sv.300.bin',
                    help='The binary file containing the pretrained fasttext vectors.')
    parser.add_argument('--fastvec', type=str, default='cc.sv.300.vec',
                    help='Name of the file containing the pretrained fasttext vectors.')
    parser.add_argument('--scriptpath', type=str, default='/Users/list/Documents/ML\ course/Project/fastText/fasttext',
                    help='Path to the fasttext print-word-vectors script.')
    args = parser.parse_args()

    with open('sammansattningar.pkl', 'rb') as f:
        sammansattningar = pickle.load(f)
    vocab = sammansattningar.values()

    #print(vocab)
    #fetch_oov(vocab)
    #os.system('{} print-word-vectors {} < OOV_words.txt > OOV_vectors.txt'.format(args.scriptpath, args.fastbin))

    flatten = lambda l: [item for sublist in vocab for item in sublist]
    vocab = flatten(vocab)
    vectors = load_vectors(args.fastvec)

    fetch_oov(vocab, vectors)
    print('Adding OOV vectors to model')
    os.system('{} print-word-vectors {} < OOV_words.txt > OOV_vectors.txt'.format(args.scriptpath, args.fastbin))

    oov = add_oov('OOV_vectors.txt')
    print('Updating vector dictionary with OOV vectors')
    vectors.update(oov)

    print('Saving fasttext vectors to fasttext_vectors.pkl.')
    with open('fasttext_vectors.pkl', 'wb') as f:
        pickle.dump(vectors, f)
