from torch.nn.utils.rnn import pad_sequence
import argparse
import torch
from torch.utils.data import Dataset, DataLoader 
import pickle
import numpy as np
from OrtLoader import OrtLoader

def model_eval(model, test_loader):
    model.eval()
    true = []
    pred = []
    with torch.no_grad():
        print('Initialising evaluation')
        for ort, label in test_loader:
            true.extend([label])
            ort = ort.unsqueeze(0)
            out = model(ort)
            _, predicted = torch.max(out.data, 1)
            predicted = predicted.to('cpu')
            pred.extend(predicted)

    a = 0
    for pred,label in zip(pred,true):
        if pred == label:
            a += 1
    print('Number of test samples:', len(test_loader))
    print('Accurate predictions: {}'.format(a))
    print('Total accuracy: {} {}'.format((a/len(test_loader)) * 100, '%'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate place name classifier.')
    parser.add_argument('--dataset', type=str, default="datasets/default_dataset",
                    help='The name of the file containing the dataset.')
    parser.add_argument('--modelfile', type=str, default="trained_models/trained_model",
                    help='The name of the file containing the model to be evaluated.')

    args = parser.parse_args()

    dev = torch.device("cuda:{}".format(hash('gusstrlip') % 4) if torch.cuda.is_available() else "cpu")

    with open('smaort_test.pkl', 'rb') as f:
        smaort = pickle.load(f)
    with open('tatort_test.pkl', 'rb') as f:
        tatort = pickle.load(f)

    # make sure to load the fasttext dataset if you load the fasttext model :)
    print('Loading dataset from {}'.format(args.dataset))
    with open(args.dataset, 'rb') as f:
        dataset = pickle.load(f)

    vocab = dataset.vocab
    max_len = dataset.max_len

    print('Loading trained model')
    trained = torch.load(args.modelfile)

    if args.modelfile == 'trained_models/trained_model':
        test_data = OrtLoader(smaort, tatort, vocab, max_len, dev, orter=None)
    elif args.modelfile == 'trained_models/fastText_trained_model':
        with open('sammansattningar.pkl', 'rb') as f:
            ss = pickle.load(f)
        test_data = OrtLoader(smaort, tatort, vocab, max_len, dev, orter=ss)

    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, num_workers=0)
    model_eval(trained, test_data)
