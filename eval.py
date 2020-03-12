from torch.nn.utils.rnn import pad_sequence
import argparse
import torch
from torch.utils.data import Dataset, DataLoader 
import pickle
import numpy as np
from OrtLoader import OrtLoader

dev = torch.device("cuda:{}".format(hash('gusstrlip') % 4) if torch.cuda.is_available() else "cpu")

# class OrtLoader():
#     def __init__(self, smaort, tatort, vocab, max_len, sammansattningar=None):
#         self.max_len = max_len
#         self.vocab = vocab
#         char2int = {t:n for n, t in enumerate(vocab)}
#         if sammansattningar == None:
#             total = smaort + tatort
#         if sammansattningar != None:
#             total = sammansattningar
#         self.len = len(total)

#         def ort2int(total):
#             encoded = []
#             for i, ort in enumerate(total):
#                 try:
#                     intort = [char2int[t] for t in ort]
#                     encoded.append(torch.Tensor(intort))
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
    print('Total accuracy: {} {}'.format((a/len(true)) * 100, '%'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate place name classifier.')
    parser.add_argument('--dataset', type=str, default="default_dataset",
                    help='The name of the file containing the dataset.')
    parser.add_argument('--modelfile', type=str, default="trained_model",
                    help='The name of the file containing the model to be evaluated.')

    args = parser.parse_args()

    print('Loading dataset from {}'.format(args.dataset))
    with open('smaort_test.pkl', 'rb') as f:
        smaort = pickle.load(f)
    with open('tatort_test.pkl', 'rb') as f:
        tatort = pickle.load(f)
    with open(args.dataset, 'rb') as f:
        dataset = pickle.load(f)

    vocab = dataset.vocab
    max_len = dataset.max_len

    print('Loading trained model')
    trained = torch.load(args.modelfile)
    
    if args.modelfile == 'trained_model':
        test_data = OrtLoader(smaort, tatort, vocab, max_len, None)
    elif args.modelfile == 'fastText_trained_model':
        with open('sammansattningar.pkl', 'rb') as f:
            ss = pickle.load(f)
        test_data = OrtLoader(smaort, tatort, vocab, max_len, ss.values())

    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, num_workers=0)
    model_eval(trained, test_data)
