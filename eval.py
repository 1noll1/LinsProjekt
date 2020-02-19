from torch.nn.utils.rnn import pad_sequence
import argparse
import torch
from prefixloader import PrefixLoader
from torch.utils.data import Dataset, DataLoader 
import pickle

dev = torch.device("cuda:{}".format(hash('gusstrlip') % 4) if torch.cuda.is_available() else "cpu")

class OrtLoader():
    def __init__(self, smaort, tatort, vocab):
        # assign each set its class

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
                #print(ort)
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

def model_eval(model, test_loader):
    model.eval()
    true = []
    pred = []
    with torch.no_grad():
        for ort, label in test_loader:
                #print(ort.shape)
                #print('ort', ort, 'label', label)
                true.extend([label])
                #print(labels)
                #print(ort)
                ort = ort.unsqueeze(0)
                out = model(ort)
                _, predicted = torch.max(out.data, 1)
                predicted = predicted.to('cpu')
                pred.extend(predicted)

    a = 0
    for pred,label in zip(pred,true):
        if pred == label:
            a += 1
    
    print('Total accuracy: {} {}'.format((a/len(true)) * 100, '%'))


if __name__ == '__main__':
    with open('smaort_test.pkl', 'rb') as f:
        smaort = pickle.load(f)

    with open('tatort_test.pkl', 'rb') as f:
        tatort = pickle.load(f)

    with open('dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)

    v1 = dataset.vocab

    trained = torch.load('trained_batch32_epoch20')
    test_data = OrtLoader(smaort, tatort, v1)
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, num_workers=0)
    model_eval(trained, test_data)
