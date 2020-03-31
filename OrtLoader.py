import numpy as np
import torch

class OrtLoader():
    def __init__(self, smaort, tatort, vocab, max_len, dev, orter=None):
        if orter == None:
            total = smaort + tatort
        if orter != None:
            total = orter.keys()

        if orter == None:
            total = smaort + tatort
        if orter != None:
            total = orter.values()

        self.dev = dev
        self.max_len = max_len
        self.vocab = vocab
        self.char2int = {t:n for n, t in enumerate(vocab)}

        self.len = len(total)

        def ort2int(total):   
            encoded = []         
            for i, ort in enumerate(total):
                try:
                    intort = [self.char2int[t] for t in ort]
                    encoded.append(intort)
                except:
                    pass
            return encoded

        encoded = ort2int(total)

        print('Padding vectors up to max len: {}'.format(max_len))
        x_padded = []
        for ort in encoded:
            seq = np.zeros(self.max_len)
            i = 0
            for _int in ort:
               seq[i] = _int
               i += 1
               x_padded.append(np.copy(seq))

        self.input_size = len(x_padded[0])

        self.X_tensors = [torch.LongTensor(seq) for seq in x_padded]
        self.y_tensors = torch.FloatTensor([0 if ort in smaort else 1 for ort in total])

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        return self.X_tensors[i].to(self.dev), self.y_tensors[i].to(self.dev)