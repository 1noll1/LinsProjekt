import numpy as np
import torch

class OrtLoader():
    #def __init__(self, smaort, tatort, vocab, max_len, dev):
    def __init__(self, smaort, tatort, vocab, max_len, dev, sammansattningar=None, orter=None):
        if sammansattningar == None:
            total = smaort + tatort
        if sammansattningar != None:
            total = sammansattningar

        self.dev = dev
        self.max_len = max_len
        self.vocab = vocab
        char2int = {t:n for n, t in enumerate(vocab)}
        #total = smaort + tatort
        self.len = len(total)

        def ort2int(total):   
            encoded = []         
            for i, ort in enumerate(total):
                try:
                    intort = [char2int[t] for t in ort]
                    encoded.append(intort)
                except:
                    pass
            return encoded

        encoded = ort2int(total)

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
        self.y_tensors = torch.LongTensor([0 if ort in smaort else 1 for ort in total])

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        return self.X_tensors[i].to(self.dev), self.y_tensors[i].to(self.dev)