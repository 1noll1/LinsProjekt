import torch
from transformers import AutoTokenizer

class OrtLoader():
    def __init__(self, smaort, tatort, vocab, max_len, dev, orter=None):

        tok = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased')

        if orter is None:
            total = smaort + tatort
        if orter is not None:
            total = orter.values()

        self.dev = dev
        self.max_len = max_len
        self.vocab = vocab
        self.char2int = {t: n for n, t in enumerate(vocab)}

        self.len = len(total)


        def ort2int(all_placenames):
            encoded_placenames = []
            for i, ort in enumerate(all_placenames):
                try:
                    intort = tok.encode(ort)
                    # intort = [self.char2int[t] for t in ort]
                    encoded_placenames.append(intort)
                except:
                    print('Encoding of placename {} failed'.format(ort))
            return encoded_placenames

        encoded = ort2int(total)

        self.X_tensors = [torch.LongTensor(x) for x in encoded]

        self.input_size = max_len

        if orter is None:
            self.y_tensors = torch.Tensor([0 if ort in smaort else 1 for ort in total])
        if orter is not None:
            self.y_tensors = torch.Tensor([0 if ort in smaort else 1 for ort in orter.keys()])

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        return self.X_tensors[i].to(self.dev), self.y_tensors[i].to(self.dev)
