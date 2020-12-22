import torch
from torch import nn
from transformers import AutoModel
bert_model = AutoModel.from_pretrained('KB/bert-base-swedish-cased')


class GRUclassifier(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, output_size, dev, pretrained):
        super().__init__()
        self.dev = dev
        self.num_layers = 1
        self.hidden_size = hidden_size
        if pretrained is not None:
            print('Using pretrained FastText weights')
            weight = pretrained
            self.embed = nn.Embedding.from_pretrained(weight)
        if pretrained is None:
            print('Using PyTorch embeddings layer')
            self.embed = nn.Embedding(vocab_size, input_size)
        self.gru = nn.GRU(768, hidden_size, num_layers=1, batch_first=True)
        self.output_size = output_size
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output, _ = bert_model(x)
        # output = self.embed(x)
        h = self.init_hidden(len(x))
        output, hidden = self.gru(output, h)
        output = output.contiguous().view(-1, self.hidden_size * len(x[0]))  # -1 just infers the size
        linear_layer = nn.Linear(len(x[0]) * self.hidden_size, self.output_size)
        output = linear_layer(output)
        return self.sigmoid(output)

    def set_dev(self, dev):
        self.dev = dev

    def init_hidden(self, x_len):
        return torch.zeros(self.num_layers, x_len, self.hidden_size).to(self.dev)
