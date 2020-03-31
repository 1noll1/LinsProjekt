import torch
from torch import optim
from torch import nn

class GRUclassifier(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, output_size, dev, pretrained):
        super().__init__()
        self.dev = dev
        self.num_layers = 1
        self.hidden_size = hidden_size
        if pretrained is not False:
            print('Using pretrained FastText weights')
            weight = pretrained
            self.embed = nn.Embedding.from_pretrained(weight)
        if pretrained is False:
            print('Using PyTorch embeddings layer')
            self.embed = nn.Embedding(vocab_size, input_size)
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.linear = nn.Linear(input_size * hidden_size, output_size) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = self.embed(x)
        h = self.init_hidden(len(x))     
        output, hidden = self.gru(output, h)
        output = output.contiguous().view(-1, self.hidden_size * len(x[0])) # -1 just infers the size
        output = self.linear(output) 
        #print(output.shape)
        return self.sigmoid(output)
        #return self.logsoftmax(output)
    
    def set_dev(self, dev):
        self.dev = dev
    
    def init_hidden(self, x_len):
        return torch.zeros(self.num_layers, x_len, self.hidden_size).to(self.dev)
