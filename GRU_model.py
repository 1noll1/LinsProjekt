import torch
from torch import optim
from torch import nn
criterion = nn.NLLLoss()

class GRUclassifier(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, output_size, dev, pretrained=False):
        super().__init__()
        self.dev = dev
        self.num_layers = 1
        self.hidden_size = hidden_size
        if pretrained == True:
            weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
            nn.Embedding.from_pretrained(weight)
        if pretrained == False:
            self.embed = nn.Embedding(vocab_size, input_size)
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.linear = nn.Linear(input_size * hidden_size, output_size) # 100 is the sequence length
        self.logsoftmax = nn.LogSoftmax(dim=1) # log softmax is needed for NLLL

    def forward(self, x):
        output = self.embed(x)

        #h = self.init_hidden(len(x[0])) #sequence length is at i=1
        
        h = self.init_hidden(len(x))
        
        output, hidden = self.gru(output, h)
        # print('output after:', output.shape)
        output = output.contiguous().view(-1, self.hidden_size * len(x[0])) # -1 just infers the size
        # print(output.shape)
        output = self.linear(output) # squish it! This is a fully connected layer!
        # print(output.shape)
        return self.logsoftmax(output)
    
    def set_dev(self, dev):
        self.dev = dev
    
    def init_hidden(self, x_len):
        return torch.zeros(self.num_layers, x_len, self.hidden_size).to(self.dev)
