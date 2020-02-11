import torch
from torch import nn
from torch import optim

def trained_batches(model, num_epochs, dev, train_loader, loss_mode=1):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()
    model = model.to(dev)
    model.set_dev(dev)

    for epoch in range(1, num_epochs+1):
        losses = []
        print('starting epoch...')
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i == len(train_loader)-1:
                print('Average loss at epoch {}: {}'.format(epoch, sum(losses)/i))
    print('Training complete.')
    return model
