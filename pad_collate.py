from torch.nn.utils.rnn import pad_sequence


def pad_collate(batch):
    (xx, yy) = zip(*batch)

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)

    return xx_pad, yy
