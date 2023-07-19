import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

words = open("names.txt", "r").read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

vocab_len = len(stoi)
