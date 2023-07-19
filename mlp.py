import torch
import torch.nn.functional as F
from init import words, stoi, itos, vocab_len, device

block_size = 3
X, Y = [], []

for word in words:
    # print(word)
    context = [0] * block_size
    for ch in word + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        # print(''.join(itos[i] for i in context), '->', itos[ix])
        context = context[1:] + [ix]

X = torch.tensor(X)
Y = torch.tensor(Y)

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((vocab_len, 2), generator=g)
W1 = torch.randn((6, 100), generator=g)
b1 = torch.rand(100, generator=g)
W2 = torch.randn((100, 27), generator=g)
b2 = torch.rand(27, generator=g)
parameters = [C, W1, b1, W2, b2]

for p in parameters:
    p.requires_grad = True

# cross entropy = softmax -> negative log likelihood

# probs = torch.softmax(logits, dim=1)
# loss = -probs[torch.arange(X.shape[0]), Y].log().mean()

loss = 0.0

for i in range(2000):
    ix = torch.randint(0, X.shape[0], (100,))  # mini batch size = 32

    embed = C[X[ix]]
    h = torch.tanh(embed.view(-1, 6) @ W1 + b1)  # (32, 2, 3) -> (32, 6) x (6, 100)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y[ix])

    if i % 200 == 0:
        print(f'loss = {loss.item()}')

    for p in parameters:
        p.grad = None

    loss.backward()

    with torch.no_grad():
        for p in parameters:
            p += -0.1 * p.grad

print(f'loss = {loss.item()}')


for i in range(10):
    out = []
    context = [0] * block_size

    while True:
        embed = C[torch.tensor([context])]
        h = torch.tanh(embed.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        p = torch.softmax(logits, dim=1)
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        context = context[1:] + [ix]
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))
