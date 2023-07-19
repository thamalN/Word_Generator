import torch
import torch.nn.functional as F

from init import words, stoi, itos, vocab_len


def bi_gram():
    N = torch.zeros((vocab_len, vocab_len), dtype=torch.int32)

    for word in words:
        chs = ['.'] + list(word) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            i = stoi[ch1]
            j = stoi[ch2]
            N[i, j] += 1

    P = (N + 1).float()  # smoothing to remove 0 probabilities / higher the numer -> more uniform distribution
    P /= P.sum(1, keepdims=True)

    # P represents the probabilities of a character occurring after another character
    # P grows exponentially if the bi-gram model is extended.

    loss = 0.0
    num = 0

    for word in words:
        chs = ['.'] + list(word) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            i = stoi[ch1]
            j = stoi[ch2]
            loss += -torch.log(P[i, j])
            num += 1

    print(f'bi_gram loss = {loss / num}')

    generator = torch.Generator().manual_seed(2147483647)

    for i in range(5):
        out = []
        ix = 0
        while True:
            p = P[ix]
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=generator).item()
            out.append(itos[ix])
            if ix == 0:
                break
        print(''.join(out))


def neural_net():
    xs, ys = [], []

    for word in words:
        chs = ['.'] + list(word) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            i = stoi[ch1]
            j = stoi[ch2]
            xs.append(i)
            ys.append(j)

    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    nums = xs.nelement()

    generator = torch.Generator().manual_seed(2147483647)

    xenc = F.one_hot(xs, num_classes=27).float()
    W = torch.randn((27, 27), generator=generator, requires_grad=True)

    loss = 0.0

    for i in range(100):
        logits = xenc @ W  # torch.matmul
        probs = torch.softmax(logits, dim=1)  # exp() for positive values and normalize to get prob = 1

        # maximum likelihood estimation = prob[0][1] * prob[0][2] * ... * prob[0][27]
        # high probability -> good prediction
        # log(mle) to increase lower values
        # -log(mle) to get positive values = negative log likelihood (loss)
        # take average nll for one loss value
        # low nll -> good prediction

        # when W values are equal to each other (0), probabilities are more uniform.
        loss = -probs[torch.arange(nums), ys].log().mean() + 0.01 * (W ** 2).mean()

        # backpropagation
        W.grad = None
        loss.backward()

        # update weight matrix (gradient descent)
        with torch.no_grad():
            W += -50 * W.grad

        # if i % 20 == 0:
        #     print(f'iteration = {i} -> loss = {loss}')

    print(f'neural_net loss = {loss}')

    generator = torch.Generator().manual_seed(2147483647)

    for i in range(5):
        out = []
        ix = 0
        while True:
            xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
            logits = xenc @ W
            p = torch.softmax(logits, dim=1)
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=generator).item()
            out.append(itos[ix])
            if ix == 0:
                break
        print(''.join(out))
