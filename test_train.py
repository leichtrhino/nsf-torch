#!/usr/bin/env python

import torch
from model import NSFModel
from losses import spectral_amplitude_distance
from losses import phase_distance

def main():
    batch_size = 1
    waveform_length = 1600
    context_length = 32
    input_dim = 81
    output_dim = 1

    x = torch.randn(batch_size, context_length, input_dim)
    y = torch.randn(batch_size, waveform_length)

    model = NSFModel(input_dim, waveform_length)
    Ls = spectral_amplitude_distance(512, 320, 80)
    Lp = phase_distance(512, 320, 80)
    criterion = lambda y_pred, y: Ls(y_pred, y) + Lp(y_pred, y)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    for t in range(101):
        y_pred = model(x, y) # give y to estimate the initial phase

        # Compute and print loss
        loss = criterion(y_pred, y)
        if t % 10 == 0:
            print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    main()
