#!/usr/bin/env python
import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

from layers import SineGenerator
from modules import SourceModule
from losses import spectral_amplitude_distance, phase_distance

def main():
    # unsqueeze to make 1x16x1 tensor
    waveform_length = 32000
    x = torch.cat((440 * torch.ones(8000),)).unsqueeze(-1).unsqueeze(0)
    coeff, bias = torch.randn(8), torch.randn(1)
    y = torch.sum(coeff * SineGenerator(waveform_length)(x), -1) + bias

    model = SourceModule(waveform_length)
    Ls = spectral_amplitude_distance(512, 320, 80)
    Lp = phase_distance(512, 320, 80)
    criterion = lambda y_pred, y: Ls(y_pred, y) + Lp(y_pred, y)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    for t in range(501):
        model.sine_generator.natural_waveforms = y
        y_pred = model(x)

        # Compute and print loss
        loss = criterion(y_pred, y)
        if t % 100 == 0:
            print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('true parameters:', coeff, bias)
    print('predicted parameters:', model.linear.weight, model.linear.bias)

if __name__ == '__main__':
    main()
