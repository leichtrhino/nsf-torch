#!/usr/bin/env python
import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

from layers import WaveNetCore

def main():
    batch_size = 8
    waveform_length = 160
    context_dim = 64
    input_dim = 64
    output_dim = 32

    context_tensor = torch.randn(batch_size, waveform_length, context_dim)
    input_tensor = torch.randn(batch_size, waveform_length, input_dim)
    x = torch.cat((context_tensor, input_tensor), dim=-1)
    y = torch.randn(batch_size, waveform_length, output_dim)

    model = WaveNetCore(context_dim, output_dim)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    for t in range(501):
        y_pred = model(x)

        # Compute and print loss
        loss = criterion(y_pred, y)
        if t % 100 == 0:
            print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    main()
