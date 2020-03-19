#!/usr/bin/env python
import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

from layers import SineGenerator

def main():
    # unsqueeze to make 1x16x1 tensor
    x = torch.cat((440 * torch.ones(8000),)).unsqueeze(-1).unsqueeze(0)
    y, _ = librosa.core.load('名称未設定.wav', sr=16000)
    waveform_length = y.size
    y = torch.from_numpy(y).unsqueeze(-1).unsqueeze(0)

    model = torch.nn.Sequential(
        SineGenerator(waveform_length),
        torch.nn.Linear(8, 1)
    )
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    for t in range(1):
        y_pred = model(x)

        D = librosa.amplitude_to_db(np.abs(librosa.stft(y_pred.detach().numpy()[0, :, 0])), ref=np.max)
        librosa.display.specshow(D, y_axis='linear')
        plt.show()

        # Compute and print loss
        loss = criterion(y_pred, y)
        if t % 100 == 99:
            print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    main()
