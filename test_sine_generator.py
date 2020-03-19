#!/usr/bin/env python
import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

from layers import SineGenerator

def main():
    # unsqueeze to make 1x16x1 tensor
    x = torch.cat(
        (torch.zeros(6), 220 * torch.ones(5), 440 * torch.ones(5))
    ).unsqueeze(-1).unsqueeze(0)
    siggen = SineGenerator(160000)
    y = siggen(x)

    #y = np.array(torch.sum(y, -1)[0]) # merge harmonics and squeeze
    #D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    #librosa.display.specshow(D, y_axis='linear')
    #plt.show()
    #librosa.output.write_wav('hoge.wav', y, 16000)

if __name__ == '__main__':
    main()
