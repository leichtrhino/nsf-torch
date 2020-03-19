import torch
from math import pi

# input shape: NxBx1
# output shape: NxTx(1+harmonics)
class SineGenerator(torch.nn.Module):
    def __init__(self, waveform_length):
        super(SineGenerator, self).__init__()
        self.waveform_length = waveform_length # T
        self.sr = 16000 # sampling rate
        self.F0_mag = 0.1 # magnitude of the sine waves
        self.noise_mag = 0.003 # magnitude of noise
        self.bins = 64 # the number of random initial phase NOTE: not impl. yet
        self.harmonics = 7 # number of harmonics

    def forward(self, x):
        # interpolate x (from NxBx1 to NxTx1)
        f = torch.nn.functional.interpolate(
            x.transpose(1, 2), self.waveform_length
        ).transpose(1, 2)
        h = torch.arange(1, self.harmonics + 2)
        phi = 2 * pi * torch.rand(1) - pi
        n = torch.normal(
            0, self.noise_mag**2, (self.waveform_length,)
        ).unsqueeze(-1)
        voiced = self.F0_mag * torch.sin(
            h * 2 * pi * torch.cumsum(f, 1) / self.sr + phi
        ) + n
        unvoiced = 1. / (3 * self.noise_mag) * n
        return torch.where(f > 0, voiced, unvoiced)

def main():
    import numpy as np
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display
    singen = SineGenerator(160000)
    linear = torch.nn.Linear(8, 1)
    # unsqueeze to make 1x16x1 tensor
    x = torch.cat(
        (torch.zeros(6), 220 * torch.ones(5), 440 * torch.ones(5))
    ).unsqueeze(-1).unsqueeze(0)
    y = linear(singen(x))
    print(x.shape, y.shape)
    '''
    y = np.array(torch.sum(y, -1)[0]) # merge harmonics and squeeze
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, y_axis='linear')
    print(D.shape)
    plt.show()
    librosa.output.write_wav('hoge.wav', y, 16000)
    '''

if __name__ == '__main__':
    main()
