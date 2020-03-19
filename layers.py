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

