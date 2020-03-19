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
        self.natural_waveforms = None

    def forward(self, x):
        # interpolate x (from NxBx1 to NxTx1)
        f = torch.nn.functional.interpolate(
            x.transpose(1, 2), self.waveform_length
        ).transpose(1, 2)
        h = torch.arange(1, self.harmonics + 2)
        n = torch.normal(
            0, self.noise_mag**2, (self.waveform_length,)
        ).unsqueeze(-1)

        if self.natural_waveforms is not None:
            # generate candidates of initial phase
            phis = torch.linspace(-pi, pi, self.bins)
            # calculate the cross correlation for each initial phase
            voiced = self.F0_mag * torch.sin(
                2 * pi * torch.cumsum(f, 1) / self.sr + phis
            ) + n
            unvoiced = 1. / (3 * self.noise_mag) * n
            signals = torch.where(f > 0, voiced, unvoiced)
            phi_idx = torch.argmax(torch.sum(
                self.natural_waveforms.unsqueeze(-1) * signals, 1
            ), 1)
            phi = phis[phi_idx]
        else:
            phi = (torch.rand(x.size(0)) - 0.5) * 2 * pi
        voiced = self.F0_mag * torch.sin(
            h * 2 * pi * torch.cumsum(f, 1) / self.sr + phi
        ) + n
        unvoiced = 1. / (3 * self.noise_mag) * n
        return torch.where(f > 0, voiced, unvoiced)

# input: NxTx(context_dim+input_dim)
# output: NxTxoutput_dim
class WaveNetCore(torch.nn.Module):
    def __init__(self, context_dim, output_dim):
        super(WaveNetCore, self).__init__()
        self.context_dim = context_dim
        self.output_dim = output_dim
        if context_dim > 0:
            self.weight = torch.nn.Parameter(
                torch.randn(context_dim, 2 * output_dim)
            )
        else:
            self.weight = None

    def forward(self, x):
        if self.context_dim > 0:
            context = x[:, :, :self.context_dim]
            weight_context = torch.matmul(context, self.weight)
        else:
            weight_context = torch.zeros(1)
        inputs = x[:, :, self.context_dim:]
        h = inputs + torch.tanh(weight_context)
        h1 = h[:, :, :self.output_dim]
        h2 = h[:, :, self.output_dim:]
        return torch.tanh(h1) * torch.sigmoid(h2)
