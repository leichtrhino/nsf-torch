import torch
from math import pi

def spectral_amplitude_distance(dft_bins, frame_length, frame_shift):
    def _distance(x, y):
        X = torch.stft(x, dft_bins, frame_shift, frame_length).permute(3, 0, 1, 2)
        Y = torch.stft(y, dft_bins, frame_shift, frame_length).permute(3, 0, 1, 2)
        eps = torch.finfo(X.dtype).eps
        return 0.5 * torch.sum(
            torch.log(
                torch.clamp(X[0]**2+X[1]**2, eps)
                / torch.clamp(Y[0]**2 + Y[1]**2, eps)
            )**2
        ) / x.shape[0]
    return _distance

def phase_distance(dft_bins, frame_length, frame_shift):
    def _distance(x, y):
        X = torch.stft(x, dft_bins, frame_shift, frame_length).permute(3, 0, 1, 2)
        Y = torch.stft(y, dft_bins, frame_shift, frame_length).permute(3, 0, 1, 2)
        eps = torch.finfo(X.dtype).eps
        return torch.sum(
            1 - (X[0]*Y[0] + X[1]*Y[1])
            / torch.sqrt(
                torch.clamp((X[0]**2+X[1]**2)*(Y[0]**2+Y[1]**2), eps)
            )
        ) / x.shape[0]
    return _distance
