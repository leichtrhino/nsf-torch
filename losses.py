import torch
from math import pi

def spectral_amplitude_distance(dft_bins, frame_length, frame_shift):
    def _distance(x, y):
        X = torch.stft(x, dft_bins, frame_shift, frame_length)
        Y = torch.stft(y, dft_bins, frame_shift, frame_length)
        return 0.5 * torch.sum(
            torch.log(
                torch.clamp(
                    X[:,:,0]**2 + X[:,:,1]**2,
                    min=torch.finfo(X.dtype).eps
                ) /
                torch.clamp(
                    Y[:,:,0]**2 + Y[:,:,1]**2,
                    min=torch.finfo(Y.dtype).eps
                )
            )**2
        )
    return _distance

def phase_distance(dft_bins, frame_length, frame_shift):
    def _distance(x, y):
        X = torch.stft(x, dft_bins, frame_shift, frame_length)
        Y = torch.stft(y, dft_bins, frame_shift, frame_length)
        return torch.sum(
            1 - (X[:,:,0]*Y[:,:,0] + X[:,:,1]*Y[:,:,1])
            / torch.sqrt(torch.clamp(
                ((X[:,:,0]**2+X[:,:,1]**2)*(Y[:,:,0]**2+Y[:,:,1]**2)),
                min=torch.finfo(X.dtype).eps
            ))
        )
    return _distance
