#!/usr/bin/env python

import librosa
import torch

from math import ceil
from model import NSFModel

sampling_rate = 16000
frame_length = sampling_rate * 25 // 1000
frame_shift = sampling_rate * 10 // 1000

batch_size = 4
waveform_length = 4000
context_length = ceil(waveform_length / sampling_rate / (10 / 1000))
input_dim = 81
output_dim = 1

statedict_path = 'model_epoch80.pth'

def main():
    model = NSFModel(input_dim, waveform_length)
    model.load_state_dict(torch.load(statedict_path))

    x = torch.cat((
        torch.cat((
            440 * torch.ones(context_length // 3 * batch_size, 1),
            torch.zeros(context_length // 3 * batch_size, 1),
            440 * torch.ones(context_length * batch_size - 2*(context_length // 3 * batch_size), 1),
        ), 0),
        torch.zeros(batch_size * context_length, input_dim-1)
    ), -1)
    x = x.reshape(batch_size, context_length, input_dim)
    y_pred = model.forward(x).squeeze(-1).detach().numpy().reshape(batch_size*waveform_length)
    print(y_pred.shape)
    librosa.output.write_wav('arai_fake.wav', y_pred, sr=sampling_rate)

if __name__ == '__main__':
    main()
