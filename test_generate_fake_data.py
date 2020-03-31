#!/usr/bin/env python

import os
import librosa
import torch
import numpy as np

from math import ceil
from model import NSFModel

sampling_rate = 16000
frame_length = sampling_rate * 25 // 1000
frame_shift = sampling_rate * 10 // 1000

batch_size = 2
waveform_length = 16000
context_length = ceil(waveform_length / sampling_rate / (10 / 1000))
input_dim = 81
output_dim = 1

statedict_path = 'model_epoch194.pth'

def generate_data():
    F0_file = os.path.expanduser('~/Desktop/arai_feats.txt')
    F0_dict = dict()
    with open(F0_file, 'r') as fp:
        for line in fp:
            cols = line.split()
            if cols[1] == '[':
                # new utterance
                current_label = cols[0]
                F0_dict[current_label] = list()
            elif len(cols) == 2:
                # add second element(F0) to current F0 list
                F0_dict[current_label].append(float(cols[1]))
            else:
                # make current F0 sequence numpy array
                F0_dict[current_label] = np.array(F0_dict[current_label])

    F0_segments = []
    for utt_label in sorted(F0_dict.keys()):
        F0 = F0_dict[utt_label]
        # align with the segment
        n_segments = int(ceil(F0.size / context_length))
        F0 = np.hstack(
            (F0, np.zeros(n_segments * context_length - F0.size))
        ).reshape((n_segments, context_length))
        F0_segments.append(F0)

    F0 = np.vstack(F0_segments)
    # align with the batch
    n_batches = ceil(F0.shape[0] / batch_size)
    F0 = np.vstack(
        (F0, np.zeros((n_batches * batch_size - F0.shape[0], context_length)))
    )
    c = np.zeros((F0.shape[0], context_length, input_dim-1))
    x = np.dstack((np.expand_dims(F0, -1), c))
    x = x.astype('float32')

    indices = np.arange(F0.shape[0])
    np.random.shuffle(indices)
    indices = indices.reshape(n_batches, batch_size)
    for idx in indices:
        yield torch.tensor(x[idx])

def main():
    model = NSFModel(input_dim, waveform_length)
    model.load_state_dict(torch.load(statedict_path))

    x = next(generate_data())
    y_pred = model(x).squeeze(-1).detach().numpy().reshape(batch_size*waveform_length)
    librosa.output.write_wav('arai_fake.wav', y_pred, sr=sampling_rate)

if __name__ == '__main__':
    main()
