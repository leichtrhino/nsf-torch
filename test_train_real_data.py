#!/usr/bin/env python

import os
import librosa
import torch

import numpy as np

from math import ceil
from model import NSFModel
from losses import spectral_amplitude_distance
from losses import phase_distance

sampling_rate = 16000
frame_length = sampling_rate * 25 // 1000
frame_shift = sampling_rate * 10 // 1000

batch_size = 16
waveform_length = 4000
context_length = ceil(waveform_length / sampling_rate / (10 / 1000))
input_dim = 81
output_dim = 1

def generate_data():
    wav_dir = os.path.expanduser('~/Desktop/arai_utt/')
    wav_list = [os.path.join(wav_dir, f) for f in os.listdir(wav_dir)]
    wav_dict = dict()

    for wav_file in wav_list:
        y, _ = librosa.core.load(wav_file, sr=sampling_rate)
        utt_label = os.path.splitext(os.path.split(wav_file)[1])[0]
        wav_dict[utt_label] = y

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
    wav_segments = []
    for utt_label in set(wav_dict.keys()).intersection(set(F0_dict.keys())):
        F0 = F0_dict[utt_label]
        y = wav_dict[utt_label][:(F0.size - 1) * frame_shift + frame_length]
        # align with the segment
        n_segments = int(ceil(y.size / waveform_length))
        F0 = np.hstack(
            (F0, np.zeros(n_segments * context_length - F0.size))
        ).reshape((n_segments, context_length))
        y = np.hstack(
            (y, np.zeros(n_segments * waveform_length - y.size))
        ).reshape((n_segments, waveform_length))
        F0_segments.append(F0)
        wav_segments.append(y)

    F0 = np.vstack(F0_segments)
    y = np.vstack(wav_segments)
    # align with the batch
    n_batches = ceil(F0.shape[0] / batch_size)
    F0 = np.vstack(
        (F0, np.zeros((n_batches * batch_size - F0.shape[0], context_length)))
    )
    y = np.vstack(
        (y, np.zeros((n_batches * batch_size - y.shape[0], waveform_length)))
    )

    c = np.zeros((F0.shape[0], context_length, input_dim-1))
    x = np.dstack((np.expand_dims(F0, -1), c))
    x = x.astype('float32')
    y = y.astype('float32')

    indices = np.arange(F0.shape[0])
    np.random.shuffle(indices)
    indices = indices.reshape(n_batches, batch_size)
    for idx in indices:
        yield torch.tensor(x[idx]), torch.tensor(y[idx])

def main():
    model = NSFModel(input_dim, waveform_length)
    Ls = spectral_amplitude_distance(512, 320, 80)
    Lp = phase_distance(512, 320, 80)
    criterion = lambda y_pred, y: Ls(y_pred, y) + Lp(y_pred, y)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    #with autograd.detect_anomaly():
    for epoch in range(100):
        for step, (x, y) in enumerate(generate_data()):
            # TODO: match dimension
            y = y.squeeze(-1)
            model.source_module.sine_generator.natural_waveforms = y
            y_pred = model(x).squeeze(-1)

            # Compute and print loss
            loss = criterion(y_pred, y)
            print('epoch', epoch, 'step', step, loss.item())
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            torch.save(
                model.state_dict(),
                f'model_epoch{epoch}_step{step}_loss{loss.item()}.pth'
            )

if __name__ == '__main__':
    main()
