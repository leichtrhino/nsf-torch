#!/usr/bin/env python

import os
import sys
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

batch_size = 8
waveform_length = 16000
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

    '''
    obtained by:
    compute-kaldi-pitch-feats --sample-frequency=16000\
        scp:wav.scp ark,t:feats.txt
    contents of wav.scp:
    utt_001 <full path to utt_001.wav>
    utt_002 <full path to utt_002.wav>
    utt_003 <full path to utt_003.wav>
    ...
    '''
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
    for utt_label in sorted(set(wav_dict.keys()) & set(F0_dict.keys())):
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
    model.load_state_dict(torch.load('model_epoch99.pth'))
    loss_functions = []
    for dft_bins, frame_length, frame_shift in [
        (512, 320, 80), # for Ls1 and Lp1
        (128, 80, 40), # for Ls2 and Lp2
        (2048, 1920, 640), # for Ls3 and Lp3
    ]:
        Ls = spectral_amplitude_distance(dft_bins, frame_length, frame_shift)
        Lp = phase_distance(dft_bins, frame_length, frame_shift)
        loss_functions.extend([Ls, Lp])
    criterion = lambda y_pred, y: sum(L(y_pred, y) for L in loss_functions)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    #with autograd.detect_anomaly():
    for epoch in range(100, 200):
        sum_loss = 0
        last_output_len = 0
        for step, (x, y) in enumerate(generate_data()):
            # TODO: mark deprecated
            model.source_module.sine_generator.natural_waveforms = y
            y_pred = model(x)

            # Compute loss
            loss = criterion(y_pred, y)
            sum_loss += loss.item() * batch_size
            ave_loss = sum_loss / (batch_size*(step+1))

            # Zero gradients, perform a backward pass, and update the weights.
            loss.backward()
            optimizer.step()
            sum_grad = sum(
                torch.sum(torch.abs(p.grad))
                for p in model.parameters() if p.grad is not None
            )
            optimizer.zero_grad()

            # Print learning statistics
            curr_output =\
                f'\repoch {epoch} step {step} loss={ave_loss} grad={sum_grad}'
            sys.stdout.write('\r' + ' ' * last_output_len)
            sys.stdout.write(curr_output)
            sys.stdout.flush()
            last_output_len = len(curr_output)

        curr_output =\
            f'\repoch {epoch} loss={ave_loss}'
        sys.stdout.write('\r' + ' ' * last_output_len)
        sys.stdout.write(f'\repoch {epoch} loss={ave_loss}\n')

        torch.save(
            model.state_dict(),
            f'model_epoch{epoch}.pth'
        )

if __name__ == '__main__':
    main()
