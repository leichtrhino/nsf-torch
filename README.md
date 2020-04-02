# nsf-torch

This is an unofficial implementation of [neural source-filter model][^1] proposed by Wang et al.
The original implementation is [project-CURRENNT][^2].
The model takes sequences of fundamental frequency also called F0 and sequences of context vectors which appears in the WaveNet model.

### Requirments

* PyTorch
* LibROSA (optional: to load audio files)
* Kaldi (optional: to extract fundamental frequency and acoustic features)

### Usage

This section describes how to train the model and generate waveforms by the model.

#### Training

The model (`NSFModel`) and two loss functions (`spectral_amplitude_distance` and `phase_distance`) are defined in `model.py` and `losses.py` respectively.
The following piece of code is a minimal (but meaningless) example of a training procedure.

```python
import torch

from model import NSFModel
from losses import spectral_amplitude_distance, phase_distance

# Constants
sampling_rate = 16000
waveform_length = 16000 # per single sample
context_length = 80     # the number of context vectors per single sample
input_dim = 81          # F0 and some acoustic features (e.g. MFC)
batch_size = 8

# Initializing the model
model = NSFModel(input_dim, waveform_length)

# Definig the loss functions
dft_bins, frame_length, frame_shift = 512, 320, 80
Ls = spectral_amplitude_distance(dft_bins, frame_length, frame_shift)
Lp = phase_distance(dft_bins, frame_length, frame_shift)
criterion = lambda y_pred, y: Ls(y_pred, y) + Lp(y_pred, y)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Simple data generator
def generate_data():
    for batch in range(16):
        # preparing the F0 and context vectors
        # to extract F0 from real data, using `compute-kaldi-pitch-feats` might by helpful
        # to extract context vectors (e.g. MFCC), please refer to https://kaldi-asr.org/doc/feat.html
        # the followings are just showing the shapes of input tensors
        F0 = torch.Tensor(batch_size, context_length, 1)
        c = torch.Tensor(batch_size, context_length, input_dim - 1)
        # Note: in the original implementation, F0 comes last but this follows the paper
        x = torch.cat((F0, c), -1)
        # preparing the natural waveforms: the following is just showing the shape
        y = torch.Tensor(batch_size, waveform_length)
        yield x, y

# Training procedure
for epoch in range(100):
    for step, (x, y) in enumerate(generate_data()):
        # Setting the natural waveform to estimate the best initial phase
        model.source_module.sine_generator.natural_waveforms = y
        # Make a predicted waveform
        y_pred = model(x).squeeze(-1)
        # Compute loss
        loss = criterion(y_pred, y)
        # Update weight
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

#### Generating waveforms

Generating waveforms is done by giving a batch of sequences of fundamental frequency and lists of context vectors.

```
# Importing, defining some constants, loading models, prepare F0 and context vectors...
# x: (F0, context_vectors) in the previous subsection
y_pred = model(x)
```

### TODO

* documentation
* make it a library (if it is combinient)

[^1]: https://nii-yamagishilab.github.io/samples-nsf/
[^2]: https://github.com/nii-yamagishilab/project-CURRENNT-scripts
