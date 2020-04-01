# nsf-torch

This is an unofficial implementation of [neural source-filter model][^1] proposed by Wang et al.
The model takes sequences of fundamental frequency also called F0 and sequences of context vectors which appears in the WaveNet model.

### Requirments

* PyTorch
* LibROSA (optional: if you run train.py)
* Kaldi (optional: if you train.py and the test script for evaluation)

### Usage

This section describes how to train the model and generate waveforms by the model.
I also describe the usage of two test scrips named `test_train_real_data.py` and `test_generate_fake_data.py` (TBA).

#### Training

The model (`NSFModel`) and two loss functions (`spectral_amplitude_distance` and `phase_distance`) are defined in `model.py` and `losses.py` respectively.
The following piece of code is a minimal (but meaningless) example of a training procedure.

```python
import torch

from model import NSFModel
from losses import spectral_amplitude_distance, phase_distance

sampling_rate = 16000
waveform_length = 16000
context_length = 80
input_dim = 81 # F0 and some acoustic features (e.g. MFC)
output_dim = 1
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

# Data generator
def generat_data():
    for batch in range(16):
        yield torch.Tensor(), torch.Tensor() # TODO

for epoch in range(100):
    for step, (x, y) in enumerate(generate_data()):
        y = y.squeeze(-1)
        model.source_module.sine_generator.natural_waveforms = y
        y_pred = model(x).squeeze(-1)

        # Compute loss
        loss = criterion(y_pred, y)

        # Update weight
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

#### Generating waveforms

#### Using `test_train_real_data.py` and `test_generate_fake_data.py`

TBA

### TODO

* documentation
* make it a library (if it is combinient)

### References

[^1]: https://nii-yamagishilab.github.io/samples-nsf/
