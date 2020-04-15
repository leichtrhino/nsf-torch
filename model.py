import torch

from modules import ConditionModule
from modules import SourceModule
from modules import NeuralFilterModule

# input: NxBxn_features
# output: NxT
class NSFModel(torch.nn.Module):
    def __init__(self, input_size, waveform_length):
        super(NSFModel, self).__init__()
        self.input_size = input_size
        self.waveform_length = waveform_length
        self.condition_module = ConditionModule(input_size, 64)
        self.source_module = SourceModule(waveform_length)
        self.neural_filter_modules = torch.nn.ModuleList(
            NeuralFilterModule()
            for _ in range(5)
        )

    def forward(self, x, y=None):
        F0 = x[:, :, 0]
        out_condition = self.condition_module(x)
        out_condition = torch.nn.functional.interpolate(
            out_condition.transpose(1, 2), self.waveform_length
        ).transpose(1, 2)
        out_source = self.source_module(F0, y)
        out_filter = out_source
        for m in self.neural_filter_modules:
            out_filter = m(out_filter, out_condition)
        return out_filter

