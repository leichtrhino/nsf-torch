import torch

from layers import SineGenerator
from layers import WaveNetCore

class ConditionModule(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(ConditionModule, self).__init__()
        hidden_size = 64
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.bilstm = torch.nn.LSTM(
            input_size=input_size, hidden_size=hidden_size // 2,
            batch_first=True, bidirectional=True
        )
        self.cnn = torch.nn.Conv1d(
            in_channels=hidden_size, out_channels=output_size-1, kernel_size=1
        )
        self.bilstm_hidden = None

    def forward(self, x):
        F0 = x[:, :, 0].unsqueeze(dim=-1)
        x, _ = self.bilstm(x)
        x = torch.tanh(self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1))
        return torch.cat((F0, x), dim=-1)

# input: NxBx1
# output: NxTx1
class SourceModule(torch.nn.Module):
    def __init__(self, waveform_length):
        super(SourceModule, self).__init__()
        self.sine_generator = SineGenerator(waveform_length)
        self.linear = torch.nn.Linear(8, 1)
    def forward(self, x):
        x = self.sine_generator(x)
        x = torch.tanh(self.linear(x))
        return torch.squeeze(x, -1)

# input: NxTxinput_size, NxTxcontext_size
# output: NxTxoutput_size (for post-output block),
#         NxTxoutput_size/2 (for next diluteblock)
class DiluteBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, context_size, dilation):
        super(DiluteBlock, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.dilation = dilation
        # padding=dilation makes the cnn causal (on kernel_size=2)
        self.cnn = torch.nn.Conv1d(
            input_size, output_size // 2, 3,
            dilation=dilation, padding=2*dilation, bias=True
        )
        self.wavenet_core = WaveNetCore(context_size, output_size // 4)
        self.linear1 = torch.nn.Linear(output_size // 4, output_size // 2)
        self.linear2 = torch.nn.Linear(output_size // 2, output_size)
    def forward(self, x, c):
        # x: output of the previous dilute block
        # c: context vector for wavenetcore
        x_in_tmp = x
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)[:, :-2*self.dilation, :]
        x = self.wavenet_core(x, c)
        x = torch.tanh(self.linear1(x))
        x = x + x_in_tmp
        return x, torch.tanh(self.linear2(x))

# Causal + Dilute1 + ... + DiluteN + PostProcessing
class NeuralFilterModule(torch.nn.Module):
    def __init__(self):
        super(NeuralFilterModule, self).__init__()
        self.context_size = 64
        self.dilute_input_size = 64
        self.dilute_output_size = 128
        self.causal_linear = torch.nn.Linear(1, self.dilute_input_size)
        self.dilute_blocks = torch.nn.ModuleList(
            DiluteBlock(
                self.dilute_input_size,
                self.dilute_output_size,
                self.context_size, 2**i
            )
            for i in range(10)
        )
        self.postoutput_linear1 = torch.nn.Linear(self.dilute_output_size, 16)
        self.postoutput_batchnorm1 = torch.nn.BatchNorm1d(16)
        self.postoutput_linear2 = torch.nn.Linear(16, 2)
        self.postoutput_batchnorm2 = torch.nn.BatchNorm1d(2)

    def forward(self, x, c):
        # x: signal tensor from previous module
        # c: context tensor
        x_in = x
        x = torch.tanh(self.causal_linear(x))
        outputs_from_blocks = []
        ysum = 0
        for blk in self.dilute_blocks:
            x, y = blk(x, c)
            ysum = y + ysum
        x = 0.01 * ysum
        x = torch.tanh(
            self.postoutput_batchnorm1(
                self.postoutput_linear1(x).transpose(1, 2)
            ).transpose(1, 2)
        )
        x = torch.tanh(
            self.postoutput_batchnorm2(
                self.postoutput_linear2(x).transpose(1, 2)
            ).transpose(1, 2)
        )
        return x_in * torch.exp(x[:, :, 1].unsqueeze(-1)) + x[:, :, 0].unsqueeze(-1)
