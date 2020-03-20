import torch

from layers import SineGenerator
from layers import WaveNetCore

class ConditionModule(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(ConditionModule, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.bilstm = torch.nn.LSTM(
            input_size=input_size, hidden_size=32,
            batch_first=True, bidirectional=True
        )
        self.cnn = torch.nn.Conv1d(
            in_channels=64, out_channels=output_size-1, kernel_size=1
        )

    def forward(self, x):
        F0 = x[:, :, 0].unsqueeze(dim=-1)
        x, _ = self.bilstm(x) # NOTE: this ignores hidden states
        x = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
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
        x = self.linear(x)
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
            input_size, output_size // 2, 2,
            dilation=dilation, padding=dilation, bias=True
        )
        self.wavenet_core = WaveNetCore(context_size, output_size // 4)
        self.linear1 = torch.nn.Linear(output_size // 4, output_size // 2)
        self.linear2 = torch.nn.Linear(output_size // 2, output_size)
    def forward(self, x, c):
        # x: output of the previous dilute block
        # c: context vector for wavenetcore
        x_in_tmp = x
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)[:, :-self.dilation, :]
        x = self.wavenet_core(torch.cat((c, x), dim=-1))
        x = x_out_tmp = self.linear1(x)
        x += x_in_tmp
        x = self.linear2(x)
        return x, x_out_tmp

# Causal + Dilute1 + ... + DiluteN + PostProcessing
class NeuralFilterModule(torch.nn.Module):
    def __init__(self):
        super(NeuralFilterModule, self).__init__()
        self.context_size = 64
        self.dilute_input_size = 64
        self.dilute_output_size = 128
        self.causal_linear = torch.nn.Linear(1, self.dilute_input_size)
        self.dilute_blocks = [
            DiluteBlock(
                self.dilute_input_size,
                self.dilute_output_size,
                self.context_size, 2**i
            )
            for i in range(10)
        ]
        self.postoutput_linear1 = torch.nn.Linear(self.dilute_output_size, 16)
        self.postoutput_linear2 = torch.nn.Linear(16, 2)

    def forward(self, x, c):
        # x: signal tensor from previous module
        # c: context tensor
        x_in = x
        x = self.causal_linear(x)
        outputs_from_blocks = []
        ysum = None
        for blk in self.dilute_blocks:
            y, x = blk(x, c)
            if ysum is None:
                ysum = y
            else:
                ysum += y
        x = ysum
        x = self.postoutput_linear1(x)
        x = self.postoutput_linear2(x)
        return x_in * torch.exp(x[:, :, 1].unsqueeze(-1)) + x[:, :, 0].unsqueeze(-1)

def main():
    batch_size = 1
    waveform_length = 1600
    context_dim = 64
    input_dim = 1
    output_dim = 1

    context_vec = torch.randn(batch_size, waveform_length, context_dim)
    x = torch.randn(batch_size, waveform_length, input_dim)
    y = torch.randn(batch_size, waveform_length, output_dim)

    module = NeuralFilterModule()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(module.parameters(), lr=1e-4)

    for t in range(501):
        y_pred = module(x, context_vec)

        # Compute and print loss
        loss = criterion(y_pred, y)
        if t % 100 == 0:
            print(t, loss.item(), x.shape)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    main()
