import torch

from layers import SineGenerator

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

class CausalBlock(torch.nn.Module):
    pass

class DiluteBlock(torch.nn.Module):
    pass

class PostProcessingBlock(torch.nn.Module):
    pass

# Causal + Dilute1 + ... + DiluteN + PostProcessing
class NeuralFilterModule(torch.nn.Module):
    pass

def main():
    batch_size = 8
    context_length = 80
    feature_dim = 81
    output_dim = 64

    x = torch.randn(batch_size, context_length, feature_dim)
    y = torch.randn(batch_size, context_length, output_dim)

    module = ConditionModule(feature_dim, output_dim)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(module.parameters(), lr=1e-4)

    for t in range(501):
        y_pred = module(x)

        # Compute and print loss
        loss = criterion(y_pred, y)
        if t % 100 == 0:
            print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    main()
