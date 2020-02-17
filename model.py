import torch

class CausalBlock(torch.nn.Module):
    pass

class DiluteBlock(torch.nn.Module):
    pass

class PostProcessingBlock(torch.nn.Module):
    pass

class ConditionModule(torch.nn.Module):
    pass

class SourceModule(torch.nn.Module):
    pass

# Causal + Dilute1 + ... + DiluteN + PostProcessing
class NeuralFilterModule(torch.nn.Module):
    pass

class NSFModel(torch.nn.Module):
    pass
