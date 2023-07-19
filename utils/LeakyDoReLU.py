import torch

class LeakyDoReLU(torch.nn.Module):
    def __init__(self, gamma):
        super(LeakyDoReLU, self).__init__()
        self.gamma = gamma

    def forward(self, x):
        y = torch.where(x > 1, 1 + (x - 1) / self.gamma, x)
        y = torch.where((x >= 0) & (x <= 1), x, y)
        y = torch.where(x < 0, x / self.gamma, y)
        return y
