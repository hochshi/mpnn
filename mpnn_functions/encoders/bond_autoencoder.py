from torch import nn


class BondAutoEncoder(nn.Module):
    def __init__(self):
        super(BondAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(8, 4, bias=False),
            nn.Tanh(),
            nn.Linear(4, 2)
        )
        self.decoder = nn.Sequential(
            nn.BatchNorm1d(2),
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 8),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))