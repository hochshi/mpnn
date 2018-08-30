from torch import nn


class AtomAutoEncoder(nn.Module):
    def __init__(self):
        super(AtomAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(30, 15, bias=False),
            nn.Tanh(),
            nn.Linear(15, 8)
        )
        self.decoder = nn.Sequential(
            nn.BatchNorm1d(8),
            nn.Linear(8, 15),
            nn.Tanh(),
            nn.Linear(15, 30),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))