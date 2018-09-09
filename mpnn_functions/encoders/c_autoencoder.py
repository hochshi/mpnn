from torch import nn
import math


class AutoEncoder(nn.Module):
    def __init__(self, in_features):
        super(AutoEncoder, self).__init__()
        self.in_f = in_features
        self.mid_f = int(math.ceil(float(in_features)/2))
        self.out_f = self.mid_f/2
        self.encoder = nn.Sequential(
            nn.Linear(self.in_f, self.mid_f, bias=False),
            nn.Tanh(),
            nn.Linear(self.mid_f, self.out_f)
        )
        self.decoder = nn.Sequential(
            nn.BatchNorm1d(self.out_f),
            nn.Linear(self.out_f, self.mid_f),
            nn.Tanh(),
            nn.Linear(self.mid_f, self.in_f),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
