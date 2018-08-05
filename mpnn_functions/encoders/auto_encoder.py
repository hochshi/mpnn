from torch import nn

class Autoencoder(nn.Module):
    def __init__(self, in_dim=784, mid_dim=400, e_dim=20):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, mid_dim, bias=False),
            nn.Sigmoid(),
            nn.Linear(mid_dim, e_dim, bias=False),
            nn.Sigmoid()
            )

        self.decoder = nn.Sequential(
            nn.Linear(e_dim, mid_dim, bias=False),
            nn.Sigmoid(),
            nn.Linear(mid_dim, in_dim, bias=False),
            nn.Sigmoid()
            )

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out
