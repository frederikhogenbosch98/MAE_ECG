import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.ReLU):
        super().__init__()
        self.c_hidden_layers = base_channel_size
        self.kernel_size = 3
        self.padding = 1

        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, self.c_hidden_layers, kernel_size = self.kernel_size, padding = self.padding, stride = self.stride),
            act_fn(),
            nn.Conv2d(self.c_hidden_layers, 2 * self.c_hidden_layers, kernel_size = self.kernel_size, padding = self.padding),
            act_fn(),
            nn.Flatten(),
            nn.Linear(2*16*self.c_hidden_layers, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.ReLU):
        super().__init__()
        self.c_hidden_layers = base_channel_size
        self.kernel_size = 3
        self.padding = 1

        self.net == nn.Sequential(
            nn.Linear(latent_dim, )
        )
