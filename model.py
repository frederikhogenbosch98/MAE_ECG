import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.net = nn.Sequential(
            # input shape: [N, 12, 4992]
            nn.Conv1d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=2), 
            nn.ReLU(), 
            nn.MaxPool1d(kernel_size=4, stride=4),  
            
            # intermediate shape: [N, 24, 1248]
            nn.Conv1d(in_channels=24, out_channels=48, kernel_size=5, stride=1, padding=2), 
            nn.ReLU(), 
            nn.MaxPool1d(kernel_size=4, stride=4), 
            
            # intermediate shape: [N, 48, 312]
            nn.Conv1d(in_channels=48, out_channels=96, kernel_size=5, stride=1, padding=2), 
            nn.ReLU(),  
            nn.MaxPool1d(kernel_size=4, stride=4),  
            
            # final shape: [N, 96, 78]
        )

    def forward(self, x):
        return self.net(x)



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.net = nn.Sequential(
            nn.ConvTranspose1d(in_channels=96, out_channels=48, kernel_size=5, stride=4, padding=2, output_padding=3),
            nn.ReLU(),
            
            # intermediate: [N, 24, 4992]
            nn.ConvTranspose1d(in_channels=48, out_channels=24, kernel_size=5, stride=4, padding=2, output_padding=3),
            nn.ReLU(),
            
            # final: [N, 12, 4992]
            nn.ConvTranspose1d(in_channels=24, out_channels=12, kernel_size=5, stride=4, padding=2, output_padding=3),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.net(x)


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded