import torch.nn as nn


class Block(nn.Module):
    def __init__(self, dim, D=3):
        super.__init__()
        self.conv_l1 = nn.Conv2d(dim, dim, kernel_size=10, bias=True, dimension=D)
        self.norm = nn.LayerNorm(dim, 1e-6)
        self.conv_l2 = nn.Linear(dim, 4*dim)
        self.act = nn.GELU()
        self.conv_l3 = nn.Linear(4*dim, dim)
        

    def forward(self, x):
        x = self.conv_l1(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.conv_l2(x)
        x = self.act(x)
        x = self.conv_l3(x)
        x = x.permute(0, 3, 1, 2)

        return x
    
    

class Encoder(nn.Module):
    def __init__(self,
                 in_chans=12,
                 depths=[3, 3, 9, 3],
                 dims=[92, 192, 384, 768]):
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
            
            nn.Conv1d(in_channels=96, out_channels=192, kernel_size=5, stride=1, padding=2), 
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
            nn.ConvTranspose1d(in_channels=192, out_channels=96, kernel_size=5, stride=4, padding=2, output_padding=3),
            nn.ReLU(),

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