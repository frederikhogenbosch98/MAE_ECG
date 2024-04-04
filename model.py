import torch.nn as nn
import torch
import torch.nn.functional as F
import pdb

class EncoderBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_l1 = nn.Conv2d(dim, dim, kernel_size=(7,3), padding=(3,1), stride=(1,1), bias=True)
        self.norm = nn.LayerNorm(dim, 1e-6)
        self.lin_up = nn.Linear(dim, 4*dim)
        self.act = nn.GELU()
        self.lin_down = nn.Linear(4*dim, dim)
        

    def forward(self, x):
        print(f'block shape: {x.shape}')
        input = x
        x = self.conv_l1(x)
        x = x.permute(0, 2, 3, 1) # from [N, C, H, W] to [N, H, W, C]
        x = self.norm(x)
        x = self.lin_up(x)
        x = self.act(x)
        x = self.lin_down(x)
        x = x.permute(0, 3, 1, 2) # from [N, H, W, C] to [N, C, H, W] 
        print(f'block end shape: {x.shape}')


        x = input + x


        return x
    
    
class DecoderBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_l1 = nn.ConvTranspose2d(dim, dim, kernel_size=(7,3), padding=(3,1), stride=(1,1), bias=True)
        self.dim = dim
        self.norm = nn.LayerNorm(dim, 1e-6)
        self.lin_up = nn.Linear(dim, 4*dim)
        self.act = nn.GELU()
        self.lin_down = nn.Linear(4*dim, dim)
        

    def forward(self, x):
        print(f'block shape: {x.shape}')
        input = x
        print('before conv')
        print(self.dim)
        x = self.conv_l1(x)
        print('after conv')
        x = x.permute(0, 2, 3, 1) # from [N, C, H, W] to [N, H, W, C]
        x = self.norm(x)
        x = self.lin_up(x)
        x = self.act(x)
        x = self.lin_down(x)
        x = x.permute(0, 3, 1, 2) # from [N, H, W, C] to [N, C, H, W] 
        print(f'block end shape: {x.shape}')


        x = input + x


        return x
        

class Encoder(nn.Module):
    def __init__(self,
                 in_chans=12,
                 depths=[12, 12, 12*3, 12],
                 dims=[92, 192, 384, 768]):
        super(Encoder, self).__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=(12,3), stride=(2,1), padding=(5,0)),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
            )

        self.downsample_layers.append(stem)
        for i in range(len(dims)-1):
            # pdb.set_trace()
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=(12,2), stride=(2,1), bias=True)
            )
            self.downsample_layers.append(downsample_layer)
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(len(dims)):
            stage = nn.Sequential(
                *[EncoderBlock(dim=dims[i]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]




    def forward(self, x):
        print(f'starting tensor input shape: {x.shape}')
        num_stages = len(self.stages)
        num_down_laywers = len(self.downsample_layers)
        for i in range(min(num_stages, num_down_laywers)):
            # pdb.set_trace()
            x = self.downsample_layers[i](x)
            print(f'after downsample layers: {x.shape}')
            x = self.stages[i](x)
            print(f'after stages: {x.shape}')


        return x

class Decoder(nn.Module):
    def __init__(self,
                 out_chans=12,
                 depths=[12, 12, 12*3, 12],
                 dims=[92, 192, 384, 768]):
        super(Decoder, self).__init__()
        self.downsample_layers = nn.ModuleList()
        print(dims)
        dims = [out_chans] + dims
        print(dims)
        # stem = nn.Sequential(
        #     nn.ConvTranspose2d(dims[1], dims[0], kernel_size=(6,3), stride=(2,1), padding=(5,0)),
        #     LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        #     )

        for i in reversed(range(len(dims))):
            # pdb.set_trace()
            print(dims[i])
            print(i)
            if i != 0:
                downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.ConvTranspose2d(dims[i], dims[i-1], kernel_size=(12,2), stride=(2,1), padding=(1,0), bias=True)

                )
            else:
                downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.ConvTranspose2d(dims[i], dims[i], kernel_size=(12,2), stride=(2,1), padding=(1,0), bias=True)

                ) 
            self.downsample_layers.append(downsample_layer)

        # self.downsample_layers.append(stem)
        self.stages = nn.ModuleList()

        for i in reversed(range(len(dims))):
            stage = nn.Sequential(
                DecoderBlock(dim=dims[i])
            )
            self.stages.append(stage)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((300, 6))


        # self.stem_final = nn.Sequential(
            # nn.ConvTranspose2d(dims[0], dims[0], kernel_size=(5,2), stride=(1,1), padding=(0,0)),
            # LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        # )


    def forward(self, x):
        print(f'starting tensor input shape: {x.shape}')
        num_stages = len(self.stages)
        num_down_laywers = len(self.downsample_layers)
        for i in range(min(num_stages, num_down_laywers)):
            # pdb.set_trace()
            x = self.stages[i](x)
            print(f'after stages: {x.shape}')
            x = self.downsample_layers[i](x)
            print(f'after downsample layers: {x.shape}')

        
        x = self.adaptive_pool(x)
        print(f'final shape: {x.shape}')
        return x

class AutoEncoder(nn.Module):
    def __init__(self, in_chans, dims=[92, 192, 384, 768], depths=[12, 12, 12*3, 12], decoder_embed_dim=12, decoder_depth=2):
        super().__init__()
        self.encoder = Encoder(in_chans=in_chans, dims=dims, depths=depths)
        self.decoder = Decoder(out_chans=in_chans, dims=dims, depths=depths)
        # decoder = [Block(
        #     dim=decoder_embed_dim) for i in range(decoder_depth)]
        # self.decoder = nn.Sequential(*decoder)        
        # self.proj = nn.Conv2d(
        #     in_channels = dims[-1],
        #     out_channels = decoder_embed_dim,
        #     kernel_size = 1
        # )

        # self.pred = nn.Conv2d(
        #     in_channels = decoder_embed_dim,
        #     out_channels = in_chans,
        #     kernel_size = 1
        # )
        

    def forward_encoder(self, x):
        x = self.encoder(x)
        return x

    def forward_decoder(self, x):
        print(f'before prod: {x.shape}')
        # x = self.proj(x)
        # print(x.shape)
        # x = self.decoder(x)
        # print(x.shape)
        # x = self.pred(x)
        x = self.decoder(x)
        print(x.shape)
        return x

    def forward(self, x):
        print("ENCODING")
        x = self.forward_encoder(x)
        print(x.shape)
        print("DECODING")
        x = self.forward_decoder(x)
        print(x.shape)
        print("FINISHED")
        return x



class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

# class Encoder(nn.Module):
#     def __init__(self,
#                  in_chans=12,
#                  depths=[3, 3, 9, 3],
#                  dims=[92, 192, 384, 768]):
#         super(Encoder, self).__init__()
        
#         self.net = nn.Sequential(
#             # input shape: [N, 12, 4992]
#             nn.Conv1d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=2), 
#             nn.ReLU(), 
#             nn.MaxPool1d(kernel_size=4, stride=4),  
            
#             # intermediate shape: [N, 24, 1248]
#             nn.Conv1d(in_channels=24, out_channels=48, kernel_size=5, stride=1, padding=2), 
#             nn.ReLU(), 
#             nn.MaxPool1d(kernel_size=4, stride=4), 
            
#             # intermediate shape: [N, 48, 312]
#             nn.Conv1d(in_channels=48, out_channels=96, kernel_size=5, stride=1, padding=2), 
#             nn.ReLU(),  
#             nn.MaxPool1d(kernel_size=4, stride=4),  
            
#             nn.Conv1d(in_channels=96, out_channels=192, kernel_size=5, stride=1, padding=2), 
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=4, stride=4),
#             # final shape: [N, 96, 78]
#         )

#     def forward(self, x):
#         return self.net(x)



# class Decoder(nn.Module):
#     def __init__(self):
#         super(Decoder, self).__init__()
        
#         self.net = nn.Sequential(
#             nn.ConvTranspose1d(in_channels=192, out_channels=96, kernel_size=5, stride=4, padding=2, output_padding=3),
#             nn.ReLU(),

#             nn.ConvTranspose1d(in_channels=96, out_channels=48, kernel_size=5, stride=4, padding=2, output_padding=3),
#             nn.ReLU(),
            
#             # intermediate: [N, 24, 4992]
#             nn.ConvTranspose1d(in_channels=48, out_channels=24, kernel_size=5, stride=4, padding=2, output_padding=3),
#             nn.ReLU(),
            
#             # final: [N, 12, 4992]
#             nn.ConvTranspose1d(in_channels=24, out_channels=12, kernel_size=5, stride=4, padding=2, output_padding=3),
#             nn.Tanh()
#         )
    
#     def forward(self, x):
#         return self.net(x)