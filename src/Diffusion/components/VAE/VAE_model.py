import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class Downblock(nn.Module): 
    def __init__(self, in_channels, out_channels,  down_sample, num_layers, norm_channels): 
        super(Downblock, self).__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample 
        self.resnet_one = nn.ModuleList([

            nn.Sequential( # group nom -- silu activation -- conv2d 
                nn.GroupNorm(norm_channels , in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels, 
                          out_channels, kernel_size=3, padding=1)
            )
            for i in range(num_layers)
        ])

        self.resnet_two = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups=32, num_channels=norm_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )
            for i in range(num_layers)
        ])

        
        self.residual_input_conv = nn.ModuleList([ 
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1, stride=2)
            for i in range(num_layers)
        ]) 
        
        self.down_conv = (nn.Conv2d(in_channels, out_channels, kernel_size=4, 
                                   stride=2, padding=1) if down_sample else None) 
    def forward(self, x):
        out = x 
        for i in range(self.num_layers) : 
            res = out 
            out = self.resnet_one[i](out)
            out = self.resnet_two[i](out)
            out += self.residual_input_conv[i](res) 



        out = self.down_conv(out) 
        return out

class Upblock(nn.Modual): 
    def __init__(self, in_channels, out_channels, up_sample, num_layers, norm_channels): 
        super(Upblock, self).__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample 

        self.up_conv = (nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, 
                                   stride=2, padding=1) if up_sample else nn.identity())
        
        self.resnet_one = nn.ModuleList([

            nn.Sequential( # group nom -- silu activation -- conv2d 
                nn.GroupNorm(norm_channels , in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels, 
                          out_channels, kernel_size=3, padding=1)
            )
            for i in range(num_layers)
        ])

        self.resnet_two = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups=32, num_channels=norm_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )
            for i in range(num_layers)
        ])

        
        self.residual_input_conv = nn.ModuleList([ 
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1, stride=2)
            for i in range(num_layers)
        ]) 
        
    def forward(self, x):
        out = self.up_conv(x)
        for i in range(self.num_layers) : 
            res = out 
            out = self.resnet_one[i](out)
            out = self.resnet_two[i](out)
            out += self.residual_input_conv[i](res) 



        out = self.up_conv(out) 
        return out
    

class Midblock(nn.Module): 
    def __init__(self,in_channels,
        out_channels,
        num_heads,
        num_layers,
        norm_channels):



        super(Midblock, self).__init__() 
        self.num_layers = num_layers

     
        self.resnet_one = nn.ModuleList([

            nn.Sequential( # group nom -- silu activation -- conv2d 
                nn.GroupNorm(norm_channels , in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels, 
                          out_channels, kernel_size=3, padding=1)
            )
            for i in range(num_layers+1)
        ])


        self.resnet_two = nn.ModuleList([ 
            nn.Sequential(
                nn.GroupNorm(num_groups=32, num_channels=norm_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )
            for i in range(num_layers+1)
        ])

        self.attention_norm = nn.ModuleList([
            nn.GroupNorm(num_groups=32, num_channels=norm_channels)
            for i in range(num_layers)
        ])

        self.attention = nn.ModuleList([ 
            nn.MultiheadAttention(embed_dim=out_channels, num_heads = num_heads ,batch_first=True)
            for i in range(num_layers)
        ])

