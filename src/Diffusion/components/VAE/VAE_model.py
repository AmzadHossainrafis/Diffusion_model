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
    def forword(self, x):
        out = x 
        resnet_input = x 

        out = self.resnet_one[0](out)
        out = self.residual_input_conv[0](resnet_input) 
        

        for i in range(self.num_layers) : 
            batch_size, channels, height, width = out.shape 
            in_attn = out.reshape(batch_size, channels, height*width)
            in_attn= self.attention_norm[i](in_attn) 
            in_attn = in_attn.transpose(1,2) 
            out_attn = out_attn[i](in_attn, in_attn, in_attn) 
            out_attn = out_attn.transpose(1,2).reshape(batch_size, channels, height, width) 
            out = out + out_attn 

            res = out

            out = self.resnet_one[i+1](out)
            out = self.resnet_two[i+1](out)
            out += self.residual_input_conv[i+1](res)
        return out 
    




class VAE(nn.Module): 
    def __init__(self, in_channels, out_channels, num_heads, num_layers, norm_channels): 
        super(VAE, self).__init__() 
        self.downblock_one = Downblock(in_channels, 64, down_sample=False, num_layers=2, norm_channels=norm_channels)
        self.downblock_two = Downblock(64, 128, down_sample=True, num_layers=2, norm_channels=norm_channels)
        self.downblock_three = Downblock(128, 256, down_sample=True, num_layers=2, norm_channels=norm_channels)
        self.downblock_four = Downblock(256, 512, down_sample=True, num_layers=2, norm_channels=norm_channels)
        self.downblock_five = Downblock(512, 1024, down_sample=True, num_layers=2, norm_channels=norm_channels)
        self.midblock = Midblock(1024, 1024, num_heads, num_layers, norm_channels)
        self.upblock_one = Upblock(1024, 512, up_sample=True, num_layers=2, norm_channels=norm_channels)
        self.upblock_two = Upblock(512, 256, up_sample=True, num_layers=2, norm_channels=norm_channels)
        self.upblock_three = Upblock(256, 128, up_sample=True, num_layers=2, norm_channels=norm_channels)
        self.upblock_four = Upblock(128, 64, up_sample=True, num_layers=2, norm_channels=norm_channels)
        self.upblock_five = Upblock(64, out_channels, up_sample=False, num_layers=2, norm_channels=norm_channels)
    def forward(self, x): 
        out = self.downblock_one(x)
        out = self.downblock_two(out)
        out = self.downblock_three(out)
        out = self.downblock_four(out)
        out = self.downblock_five(out)
        
        mean , variance = torch.chunk(out, 2, dim=1) 
        out = mean + torch.randn_like(variance) * torch.exp(0.5 * variance)

        log_variance = torch.clamp(log_variance, min=-30, max=20)
        variance = torch.exp(log_variance)  
        sqrt_variance = torch.sqrt(variance) 


        return out



class VQVAE(nn.Module):
    def __init__(self, in_channels , model_config):
        super(VQVAE, self).__init__()
        self.down_channels = model_config["down_channels"]
        self.mid_channels = model_config["mid_channels"]
        self.down_sample = model_config["down_sample"]
        self.num_down_layers = model_config["num_down_layers"]
        self.num_mid_layers = model_config["num_mid_layers"]
        self.num_up_layers = model_config["num_up_layers"]

        # To disable attention in Downblock of Encoder and Upblock of Decoder
        self.attns = model_config["attn_down"]

        # Latent Dimension
        self.z_channels = model_config["z_channels"]
        self.codebook_size = model_config["codebook_size"]
        self.norm_channels = model_config["norm_channels"]
        self.num_heads = model_config["num_heads"]
        self.up_sample = reversed(self.down_sample)

        self.downblocks = nn.ModuleList([])

        for i in range(self.num_down_layers):
            self.downblocks.append(Downblock(in_channels if i == 0 else self.down_channels[i-1],
                                              self.down_channels[i],
                                              self.down_sample[i],
                                              self.num_layers,
                                              self.norm_channels))
            
        self.mid_blocks = nn.ModuleList([])
        for i in range(self.num_mid_layers):
            self.mid_blocks.append(Midblock(self.down_channels[-1] if i == 0 else self.mid_channels[i-1],
                                            self.mid_channels[i],
                                            self.num_heads,
                                            self.num_layers,
                                            self.norm_channels))
        self.encoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[-1])
        self.encoder_conv_out = nn.Conv2d(
            self.down_channels[-1], self.z_channels, kernel_size=3, padding=1
        )

        # Pre Quantization Convolution
        self.pre_quant_conv = nn.Conv2d(self.z_channels, self.z_channels, kernel_size=1)

        # Codebook
        self.embedding = nn.Embedding(self.codebook_size, self.z_channels)

        self.post_quant_conv = nn.Conv2d(self.z_channels, self.z_channels, kernel_size=1)

        #mid+up 
        self.decoder_mid=   nn.ModuleList([]) 
        for i in range(self.num_up_layers): 
            self.decoder_mid.append(Upblock(self.mid_channels[-1] if i == 0 else self.mid_channels[i-1],
                                        self.mid_channels[i],
                                        self.up_sample[i],
                                        self.num_layers,
                                        self.norm_channels))
            

            self.decoder_layers = nn.ModuleList([]) 
            for i in range(self.num_up_layers): 
                self.decoder_layers.append(Upblock(self.mid_channels[-1] if i == 0 else self.mid_channels[i-1],
                                        self.mid_channels[i],
                                        self.up_sample[i],
                                        self.num_layers,
                                        self.norm_channels))
                

        def quataize(self, x ):
            pass 
        


        def forward(self, x): 
            pass 
        
            
                
                
            

