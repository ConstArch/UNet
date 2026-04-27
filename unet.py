import torch


class UNetDoubleConvolutionBlock(torch.nn.Module):
    
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv1 = torch.nn.Conv2d( in_chan, out_chan, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(out_chan, out_chan, kernel_size=3)
        self.relu  = torch.nn.ReLU()
    
    def forward(self, arg):
        return self.relu(self.conv2(self.relu(self.conv1(arg))))


class UNetEncoderBlock(torch.nn.Module):
    
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv = UNetDoubleConvolutionBlock(in_chan, out_chan)
        self.pool = torch.nn.MaxPool2d((2, 2))
    
    def forward(self, arg):
        res = self.conv(arg)
        return self.pool(res), res


class UNetDecoderBlock(torch.nn.Module):
    
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.up   = torch.nn.ConvTranspose2d(in_chan, out_chan, kernel_size=2, stride=2)
        self.conv = UNetDoubleConvolutionBlock(2 * out_chan, out_chan)
    
    def forward(self, arg, internal):
        pad_height = internal.shape[-2] // 2 - arg.shape[-2]
        pad_width  = internal.shape[-1] // 2 - arg.shape[-1]
        pad = (pad_width, pad_width, pad_height, pad_height)
        return self.conv(torch.cat([torch.nn.functional.pad(self.up(arg), pad=pad), internal], dim=-3))


class UNet(torch.nn.Module):
    
    def __init__(self, output_channel_count, min_channel_shape):
        
        super().__init__()
        
        u4_shape = torch.tensor(min_channel_shape)
        
        r4_shape = 2 * (u4_shape + 4)
        
        r3_shape = 2 * (r4_shape + 4)
        r2_shape = 2 * (r3_shape + 4)
        r1_shape = 2 * (r2_shape + 4)
        
        self.input_shape  = tuple((r1_shape + 4).tolist())
        self.output_shape = tuple((r1_shape - 4).tolist())
        
        self.enc1 = UNetEncoderBlock(  3,  64)
        self.enc2 = UNetEncoderBlock( 64, 128)
        self.enc3 = UNetEncoderBlock(128, 256)
        self.enc4 = UNetEncoderBlock(256, 512)
        
        self.conv = UNetDoubleConvolutionBlock(512, 1024)
        
        self.dec4 = UNetDecoderBlock(1024, 512)
        self.dec3 = UNetDecoderBlock( 512, 256)
        self.dec2 = UNetDecoderBlock( 256, 128)
        self.dec1 = UNetDecoderBlock( 128,  64)
        
        self.outp = torch.nn.Conv2d(64, output_channel_count, kernel_size=1)
    
    def forward(self, arg):
        
        p1, r1 = self.enc1(arg)
        p2, r2 = self.enc2(p1)
        p3, r3 = self.enc3(p2)
        p4, r4 = self.enc4(p3)
        
        u4 = self.conv(p4)
        
        u3 = self.dec4(u4, r4)
        u2 = self.dec3(u3, r3)
        u1 = self.dec2(u2, r2)
        res = self.dec1(u1, r1)
        
        return self.outp(res)
