import torch
import torch.nn as nn
import torch.nn.functional as F
import models.basicblock as B
from torch import Tensor

class UNetRes_EnBorder_v2(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=4, nt=1, act_mode='R', 
                 Train_size=128, ker_size=3, is_padding=True, bias=False):
        super(UNetRes_EnBorder_v2, self).__init__()

        if is_padding:
            rb_padding = (ker_size-1)//2
        else:
            rb_padding = 0

        grad_gain_body1 = calculate_grad_gain(Train_size=Train_size,    patch_size=ker_size, padding=rb_padding)
        grad_gain_up1   = calculate_grad_gain(Train_size=Train_size,    patch_size=3,        padding=1         )
        grad_gain_body2 = calculate_grad_gain(Train_size=Train_size//2, patch_size=ker_size, padding=rb_padding)
        grad_gain_up2   = calculate_grad_gain(Train_size=Train_size//2, patch_size=3,        padding=1         )
        grad_gain_body3 = calculate_grad_gain(Train_size=Train_size//4, patch_size=ker_size, padding=rb_padding)
        grad_gain_up3   = calculate_grad_gain(Train_size=Train_size//4, patch_size=3,        padding=1         )
        grad_gain_body4 = calculate_grad_gain(Train_size=Train_size//8, patch_size=ker_size, padding=rb_padding)

        self.m_head  = B.conv(in_nc+1, nc[0], kernel_size=3, padding=1, bias=False, mode='C')
        self.m_body1 = B.sequential(*[ResBlock_EnBorder(nc = [nc[0],nt*nc[0],nc[0]], kernel_size=ker_size, padding=rb_padding, 
                                                        act_mode=act_mode, grad_gain=grad_gain_body1, bias=bias) for _ in range(nb)])
        self.m_down1 = nn.Sequential(nn.Conv2d(nc[0], nc[0]//2, kernel_size=3, stride=1, padding=1, bias=False), 
                                     nn.PixelUnshuffle(2))
        self.m_body2 = B.sequential(*[ResBlock_EnBorder(nc = [nc[1],nt*nc[1],nc[1]], kernel_size=ker_size, padding=rb_padding,
                                                        act_mode=act_mode, grad_gain=grad_gain_body2, bias=bias) for _ in range(nb)])
        self.m_down2 = nn.Sequential(nn.Conv2d(nc[1], nc[1]//2, kernel_size=3, stride=1, padding=1, bias=False), 
                                     nn.PixelUnshuffle(2))
        self.m_body3 = B.sequential(*[ResBlock_EnBorder(nc = [nc[2],nt*nc[2],nc[2]], kernel_size=ker_size, padding=rb_padding,
                                                        act_mode=act_mode, grad_gain=grad_gain_body3, bias=bias) for _ in range(nb)])
        self.m_down3 = nn.Sequential(nn.Conv2d(nc[2], nc[2]//2, kernel_size=3, stride=1, padding=1, bias=False), 
                                     nn.PixelUnshuffle(2))
        self.m_body4 = B.sequential(*[ResBlock_EnBorder(nc = [nc[3],nt*nc[3],nc[3]], kernel_size=ker_size, padding=rb_padding,
                                                        act_mode=act_mode, grad_gain=grad_gain_body4, bias=bias) for _ in range(nb)])
        self.m_up3   = nn.Sequential(nn.PixelShuffle(2), 
                                     TConv_EnBorder(nc[2]//2, nc[2], kernel_size=3, padding=1, bias=False, grad_gain=grad_gain_up3))
        self.m_body5 = B.sequential(*[ResBlock_EnBorder(nc = [nc[2],nt*nc[2],nc[2]], kernel_size=ker_size, padding=rb_padding,
                                                        act_mode=act_mode, grad_gain=grad_gain_body3, bias=bias) for _ in range(nb)])
        self.m_up2   = nn.Sequential(nn.PixelShuffle(2), 
                                     TConv_EnBorder(nc[1]//2, nc[1], kernel_size=3, padding=1, bias=False, grad_gain=grad_gain_up2))
        self.m_body6 = B.sequential(*[ResBlock_EnBorder(nc = [nc[1],nt*nc[1],nc[1]], kernel_size=ker_size, padding=rb_padding,
                                                        act_mode=act_mode, grad_gain=grad_gain_body2, bias=bias) for _ in range(nb)])
        self.m_up1   = nn.Sequential(nn.PixelShuffle(2), 
                                     TConv_EnBorder(nc[0]//2, nc[0], kernel_size=3, padding=1, bias=False, grad_gain=grad_gain_up1))
        self.m_body7 = B.sequential(*[ResBlock_EnBorder(nc = [nc[0],nt*nc[0],nc[0]], kernel_size=ker_size, padding=rb_padding,
                                                        act_mode=act_mode, grad_gain=grad_gain_body1, bias=bias) for _ in range(nb)])
        self.m_tail  = TConv_EnBorder(nc[0], out_nc, kernel_size=3, padding=1, bias=False, grad_gain=grad_gain_up1)


    def forward(self, x0, sigma):

        s  = sigma.repeat(1, 1, x0.size()[-2], x0.size()[-1])
        xs = torch.cat((x0, s), 1)
        x1 = self.m_head(xs)
        x1 = self.m_body1(x1)

        x2 = self.m_down1(x1)
        x2 = self.m_body2(x2)

        x3 = self.m_down2(x2)
        x3 = self.m_body3(x3)

        x  = self.m_down3(x3)
        x  = self.m_body4(x)
        x  = self.m_up3(x)

        x  = self.m_body5(x+x3)
        x  = self.m_up2(x)

        x  = self.m_body6(x+x2)
        x  = self.m_up1(x)

        x  = self.m_body7(x+x1)
        x  = self.m_tail(x)

        return x+x0

class UNetRes_EnBorder_Blind_v2(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=4, nt=1, act_mode='R', 
                 Train_size=128, ker_size=3, is_padding=True, bias=False):
        super(UNetRes_EnBorder_Blind_v2, self).__init__()

        if is_padding:
            rb_padding = (ker_size-1)//2
        else:
            rb_padding = 0

        grad_gain_body1 = calculate_grad_gain(Train_size=Train_size,    patch_size=ker_size, padding=rb_padding)
        grad_gain_up1   = calculate_grad_gain(Train_size=Train_size,    patch_size=3,        padding=1         )
        grad_gain_body2 = calculate_grad_gain(Train_size=Train_size//2, patch_size=ker_size, padding=rb_padding)
        grad_gain_up2   = calculate_grad_gain(Train_size=Train_size//2, patch_size=3,        padding=1         )
        grad_gain_body3 = calculate_grad_gain(Train_size=Train_size//4, patch_size=ker_size, padding=rb_padding)
        grad_gain_up3   = calculate_grad_gain(Train_size=Train_size//4, patch_size=3,        padding=1         )
        grad_gain_body4 = calculate_grad_gain(Train_size=Train_size//8, patch_size=ker_size, padding=rb_padding)

        self.m_head  = B.conv(in_nc, nc[0], kernel_size=3, padding=1, bias=False, mode='C')
        self.m_body1 = B.sequential(*[ResBlock_EnBorder(nc = [nc[0],nt*nc[0],nc[0]], kernel_size=ker_size, padding=rb_padding, 
                                                        act_mode=act_mode, grad_gain=grad_gain_body1, bias=bias) for _ in range(nb)])
        self.m_down1 = nn.Sequential(nn.Conv2d(nc[0], nc[0]//2, kernel_size=3, stride=1, padding=1, bias=False), 
                                     nn.PixelUnshuffle(2))
        self.m_body2 = B.sequential(*[ResBlock_EnBorder(nc = [nc[1],nt*nc[1],nc[1]], kernel_size=ker_size, padding=rb_padding,
                                                        act_mode=act_mode, grad_gain=grad_gain_body2, bias=bias) for _ in range(nb)])
        self.m_down2 = nn.Sequential(nn.Conv2d(nc[1], nc[1]//2, kernel_size=3, stride=1, padding=1, bias=False), 
                                     nn.PixelUnshuffle(2))
        self.m_body3 = B.sequential(*[ResBlock_EnBorder(nc = [nc[2],nt*nc[2],nc[2]], kernel_size=ker_size, padding=rb_padding,
                                                        act_mode=act_mode, grad_gain=grad_gain_body3, bias=bias) for _ in range(nb)])
        self.m_down3 = nn.Sequential(nn.Conv2d(nc[2], nc[2]//2, kernel_size=3, stride=1, padding=1, bias=False), 
                                     nn.PixelUnshuffle(2))
        self.m_body4 = B.sequential(*[ResBlock_EnBorder(nc = [nc[3],nt*nc[3],nc[3]], kernel_size=ker_size, padding=rb_padding,
                                                        act_mode=act_mode, grad_gain=grad_gain_body4, bias=bias) for _ in range(nb)])
        self.m_up3   = nn.Sequential(nn.PixelShuffle(2), 
                                     TConv_EnBorder(nc[2]//2, nc[2], kernel_size=3, padding=1, bias=False, grad_gain=grad_gain_up3))
        self.m_body5 = B.sequential(*[ResBlock_EnBorder(nc = [nc[2],nt*nc[2],nc[2]], kernel_size=ker_size, padding=rb_padding,
                                                        act_mode=act_mode, grad_gain=grad_gain_body3, bias=bias) for _ in range(nb)])
        self.m_up2   = nn.Sequential(nn.PixelShuffle(2), 
                                     TConv_EnBorder(nc[1]//2, nc[1], kernel_size=3, padding=1, bias=False, grad_gain=grad_gain_up2))
        self.m_body6 = B.sequential(*[ResBlock_EnBorder(nc = [nc[1],nt*nc[1],nc[1]], kernel_size=ker_size, padding=rb_padding,
                                                        act_mode=act_mode, grad_gain=grad_gain_body2, bias=bias) for _ in range(nb)])
        self.m_up1   = nn.Sequential(nn.PixelShuffle(2), 
                                     TConv_EnBorder(nc[0]//2, nc[0], kernel_size=3, padding=1, bias=False, grad_gain=grad_gain_up1))
        self.m_body7 = B.sequential(*[ResBlock_EnBorder(nc = [nc[0],nt*nc[0],nc[0]], kernel_size=ker_size, padding=rb_padding,
                                                        act_mode=act_mode, grad_gain=grad_gain_body1, bias=bias) for _ in range(nb)])
        self.m_tail  = TConv_EnBorder(nc[0], out_nc, kernel_size=3, padding=1, bias=False, grad_gain=grad_gain_up1)

    def forward(self, x0):

        x1 = self.m_head(x0)
        x1 = self.m_body1(x1)

        x2 = self.m_down1(x1)
        x2 = self.m_body2(x2)

        x3 = self.m_down2(x2)
        x3 = self.m_body3(x3)

        x  = self.m_down3(x3)
        x  = self.m_body4(x)
        x  = self.m_up3(x)

        x  = self.m_body5(x+x3)
        x  = self.m_up2(x)

        x  = self.m_body6(x+x2)
        x  = self.m_up1(x)

        x  = self.m_body7(x+x1)
        x  = self.m_tail(x)

        return x+x0

class ResBlock_EnBorder(nn.Module):
    def __init__(self, nc = [64, 64, 64], act_mode='R', kernel_size=3, stride=1, padding=1, negative_slope=0.1, grad_gain=9, bias=True):
        super(ResBlock_EnBorder, self).__init__()
        self.Conv   = B.conv(in_channels=nc[0], out_channels=nc[1], kernel_size=kernel_size, stride=stride, 
                             padding=padding, bias=bias, mode='C')
        self.Act    = B.conv(mode=act_mode, negative_slope=negative_slope)
        self.TConv  = B.conv(in_channels=nc[1], out_channels=nc[2], kernel_size=kernel_size, stride=stride, 
                             padding=padding, bias=False, mode='T')
        self.kSize  = kernel_size
        self.stride = stride
        self.ggain  = grad_gain
        self.pad    = padding
        self.isBias = bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(nc[2]))

    def forward(self, x):
        res = self.Conv(x)
        res = self.Act(res)
        batch_size, channels, in_height, in_width = res.size()
        res = self.TConv(res)

        out_height = (in_height*self.stride)+self.kSize-1
        out_width  = (in_width *self.stride)+self.kSize-1
        snum = torch.ones((1, self.kSize**2, in_height*in_width), dtype=res.dtype, device=res.device)
        fold = nn.Fold(output_size=(out_height, out_width), kernel_size=(self.kSize, self.kSize), padding=0, stride=self.stride)
        counts = fold(snum)
        if self.pad > 0:
            res = res*self.ggain/counts[..., self.pad:-self.pad, self.pad:-self.pad]
        else:
            res = res*self.ggain/counts

        if self.isBias:
            res = res + self.bias.view(1, -1, 1, 1)

        return x + res

class TConv_EnBorder(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, grad_gain=9, bias=True):
        super(TConv_EnBorder, self).__init__()
        self.TConv  = B.conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, 
                             padding=padding, bias=False, mode='T')
        self.kSize  = kernel_size
        self.stride = stride
        self.ggain  = grad_gain
        self.pad    = padding
        self.isBias = bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        _, _, in_height, in_width = x.size()
        x = self.TConv(x)

        out_height = (in_height*self.stride)+self.kSize-1
        out_width  = (in_width *self.stride)+self.kSize-1
        snum = torch.ones((1, self.kSize**2, in_height*in_width), dtype=x.dtype, device=x.device)
        fold = nn.Fold(output_size=(out_height, out_width), kernel_size=(self.kSize, self.kSize), padding=0, stride=self.stride)
        counts = fold(snum)
        if self.pad > 0:
            x = x*self.ggain/counts[..., self.pad:-self.pad, self.pad:-self.pad]
        else:
            x = x*self.ggain/counts

        if self.isBias:
            x = x + self.bias.view(1, -1, 1, 1)

        return x

# --------------------------------------------
# Compute the gradient backpropagation ratio to ensure 
# consistency of convergence behavior with existing TConv and Conv layers 
# 计算梯度反传比例，确保与现有TConv和Conv层收敛效果一致
# --------------------------------------------
def calculate_grad_gain(Train_size, patch_size=3, stride=1, dilate=1, padding=0):
    mat   = torch.ones((1, 1, Train_size, Train_size))
    if padding > 0:
        mat = F.pad(mat, pad=(padding, padding, padding, padding), mode='constant', value=0)
        
    unfold = nn.Unfold(kernel_size=(patch_size, patch_size), dilation=dilate, padding=0, stride=stride)
    unfolded_mat = unfold(mat)
    
    grad_gain = unfolded_mat.sum()/mat.sum()
    return grad_gain

# --------------------------------------------
# Basic ResUnet (baseline method 1)
# 基础残差U-Net（基准方法1）
# --------------------------------------------
class UNetRes_Basic(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=[64, 128, 256, 516], nb=4, nt=1, act_mode='R', 
                 ker_size=3, padding_mode='zeros', bias=False):
        super(UNetRes_Basic, self).__init__()

        rb_padding   = (ker_size-1)//2

        self.m_head  = B.conv(in_nc+1, nc[0], kernel_size=3, padding=1, bias=False, mode='C')
        self.m_body1 = B.sequential(*[ResBlock_Basic(nc = [nc[0],nt*nc[0],nc[0]], act_mode=act_mode, kernel_size=ker_size, padding=rb_padding, 
                                                     padding_mode=padding_mode, bias=bias) for _ in range(nb)])
        self.m_down1 = nn.Sequential(B.conv(nc[0], nc[0]//2, kernel_size=3, padding=1, bias=False, mode='C'), 
                                     nn.PixelUnshuffle(2))
        self.m_body2 = B.sequential(*[ResBlock_Basic(nc = [nc[1],nt*nc[1],nc[1]], act_mode=act_mode, kernel_size=ker_size, padding=rb_padding, 
                                                     padding_mode=padding_mode, bias=bias) for _ in range(nb)])
        self.m_down2 = nn.Sequential(B.conv(nc[1], nc[1]//2, kernel_size=3, padding=1, bias=False, mode='C'), 
                                     nn.PixelUnshuffle(2))
        self.m_body3 = B.sequential(*[ResBlock_Basic(nc = [nc[2],nt*nc[2],nc[2]], act_mode=act_mode, kernel_size=ker_size, padding=rb_padding, 
                                                     padding_mode=padding_mode, bias=bias) for _ in range(nb)])
        self.m_down3 = nn.Sequential(B.conv(nc[2], nc[2]//2, kernel_size=3, padding=1, bias=False, mode='C'), 
                                     nn.PixelUnshuffle(2))
        self.m_body4 = B.sequential(*[ResBlock_Basic(nc = [nc[3],nt*nc[3],nc[3]], act_mode=act_mode, kernel_size=ker_size, padding=rb_padding, 
                                                     padding_mode=padding_mode, bias=bias) for _ in range(nb)])
        self.m_up3   = nn.Sequential(nn.PixelShuffle(2), 
                                     B.conv(nc[2]//2, nc[2], kernel_size=3, padding=1, bias=False, mode='C'))
        self.m_body5 = B.sequential(*[ResBlock_Basic(nc = [nc[2],nt*nc[2],nc[2]], act_mode=act_mode, kernel_size=ker_size, padding=rb_padding, 
                                                     padding_mode=padding_mode, bias=bias) for _ in range(nb)])
        self.m_up2   = nn.Sequential(nn.PixelShuffle(2), 
                                     B.conv(nc[1]//2, nc[1], kernel_size=3, padding=1, bias=False, mode='C'))
        self.m_body6 = B.sequential(*[ResBlock_Basic(nc = [nc[1],nt*nc[1],nc[1]], act_mode=act_mode, kernel_size=ker_size, padding=rb_padding, 
                                                     padding_mode=padding_mode, bias=bias) for _ in range(nb)])
        self.m_up1   = nn.Sequential(nn.PixelShuffle(2), 
                                     B.conv(nc[0]//2, nc[0], kernel_size=3, padding=1, bias=False, mode='C'))
        self.m_body7 = B.sequential(*[ResBlock_Basic(nc = [nc[0],nt*nc[0],nc[0]], act_mode=act_mode, kernel_size=ker_size, padding=rb_padding, 
                                                     padding_mode=padding_mode, bias=bias) for _ in range(nb)])
        self.m_tail  = B.conv(nc[0], out_nc, kernel_size=3, padding=1, bias=False, mode='C')

    def forward(self, x0, sigma):

        s  = sigma.repeat(1, 1, x0.size()[-2], x0.size()[-1])
        xs = torch.cat((x0, s), 1)
        x1 = self.m_head(xs)
        x1 = self.m_body1(x1)

        x2 = self.m_down1(x1)
        x2 = self.m_body2(x2)

        x3 = self.m_down2(x2)
        x3 = self.m_body3(x3)

        x  = self.m_down3(x3)
        x  = self.m_body4(x)
        x  = self.m_up3(x)

        x  = self.m_body5(x+x3)
        x  = self.m_up2(x)

        x  = self.m_body6(x+x2)
        x  = self.m_up1(x)

        x  = self.m_body7(x+x1)
        x  = self.m_tail(x)

        return x+x0

class ResBlock_Basic(nn.Module):
    def __init__(self, nc = [64, 64, 64], act_mode='R', kernel_size=3, stride=1, padding=1, padding_mode='zeros', 
                 negative_slope=0.01, bias=True):
        super(ResBlock_Basic, self).__init__()
        self.Conv1  = nn.Conv2d(in_channels=nc[0], out_channels=nc[1], kernel_size=kernel_size, stride=stride, 
                               padding=padding, padding_mode=padding_mode, bias=bias)
        self.Act   = B.conv(mode=act_mode, negative_slope=negative_slope)
        self.Conv2 = nn.Conv2d(in_channels=nc[1], out_channels=nc[2], kernel_size=kernel_size, stride=stride, 
                               padding=padding, padding_mode=padding_mode, bias=bias)

    def forward(self, x):
        res = self.Conv1(x)
        res = self.Act(res)
        res = self.Conv2(res)

        return x + res

# --------------------------------------------
# Basic TConv ResUNet (baseline method 2)
# 基础反卷积残差U-Net（基准方法2）
# --------------------------------------------
class UNetRes_TConv(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=4, nt=1, act_mode='R', 
                 ker_size=3, is_padding=True,  bias=False):
        super(UNetRes_TConv, self).__init__()

        if is_padding:
            rb_padding = (ker_size-1)//2
        else:
            rb_padding = 0

        self.m_head  = B.conv(in_nc+1, nc[0], kernel_size=3, padding=1, bias=False, mode='C')
        self.m_body1 = B.sequential(*[ResBlock_TConv(nc = [nc[0],nt*nc[0],nc[0]], kernel_size=ker_size, padding=rb_padding, 
                                                     act_mode=act_mode, bias=bias) for _ in range(nb)])
        self.m_down1 = nn.Sequential(B.conv(nc[0], nc[0]//2, kernel_size=3, padding=1, bias=False, mode='C'), 
                                     nn.PixelUnshuffle(2))
        self.m_body2 = B.sequential(*[ResBlock_TConv(nc = [nc[1],nt*nc[1],nc[1]], kernel_size=ker_size, padding=rb_padding, 
                                                     act_mode=act_mode, bias=bias) for _ in range(nb)])
        self.m_down2 = nn.Sequential(B.conv(nc[1], nc[1]//2, kernel_size=3, padding=1, bias=False, mode='C'), 
                                     nn.PixelUnshuffle(2))
        self.m_body3 = B.sequential(*[ResBlock_TConv(nc = [nc[2],nt*nc[2],nc[2]], kernel_size=ker_size, padding=rb_padding, 
                                                     act_mode=act_mode, bias=bias) for _ in range(nb)])
        self.m_down3 = nn.Sequential(B.conv(nc[2], nc[2]//2, kernel_size=3, padding=1, bias=False, mode='C'), 
                                     nn.PixelUnshuffle(2))
        self.m_body4 = B.sequential(*[ResBlock_TConv(nc = [nc[3],nt*nc[3],nc[3]], kernel_size=ker_size, padding=rb_padding, 
                                                     act_mode=act_mode, bias=bias) for _ in range(nb)])
        self.m_up3   = nn.Sequential(nn.PixelShuffle(2), 
                                     B.conv(nc[2]//2, nc[2], kernel_size=3, padding=1, bias=False, mode='T'))
        self.m_body5 = B.sequential(*[ResBlock_TConv(nc = [nc[2],nt*nc[2],nc[2]], kernel_size=ker_size, padding=rb_padding, 
                                                     act_mode=act_mode, bias=bias) for _ in range(nb)])
        self.m_up2   = nn.Sequential(nn.PixelShuffle(2), 
                                     B.conv(nc[1]//2, nc[1], kernel_size=3, padding=1, bias=False, mode='T'))
        self.m_body6 = B.sequential(*[ResBlock_TConv(nc = [nc[1],nt*nc[1],nc[1]], kernel_size=ker_size, padding=rb_padding, 
                                                     act_mode=act_mode, bias=bias) for _ in range(nb)])
        self.m_up1   = nn.Sequential(nn.PixelShuffle(2), 
                                     B.conv(nc[0]//2, nc[0], kernel_size=3, padding=1, bias=False, mode='T'))
        self.m_body7 = B.sequential(*[ResBlock_TConv(nc = [nc[0],nt*nc[0],nc[0]], kernel_size=ker_size, padding=rb_padding, 
                                                     act_mode=act_mode, bias=bias) for _ in range(nb)])
        self.m_tail  = B.conv(nc[0], out_nc, kernel_size=3, padding=1, bias=False, mode='T')

    def forward(self, x0, sigma):

        s  = sigma.repeat(1, 1, x0.size()[-2], x0.size()[-1])
        xs = torch.cat((x0, s), 1)
        x1 = self.m_head(xs)
        x1 = self.m_body1(x1)

        x2 = self.m_down1(x1)
        x2 = self.m_body2(x2)

        x3 = self.m_down2(x2)
        x3 = self.m_body3(x3)

        x  = self.m_down3(x3)
        x  = self.m_body4(x)
        x  = self.m_up3(x)

        x  = self.m_body5(x+x3)
        x  = self.m_up2(x)

        x  = self.m_body6(x+x2)
        x  = self.m_up1(x)

        x  = self.m_body7(x+x1)
        x  = self.m_tail(x)

        return x+x0

class ResBlock_TConv(nn.Module):
    def __init__(self, nc = [64, 64, 64], act_mode='R', kernel_size=3, stride=1, padding=1, negative_slope=0.01, bias=True):
        super(ResBlock_TConv, self).__init__()
        self.Conv  = B.conv(in_channels=nc[0], out_channels=nc[1], kernel_size=kernel_size, stride=stride, 
                            padding=padding, bias=bias, mode='C')
        self.Act   = B.conv(mode=act_mode, negative_slope=negative_slope)
        self.TConv = B.conv(in_channels=nc[1], out_channels=nc[2], kernel_size=kernel_size, stride=stride, 
                            padding=padding, bias=bias, mode='T')

    def forward(self, x):
        res = self.Conv(x)
        res = self.Act(res)
        res = self.TConv(res)

        return x + res
    
# --------------------------------------------
# Patch-wise BERUnet (baseline method 3)
# 显式图像块级BERUNet（基准方法3）
# --------------------------------------------
class UNetRes_EnBorder_Patchwise(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=4, nt=1, act_mode='R', 
                 Train_size=128, ker_size=3, is_padding=True, bias=False):
        super(UNetRes_EnBorder_Patchwise, self).__init__()

        if is_padding:
            rb_padding = (ker_size-1)//2
        else:
            rb_padding = 0

        grad_gain_body1 = calculate_grad_gain(Train_size=Train_size,    patch_size=ker_size, padding=rb_padding)
        grad_gain_up1   = calculate_grad_gain(Train_size=Train_size,    patch_size=3,        padding=1         )
        grad_gain_body2 = calculate_grad_gain(Train_size=Train_size//2, patch_size=ker_size, padding=rb_padding)
        grad_gain_up2   = calculate_grad_gain(Train_size=Train_size//2, patch_size=3,        padding=1         )
        grad_gain_body3 = calculate_grad_gain(Train_size=Train_size//4, patch_size=ker_size, padding=rb_padding)
        grad_gain_up3   = calculate_grad_gain(Train_size=Train_size//4, patch_size=3,        padding=1         )
        grad_gain_body4 = calculate_grad_gain(Train_size=Train_size//8, patch_size=ker_size, padding=rb_padding)

        self.m_head  = B.conv(in_nc+1, nc[0], kernel_size=3, padding=1, bias=False, mode='C')
        self.m_body1 = B.sequential(*[ResBlock_EnBorder_Patchwise(nc = [nc[0],nt*nc[0],nc[0]], kernel_size=ker_size, padding=rb_padding, 
                                                        act_mode=act_mode, grad_gain=grad_gain_body1, bias=bias) for _ in range(nb)])
        self.m_down1 = nn.Sequential(nn.Conv2d(nc[0], nc[0]//2, kernel_size=3, stride=1, padding=1, bias=False), 
                                     nn.PixelUnshuffle(2))
        self.m_body2 = B.sequential(*[ResBlock_EnBorder_Patchwise(nc = [nc[1],nt*nc[1],nc[1]], kernel_size=ker_size, padding=rb_padding,
                                                        act_mode=act_mode, grad_gain=grad_gain_body2, bias=bias) for _ in range(nb)])
        self.m_down2 = nn.Sequential(nn.Conv2d(nc[1], nc[1]//2, kernel_size=3, stride=1, padding=1, bias=False), 
                                     nn.PixelUnshuffle(2))
        self.m_body3 = B.sequential(*[ResBlock_EnBorder_Patchwise(nc = [nc[2],nt*nc[2],nc[2]], kernel_size=ker_size, padding=rb_padding,
                                                        act_mode=act_mode, grad_gain=grad_gain_body3, bias=bias) for _ in range(nb)])
        self.m_down3 = nn.Sequential(nn.Conv2d(nc[2], nc[2]//2, kernel_size=3, stride=1, padding=1, bias=False), 
                                     nn.PixelUnshuffle(2))
        self.m_body4 = B.sequential(*[ResBlock_EnBorder_Patchwise(nc = [nc[3],nt*nc[3],nc[3]], kernel_size=ker_size, padding=rb_padding,
                                                        act_mode=act_mode, grad_gain=grad_gain_body4, bias=bias) for _ in range(nb)])
        self.m_up3   = nn.Sequential(nn.PixelShuffle(2), 
                                     TConv_EnBorder(nc[2]//2, nc[2], kernel_size=3, padding=1, bias=False, grad_gain=grad_gain_up3))
        self.m_body5 = B.sequential(*[ResBlock_EnBorder_Patchwise(nc = [nc[2],nt*nc[2],nc[2]], kernel_size=ker_size, padding=rb_padding,
                                                        act_mode=act_mode, grad_gain=grad_gain_body3, bias=bias) for _ in range(nb)])
        self.m_up2   = nn.Sequential(nn.PixelShuffle(2), 
                                     TConv_EnBorder(nc[1]//2, nc[1], kernel_size=3, padding=1, bias=False, grad_gain=grad_gain_up2))
        self.m_body6 = B.sequential(*[ResBlock_EnBorder_Patchwise(nc = [nc[1],nt*nc[1],nc[1]], kernel_size=ker_size, padding=rb_padding,
                                                        act_mode=act_mode, grad_gain=grad_gain_body2, bias=bias) for _ in range(nb)])
        self.m_up1   = nn.Sequential(nn.PixelShuffle(2), 
                                     TConv_EnBorder(nc[0]//2, nc[0], kernel_size=3, padding=1, bias=False, grad_gain=grad_gain_up1))
        self.m_body7 = B.sequential(*[ResBlock_EnBorder_Patchwise(nc = [nc[0],nt*nc[0],nc[0]], kernel_size=ker_size, padding=rb_padding,
                                                        act_mode=act_mode, grad_gain=grad_gain_body1, bias=bias) for _ in range(nb)])
        self.m_tail  = TConv_EnBorder(nc[0], out_nc, kernel_size=3, padding=1, bias=False, grad_gain=grad_gain_up1)


    def forward(self, x0, sigma):

        s  = sigma.repeat(1, 1, x0.size()[-2], x0.size()[-1])
        xs = torch.cat((x0, s), 1)
        x1 = self.m_head(xs)
        x1 = self.m_body1(x1)

        x2 = self.m_down1(x1)
        x2 = self.m_body2(x2)

        x3 = self.m_down2(x2)
        x3 = self.m_body3(x3)

        x  = self.m_down3(x3)
        x  = self.m_body4(x)
        x  = self.m_up3(x)

        x  = self.m_body5(x+x3)
        x  = self.m_up2(x)

        x  = self.m_body6(x+x2)
        x  = self.m_up1(x)

        x  = self.m_body7(x+x1)
        x  = self.m_tail(x)

        return x+x0

class ResBlock_EnBorder_Patchwise(nn.Module):
    def __init__(self, nc = [64, 64, 64], act_mode='R', patch_size=3, stride=1, padding=1, negative_slope=0.1, grad_gain=9, bias=True):
        super(ResBlock_EnBorder_Patchwise, self).__init__()
        self.Im2Pat = Im2Patches(patch_size=patch_size, stride=stride, padding=padding)
        self.Conv1  = B.conv(in_channels=nc[0]*(patch_size**2), out_channels=nc[1], kernel_size=1, stride=1, 
                             padding=0, bias=bias, mode='C')
        self.Act    = B.conv(mode=act_mode, negative_slope=negative_slope)
        self.Conv2  = B.conv(in_channels=nc[1], out_channels=nc[2]*(patch_size**2), kernel_size=1, stride=1, 
                             padding=0, bias=bias, mode='C')
        self.Pat2Im = Patches2Im(patch_size=patch_size, stride=stride, padding=padding, grad_gain=grad_gain)

    def forward(self, x):
        res = self.Im2Pat(x)
        res = self.Conv1(res)
        res = self.Act(res)
        res = self.Conv2(res)
        res = self.Pat2Im(res)

        return x + res
    

# --------------------------------------------
# Im2Patches
# --------------------------------------------
class Im2Patches(nn.Module):
    def __init__(self, patch_size:int, stride=1, dilate=1, padding=0):
        super(Im2Patches, self).__init__()
        self.patch_size = patch_size
        self.stride     = stride
        self.dilate     = dilate
        self.padding    = padding
        self.unfold     = nn.Unfold(kernel_size=(patch_size, patch_size), dilation=self.dilate, padding=0, stride=stride)
    
    def forward(self, input):
        if self.padding > 0:
            input = F.pad(input, pad=(self.padding, self.padding, self.padding, self.padding), mode='constant', value=0)
        
        batch_size, channels, in_height, in_width = input.size()
        unfolded_input = self.unfold(input)
        out_height = (in_height-(self.patch_size-1)*self.dilate-1)//self.stride + 1
        out_width  = (in_width -(self.patch_size-1)*self.dilate-1)//self.stride + 1
        output     = unfolded_input.reshape(batch_size, channels*self.patch_size**2, out_height, out_width)

        return output

# --------------------------------------------
# Patches2Im
# --------------------------------------------
class Patches2Im(nn.Module):
    def __init__(self, patch_size=3, stride=1, dilate=1, grad_gain=1, padding=0):
        super(Patches2Im, self).__init__()
        self.patch_size = patch_size
        self.stride     = stride
        self.dilate     = dilate
        self.grad_gain  = grad_gain
        self.padding    = padding

    def _Patches2Im_forward_(self, input:Tensor, patch_size:int, stride:int, dilate:int, padding:int, grad_gain):
        batch_size, channels, in_height, in_width = input.size()

        out_height = (patch_size-1)*dilate + (in_height-1)*stride + 1
        out_width  = (patch_size-1)*dilate + (in_width-1)*stride  + 1

        input  = input.reshape(batch_size, channels, in_height*in_width)
        snum   = torch.ones((1, patch_size**2, in_height*in_width), dtype=input.dtype, device=input.device)
        fold   = nn.Fold(output_size=(out_height, out_width), kernel_size=(patch_size, patch_size), dilation=dilate, padding=0, stride=stride)
        output = fold(input)
        counts = fold(snum)
        output = output*grad_gain/counts

        if padding > 0:
            output = output[..., padding:-padding, padding:-padding]

        return output

    def forward(self, input):
        return self._Patches2Im_forward_(input, self.patch_size, self.stride, self.dilate, self.padding, self.grad_gain)