import functools
import torch
import torch.nn as nn
from torch.nn import init


"""
# --------------------------------------------
# select the network of G, D and F
# --------------------------------------------
"""

# --------------------------------------------
# Generator, netG, G
# --------------------------------------------
def define_G(opt):
    opt_net = opt['netG']
    net_type = opt_net['net_type']


    # ----------------------------------------
    # denoising task
    # ----------------------------------------

    # ----------------------------------------
    # BERUNet
    # ----------------------------------------
    if net_type == 'ResUNet_EnBorder_v2':
        from models.network_BERUNet import UNetRes_EnBorder_v2 as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nc=opt_net['nc'],
                   nb=opt_net['nb'],
                   nt=opt_net['nt'],
                   act_mode=opt_net['act_mode'],
                   Train_size=opt['datasets']['train']['H_size'],
                   ker_size=opt_net['ker_size'],
                   is_padding=opt_net['is_padding'],
                   bias=opt_net['bias'])
 
    # ----------------------------------------
    # BERUNet_Blind
    # ----------------------------------------
    elif net_type == 'ResUNet_EnBorder_Blind_v2':
        from models.network_BERUNet import UNetRes_EnBorder_Blind_v2 as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nc=opt_net['nc'],
                   nb=opt_net['nb'],
                   nt=opt_net['nt'],
                   act_mode=opt_net['act_mode'],
                   Train_size=opt['datasets']['train']['H_size'],
                   ker_size=opt_net['ker_size'],
                   is_padding=opt_net['is_padding'],
                   bias=opt_net['bias'])

    # ----------------------------------------
    # baseline 1 for BERUNet
    # ----------------------------------------
    elif net_type == 'UNetRes_Basic':
        from models.network_BERUNet import UNetRes_Basic as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nc=opt_net['nc'],
                   nb=opt_net['nb'],
                   nt=opt_net['nt'],
                   act_mode=opt_net['act_mode'],
                   bias=opt_net['bias'])

    # ----------------------------------------
    # baseline 2 for BERUNet
    # ----------------------------------------
    elif net_type == 'ResUNet_TConv':
        from models.network_BERUNet import UNetRes_TConv as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nc=opt_net['nc'],
                   nb=opt_net['nb'],
                   nt=opt_net['nt'],
                   act_mode=opt_net['act_mode'],
                   is_padding=opt_net['is_padding'],
                   bias=opt_net['bias'])

    # ----------------------------------------
    # baseline 3 for BERUNet
    # ----------------------------------------
    elif net_type == 'ResUNet_EnBorder_Patchwise':
        from models.network_BERUNet import UNetRes_EnBorder_Patchwise as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nc=opt_net['nc'],
                   nb=opt_net['nb'],
                   nt=opt_net['nt'],
                   act_mode=opt_net['act_mode'],
                   Train_size=opt['datasets']['train']['H_size'],
                   ker_size=opt_net['ker_size'],
                   is_padding=opt_net['is_padding'],
                   bias=opt_net['bias'])

    # ----------------------------------------
    # others
    # ----------------------------------------
    # TODO

    else:
        raise NotImplementedError('netG [{:s}] is not found.'.format(net_type))

    # ----------------------------------------
    # initialize weights
    # ----------------------------------------
    if opt['is_train']:
        init_weights(netG,
                     init_type=opt_net['init_type'],
                     init_bn_type=opt_net['init_bn_type'],
                     gain=opt_net['init_gain'])

    return netG


# --------------------------------------------
# Discriminator, netD, D
# --------------------------------------------
def define_D(opt):
    opt_net = opt['netD']
    net_type = opt_net['net_type']

    # ----------------------------------------
    # discriminator_vgg_96
    # ----------------------------------------
    if net_type == 'discriminator_vgg_96':
        from models.network_discriminator import Discriminator_VGG_96 as discriminator
        netD = discriminator(in_nc=opt_net['in_nc'],
                             base_nc=opt_net['base_nc'],
                             ac_type=opt_net['act_mode'])

    # ----------------------------------------
    # discriminator_vgg_128
    # ----------------------------------------
    elif net_type == 'discriminator_vgg_128':
        from models.network_discriminator import Discriminator_VGG_128 as discriminator
        netD = discriminator(in_nc=opt_net['in_nc'],
                             base_nc=opt_net['base_nc'],
                             ac_type=opt_net['act_mode'])

    # ----------------------------------------
    # discriminator_vgg_192
    # ----------------------------------------
    elif net_type == 'discriminator_vgg_192':
        from models.network_discriminator import Discriminator_VGG_192 as discriminator
        netD = discriminator(in_nc=opt_net['in_nc'],
                             base_nc=opt_net['base_nc'],
                             ac_type=opt_net['act_mode'])

    # ----------------------------------------
    # discriminator_vgg_128_SN
    # ----------------------------------------
    elif net_type == 'discriminator_vgg_128_SN':
        from models.network_discriminator import Discriminator_VGG_128_SN as discriminator
        netD = discriminator()

    elif net_type == 'discriminator_patchgan':
        from models.network_discriminator import Discriminator_PatchGAN as discriminator
        netD = discriminator(input_nc=opt_net['in_nc'],
                             ndf=opt_net['base_nc'],
                             n_layers=opt_net['n_layers'],
                             norm_type=opt_net['norm_type'])

    elif net_type == 'discriminator_unet':
        from models.network_discriminator import Discriminator_UNet as discriminator
        netD = discriminator(input_nc=opt_net['in_nc'],
                             ndf=opt_net['base_nc'])

    else:
        raise NotImplementedError('netD [{:s}] is not found.'.format(net_type))

    # ----------------------------------------
    # initialize weights
    # ----------------------------------------
    init_weights(netD,
                 init_type=opt_net['init_type'],
                 init_bn_type=opt_net['init_bn_type'],
                 gain=opt_net['init_gain'])

    return netD


# --------------------------------------------
# VGGfeature, netF, F
# --------------------------------------------
def define_F(opt, use_bn=False):
    device = torch.device('cuda' if opt['gpu_ids'] else 'cpu')
    from models.network_feature import VGGFeatureExtractor
    # pytorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = VGGFeatureExtractor(feature_layer=feature_layer,
                               use_bn=use_bn,
                               use_input_norm=True,
                               device=device)
    netF.eval()  # No need to train, but need BP to input
    return netF


"""
# --------------------------------------------
# weights initialization
# --------------------------------------------
"""

def init_weights(net, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
    """
    Extended weight initialization function (KAIR-based)
    Now supports: Conv, Linear, BatchNorm2d, WeightNorm (via weight_v & weight_g)
    
    Args:
        net: torch.nn.Module
        init_type: weight initialization type
        init_bn_type: batchnorm initialization type
        gain: scaling factor
    """

    def init_fn(m, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
        classname = m.__class__.__name__

        # 支持 weight_norm 包装后的模块
        if hasattr(m, 'weight_v'):
            if init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight_v.data, gain=gain)
            elif init_type == 'xavier_normal':
                init.xavier_normal_(m.weight_v.data, gain=gain)
            elif init_type == 'kaiming_normal':
                init.kaiming_normal_(m.weight_v.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight_v.data.mul_(gain).clamp_(-1, 1)
            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight_v.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight_v.data.mul_(gain)
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight_v.data, gain=gain)
            elif init_type == 'normal':
                init.normal_(m.weight_v.data, 0, 0.1)
                m.weight_v.data.mul_(gain).clamp_(-1, 1)
            elif init_type == 'uniform':
                init.uniform_(m.weight_v.data, -0.2, 0.2)
                m.weight_v.data.mul_(gain)
            else:
                raise NotImplementedError(f'Unsupported init_type: {init_type}')

            # 设置 weight_g = ||weight_v||，兼容所有维度（不使用 .norm）
            with torch.no_grad():
                if isinstance(m.weight_v, nn.Parameter):
                    num_dims = m.weight_v.dim()
                    if num_dims >= 2:
                        norm_dim = tuple(range(1, num_dims))
                        norm = (m.weight_v.data ** 2).sum(dim=norm_dim, keepdim=True).sqrt()
                        m.weight_g.data.copy_(norm)
                    else:
                        raise NotImplementedError(f"Unsupported weight_v shape: {m.weight_v.shape}")

            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.data.zero_()

        # 普通 Conv / Linear 模块
        elif (classname.find('Conv') != -1 or classname.find('Linear') != -1) and hasattr(m, 'weight'):
            if m.weight is not None and m.weight.requires_grad:
                if init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=gain)
                elif init_type == 'xavier_normal':
                    init.xavier_normal_(m.weight.data, gain=gain)
                    m.weight.data.clamp_(-1, 1)
                elif init_type == 'kaiming_normal':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                    m.weight.data.mul_(gain).clamp_(-1, 1)
                elif init_type == 'kaiming_uniform':
                    init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                    m.weight.data.mul_(gain)
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'normal':
                    init.normal_(m.weight.data, 0, 0.1)
                    m.weight.data.mul_(gain).clamp_(-1, 1)
                elif init_type == 'uniform':
                    init.uniform_(m.weight.data, -0.2, 0.2)
                    m.weight.data.mul_(gain)
                else:
                    raise NotImplementedError(f'Unsupported init_type: {init_type}')
            if m.bias is not None:
                m.bias.data.zero_()

        # BatchNorm2d
        elif classname.find('BatchNorm2d') != -1 and hasattr(m, 'weight') and m.affine:
            if init_bn_type == 'uniform':  # preferred
                init.uniform_(m.weight.data, 0.1, 1.0)
                init.constant_(m.bias.data, 0.0)
            elif init_bn_type == 'constant':
                init.constant_(m.weight.data, 1.0)
                init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError(f'Unsupported init_bn_type: {init_bn_type}')

    if init_type not in ['default', 'none']:
        print(f'Initialization method [{init_type} + {init_bn_type}], gain = {gain:.2f}')
        fn = functools.partial(init_fn, init_type=init_type, init_bn_type=init_bn_type, gain=gain)
        net.apply(fn)
    else:
        print('Pass this initialization! Initialization was done during network definition!')
