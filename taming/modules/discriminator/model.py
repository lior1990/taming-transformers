import functools
import torch.nn as nn
import torch.nn.functional as F


from taming.modules.util import ActNorm


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, max_n_scales=5, scale_factor=2, base_channels=128, extra_conv_layers=0):
        super(MultiScaleDiscriminator, self).__init__()
        self.base_channels = base_channels
        self.scale_factor = scale_factor
        self.min_size = 16
        self.extra_conv_layers = extra_conv_layers

        # We want the max num of scales to fit the size of the real examples. further scaling would create networks that
        # only train on fake examples
        self.max_n_scales = max_n_scales

        # Prepare a list of all the networks for all the wanted scales
        self.nets = nn.ModuleList()

        # Create a network for each scale
        for _ in range(self.max_n_scales):
            self.nets.append(self.make_net())

        self.scales_weight = [1 / self.max_n_scales for _ in range(self.max_n_scales)]

    def make_net(self):
        base_channels = self.base_channels
        net = []

        # Entry block
        net += [nn.utils.spectral_norm(nn.Conv2d(3, base_channels, kernel_size=3, stride=1)),
                nn.BatchNorm2d(base_channels),
                nn.LeakyReLU(0.2, True)]

        # Downscaling blocks
        # A sequence of strided conv-blocks. Image dims shrink by 2, channels dim expands by 2 at each block
        net += [nn.utils.spectral_norm(nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2)),
                nn.BatchNorm2d(base_channels * 2),
                nn.LeakyReLU(0.2, True)]

        # Regular conv-block
        net += [nn.utils.spectral_norm(nn.Conv2d(in_channels=base_channels * 2,
                                                 out_channels=base_channels * 2,
                                                 kernel_size=3,
                                                 bias=True)),
                nn.BatchNorm2d(base_channels * 2),
                nn.LeakyReLU(0.2, True)]

        # Additional 1x1 conv-blocks
        for _ in range(self.extra_conv_layers):
            net += [nn.utils.spectral_norm(nn.Conv2d(in_channels=base_channels * 2,
                                                     out_channels=base_channels * 2,
                                                     kernel_size=3,
                                                     bias=True)),
                    nn.BatchNorm2d(base_channels * 2),
                    nn.LeakyReLU(0.2, True)]

        # Final conv-block
        # Ends with a Sigmoid to get a range of 0-1
        net += nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(base_channels * 2, 1, kernel_size=1)),
                             nn.Sigmoid())

        # Make it a valid layers sequence and return
        return nn.Sequential(*net)

    def forward(self, input_tensor):
        scale_weights = self.scales_weight  # todo: set this as parameter and make it dynamic

        aggregated_result_maps_from_all_scales = self.nets[0](input_tensor) * scale_weights[0]
        map_size = aggregated_result_maps_from_all_scales.shape[2:]

        # Run all nets over all scales and aggregate the interpolated results
        for net, scale_weight, i in zip(self.nets[1:], scale_weights[1:], range(1, len(scale_weights))):
            downscaled_image = F.interpolate(input_tensor, scale_factor=self.scale_factor**(-i), mode='bilinear')
            result_map_for_current_scale = net(downscaled_image)
            upscaled_result_map_for_current_scale = F.interpolate(result_map_for_current_scale,
                                                                  size=map_size,
                                                                  mode='bilinear')
            aggregated_result_maps_from_all_scales += upscaled_result_map_for_current_scale * scale_weight

        return aggregated_result_maps_from_all_scales
