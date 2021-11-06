import torch
import torch.nn as nn
import torch.nn.functional as F


def Conv1x1(in_chans, out_chans, kernel_size=1):
    return nn.Conv2d(in_channels=in_chans, out_channels=out_chans,
                              kernel_size=kernel_size, stride=1, padding=0, dilation=1, bias=False)

def Conv3x3(in_chans, out_chans):
    return nn.Conv2d(in_channels=in_chans, out_channels=out_chans,
                              kernel_size=3, stride=1, padding=1, dilation=1, bias=False)


class FPN(nn.Module):
    def __init__(self, in_chans, mid_chans=256, for_detect=False, use_p6=False, **kwargs):
        super(FPN, self).__init__()
        self.use_p6 = use_p6
        self.for_detect = for_detect
        divisor = 5 if use_p6 else 4
        divisor = 1 if for_detect else divisor
        if self.use_p6:
            self.conv1x1_c6 = nn.MaxPool2d(kernel_size=1, stride=2)
            self.conv3x3_c6 = Conv3x3(mid_chans, mid_chans//divisor)

        self.conv1x1_c5 = Conv1x1(in_chans[3], mid_chans)
        self.conv1x1_c4 = Conv1x1(in_chans[2], mid_chans)
        self.conv1x1_c3 = Conv1x1(in_chans[1], mid_chans)
        self.conv1x1_c2 = Conv1x1(in_chans[0], mid_chans)

        self.conv3x3_c5 = Conv3x3(mid_chans, mid_chans//divisor)
        self.conv3x3_c4 = Conv3x3(mid_chans, mid_chans//divisor)
        self.conv3x3_c3 = Conv3x3(mid_chans, mid_chans//divisor)
        self.conv3x3_c2 = Conv3x3(mid_chans, mid_chans//divisor)

        self._init_layers()


    def forward(self, inp):
        bottom_ups = self.bottom_up(*inp)
        return self.top_down(bottom_ups)


    def _init_layers(self):
        if self.use_p6:
            self.conv1x1_c6.apply(self.weights_init)
        self.conv1x1_c2.apply(self.weights_init)
        self.conv1x1_c3.apply(self.weights_init)
        self.conv1x1_c4.apply(self.weights_init)
        self.conv1x1_c5.apply(self.weights_init)

        self.conv3x3_c2.apply(self.weights_init)
        self.conv3x3_c3.apply(self.weights_init)
        self.conv3x3_c4.apply(self.weights_init)
        self.conv3x3_c5.apply(self.weights_init)


    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)


    def bottom_up(self, c2, c3, c4, c5):
        p5 = self.conv1x1_c5(c5)
        p4 = self.conv1x1_c4(c4)
        p4 = p4 + F.interpolate(p5, size=p4.size()[2:], mode='nearest')

        p3 = self.conv1x1_c3(c3)
        p3 = p3 + F.interpolate(p4, size=p3.size()[2:], mode='nearest')

        p2 = self.conv1x1_c2(c2)
        p2 = p2 + F.interpolate(p3, size=p2.size()[2:], mode='nearest')

        res = [p2, p3, p4, p5]
        if self.use_p6:
            p6 = self.conv1x1_c6(p5)
            res.append(p6)

        return res


    def top_down(self, bottom_ups):
        if self.use_p6:
            p2, p3, p4, p5, p6 = bottom_ups
        else:
            p2, p3, p4, p5 = bottom_ups


        h, w = p2.size()[2:]

        new_p5 = self.conv3x3_c5(p5)
        if self.use_p6:
            new_p6 = self.conv3x3_c6(p6)
            new_p5 += F.interpolate(new_p6, size=p5.size()[2:], mode='nearest')
        new_p4 = self.conv3x3_c4(p4) + F.interpolate(new_p5, size=p4.size()[2:], mode='nearest')
        new_p3 = self.conv3x3_c3(p3) + F.interpolate(new_p4, size=p3.size()[2:], mode='nearest')
        new_p2 = self.conv3x3_c2(p2) + F.interpolate(new_p3, size=p2.size()[2:], mode='nearest')

        res = [new_p2, new_p3, new_p4, new_p5]
        if self.use_p6:
            res.append(new_p6)

        if self.for_detect:
            return res
        else:
            for idx in range(len(res)):
                new_res = F.interpolate(res[idx], size=(h, w), mode='nearest')
                res[idx] = new_res

            return torch.cat(res, dim=1)


if __name__ == '__main__':
    import os
    import sys
    add_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    sys.path.append(add_dir)
    add_dir = f'{os.path.sep}'.join(add_dir.split(os.path.sep)[:-1])
    sys.path.append(add_dir)
    add_dir = f'{os.path.sep}'.join(add_dir.split(os.path.sep)[:-1])
    sys.path.append(add_dir)

    from src.modules.backbone import resnet50

    backbone = resnet50(False)
    x = torch.rand(2, 3, 256, 256)

    backbone_output = backbone(x)

    print('='*15, 'Backbone Output Information', '='*15)
    for output in backbone_output:
        print(f'feature_map_size: {output.shape}')
        print('-'*58)
    print('='*58, '\n\n')


    fpn = FPN([256, 512, 1024, 2048], 256, True, True)
    fpn_output = fpn(backbone_output)
    print('=' * 15, 'FPN Ouput Information', '=' * 15)
    if isinstance(fpn_output, dict):
        for output in fpn_output:
            print(f'feature_map_size - {output} : {fpn_output[output].shape}')
            print('-' * 58)
    elif isinstance(fpn_output, list):
        for idx, output in enumerate(fpn_output):
            print(f'feature_map_size - fpn_output[{idx}] : {output.shape}')
            print('-' * 58)
    else:
        print(f'feature_map_size : {fpn_output.shape}')
    print('=' * 58)