import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from functools import partial

nonlinearity = partial(F.relu,inplace=True)

def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)

class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

   
class DecoderBlock(nn.Module):
    
    """
    Paramaters for Deconvolution were chosen to avoid artifacts, following
    link https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=False):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=16, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)
    
class CEBlock(nn.Module):
    
    """
    Our proposed context ensemble module. By coupling each pair of the decomposed tasks, the receptive fields
    enlarge their sizes, through the stacked dilated convolution with the dilated rates 1, 2, 4, 8, 16 in, respectively. 
    In the output of the module, the task-task CE modules are further paralleled which could obtain features with 
    dilated rates 1, 3, 7, 15, 31, 63.

    Attributes:
        out0_1: segment, class branch sharing
        out0_2: segment, scene branch sharing
        out1_2: scene, class branch sharing
    """  
    
    
    def __init__(self,channel):
        super(CEBlock, self).__init__()
        
        self.dilate0 = nn.ConvTranspose2d(channel, 256, kernel_size=4, stride=2, padding=1)       
        self.dilate1 = nn.Conv2d(256, 256, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(256, 256, kernel_size=3, dilation=2, padding=2)
        self.dilate4 = nn.Conv2d(256, 256, kernel_size=3, dilation=4, padding=4)
        self.dilate8 = nn.Conv2d(256, 256, kernel_size=3, dilation=8, padding=8)
        self.dilate16 = nn.Conv2d(256, 256, kernel_size=3, dilation=16, padding=16)
        
        self.task0_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)  
        self.task0_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)  
        self.task1_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)  

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                                      
    def forward(self, x):
        
        x = nonlinearity(self.dilate0(x))
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate4(dilate2_out))
        dilate4_out = nonlinearity(self.dilate8(dilate3_out))
        dilate5_out = nonlinearity(self.dilate16(dilate4_out))

        latent = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out + dilate5_out
        out0_1 = nonlinearity(self.task0_1(latent))
        out0_2 = nonlinearity(self.task0_2(latent))
        out1_2 = nonlinearity(self.task1_2(latent)) 
        
        return out0_1, out0_2, out1_2
    
    
class UNet16(nn.Module):
    
    """
    num_classes: param num_classes
    num_filters: param num_filters
    pretrained: param pretrained
                False - no pre-trained network used
                True - encoder pre-trained with VGG16
    """    
    
    def __init__(self, num_classes=1, num_filters=32, pretrained=False):
        
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.vgg16(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder[0],
                                   self.relu,
                                   self.encoder[2],
                                   self.relu)

        self.conv2 = nn.Sequential(self.encoder[5],
                                   self.relu,
                                   self.encoder[7],
                                   self.relu)

        self.conv3 = nn.Sequential(self.encoder[10],
                                   self.relu,
                                   self.encoder[12],
                                   self.relu,
                                   self.encoder[14],
                                   self.relu)

        self.conv4 = nn.Sequential(self.encoder[17],
                                   self.relu,
                                   self.encoder[19],
                                   self.relu,
                                   self.encoder[21],
                                   self.relu)

        self.conv5 = nn.Sequential(self.encoder[24],
                                   self.relu,
                                   self.encoder[26],
                                   self.relu,
                                   self.encoder[28],
                                   self.relu)

        self.center = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8)

        self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec3 = DecoderBlock(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec1), dim=1)
        else:
            x_out = self.final(dec1)

        return x_out
    

  
class TDSNet(nn.Module):
    
    """
    The image is processed through the encoder and task-task context ensemble to arrive at the latent space. 
    Then, the segmentation, class, and scene tasks are solved through individual decoders. A strong sync-regularization between 
    the segmentation and class tasks is further used to augment the coherence of multi-task learning.
    
    Attributes:
        num_classes: class task number
        num_scenes: scene task number
        num_filters: param num_filters
        pretrained: param pretrained
                False - no pre-trained network used
                True - encoder pre-trained with VGG16
    Returns:
        segment, class and scene output block
        
    """      
    
    def __init__(self, num_classes=1, num_scenes=1, num_filters=32, pretrained=False):
        
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)
        
        self.encoder = torchvision.models.vgg16(pretrained=pretrained).features
        self.relu = nn.ReLU(inplace=True)
        
        self.conv1 = nn.Sequential(self.encoder[0],
                                   self.relu,
                                   self.encoder[2],
                                   self.relu)

        self.conv2 = nn.Sequential(self.encoder[5],
                                   self.relu,
                                   self.encoder[7],
                                   self.relu)

        self.conv3 = nn.Sequential(self.encoder[10],
                                   self.relu,
                                   self.encoder[12],
                                   self.relu,
                                   self.encoder[14],
                                   self.relu)

        self.conv4 = nn.Sequential(self.encoder[17],
                                   self.relu,
                                   self.encoder[19],
                                   self.relu,
                                   self.encoder[21],
                                   self.relu)

        self.conv5 = nn.Sequential(self.encoder[24],
                                   self.relu,
                                   self.encoder[26],
                                   self.relu,
                                   self.encoder[28],
                                   self.relu)
        
        self.center = CEBlock(512)
                
        self.fc_scene = nn.Linear(512, 256)
        self.fc_scene_out = nn.Linear(256, num_scenes)
        self.avgpool_scene = nn.AdaptiveAvgPool2d(1)
        
        self.fc_class = nn.Linear(512, 256)
        self.fc_class_out = nn.Linear(256, num_classes)
        self.avgpool_class = nn.AdaptiveAvgPool2d(1)

        self.dec1 = DecoderBlock(512, 64, 32)
        self.dec0 = ConvRelu(32, 32)
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)
        
    def forward(self, x):
        
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))
                
        center0_1, center0_2, center1_2 = self.center(self.pool(conv5))     
        
        x = self.avgpool_class(torch.cat([center0_1, center1_2], 1))
        x = x.view(x.size(0), -1)
        out_class = self.fc_class(x)
        out_class = self.fc_class_out(out_class)
        
        x = self.avgpool_scene(torch.cat([center0_2, center1_2], 1))
        x = x.view(x.size(0), -1)
        out_scene = self.fc_scene(x)
        out_scene = self.fc_scene_out(out_scene)
        
        dec1 = self.dec1(torch.cat([center0_1, center0_2], 1))
        dec0 = self.dec0(dec1)        
        if self.num_classes > 1:
            out_seg = F.log_softmax(self.final(dec0), dim=1)
        else:
            out_seg = self.final(dec0)

        return out_seg, out_class, out_scene
    
