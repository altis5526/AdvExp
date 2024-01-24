from email.mime import base
import torch
import torch.nn as nn
import torch.nn.functional as F
from padding_same_conv import Conv2d
import torchvision.models as models
import functools
from torch.autograd import Function
from torchvision.transforms.functional import equalize
from spectral_norm import SNConv2d
from spectral_norm import SNLinear
import torch.nn.functional as F
from iunets import iUNet


# Decide which device we want to run on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        
        return grad_output.neg() * ctx.alpha, None



class mimic_AutoEncoder(nn.Module):
    def __init__(self, input_nc=3, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(mimic_AutoEncoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        use_bias = True
        # construct unet structure
        self.downsample_0 = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias)

        self.downRelu_1 = nn.LeakyReLU(0.2, True)
        self.downSample_1 = nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.downNorm_1 = norm_layer(ngf * 2)

        self.downRelu_2 = nn.LeakyReLU(0.2, True)
        self.downSample_2 = nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.downNorm_2 = norm_layer(ngf * 4)

        self.downRelu_3 = nn.LeakyReLU(0.2, True)
        self.downSample_3 = nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.downNorm_3 = norm_layer(ngf * 8)

        self.innerLeakyRelu = nn.LeakyReLU(0.2, True)
        self.innerDownSample = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.innerRelu = nn.ReLU(True)
        innerUpSample = []
        innerUpSample.append(nn.Upsample(scale_factor = 2, mode='bilinear'))
        innerUpSample.append(nn.ReflectionPad2d((2, 1, 2, 1)))
        innerUpSample.append(nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=1, padding=0, bias=use_bias))
        self.innerUpSample = nn.Sequential(*innerUpSample)

        self.innerNorm = norm_layer(ngf * 8)

        self.upRelu_3 = nn.ReLU(True)
        upSample_3 = []
        upSample_3.append(nn.Upsample(scale_factor = 2, mode='bilinear'))
        upSample_3.append(nn.ReflectionPad2d((2, 1, 2, 1)))
        upSample_3.append(nn.Conv2d(ngf * 16, ngf * 4, kernel_size=4, stride=1, padding=0, bias=use_bias))
        self.upSample_3 = nn.Sequential(*upSample_3)
        self.upNorm_3 = norm_layer(ngf * 4)

        self.upRelu_2 = nn.ReLU(True)
        upSample_2 = []
        upSample_2.append(nn.Upsample(scale_factor = 2, mode='bilinear'))
        upSample_2.append(nn.ReflectionPad2d((2, 1, 2, 1)))
        upSample_2.append(nn.Conv2d(ngf * 8, ngf * 2, kernel_size=4, stride=1, padding=0, bias=use_bias))
        self.upSample_2 = nn.Sequential(*upSample_2)
        self.upNorm_2 = norm_layer(ngf * 2)

        self.upRelu_1 = nn.ReLU(True)
        upSample_1 = []
        upSample_1.append(nn.Upsample(scale_factor = 2, mode='bilinear'))
        upSample_1.append(nn.ReflectionPad2d((2, 1, 2, 1)))
        upSample_1.append(nn.Conv2d(ngf * 4, ngf, kernel_size=4, stride=1, padding=0, bias=use_bias))
        self.upSample_1 = nn.Sequential(*upSample_1)
        self.upNorm_1 = norm_layer(ngf)

        self.upRelu_0 = nn.ReLU(True)
        upSample_0 = []
        upSample_0.append(nn.Upsample(scale_factor = 2, mode='bilinear'))
        upSample_0.append(nn.ReflectionPad2d((2, 1, 2, 1)))
        upSample_0.append(nn.Conv2d(ngf * 2, 1, kernel_size=4, stride=1, padding=0, bias=use_bias))
        self.upSample_0 = nn.Sequential(*upSample_0)

        ## initialize bias
        nn.init.normal_(self.upSample_0[-1].bias, mean=3, std=1)

        self.activation = nn.Sigmoid()

    def forward(self, input):
        # assume input image size = 224
        x_down_0 = self.downsample_0(input) # (ngf, 112, 112)

        x_down_1 = self.downNorm_1(self.downSample_1(self.downRelu_1(x_down_0))) # (ngf*2, 56, 56)
        x_down_2 = self.downNorm_2(self.downSample_2(self.downRelu_2(x_down_1))) # (ngf*4, 28, 28)
        x_down_3 = self.downNorm_3(self.downSample_3(self.downRelu_3(x_down_2))) # (ngf*8, 14, 14)

        latent = self.innerDownSample(self.innerLeakyRelu(x_down_3)) # (ngf*8, 7, 7)

        x = self.innerNorm(self.innerUpSample(self.innerRelu(latent))) # (ngf*8, 14, 14)

        x_up_3 = self.upNorm_3(self.upSample_3(self.upRelu_3(torch.cat([x, x_down_3], 1)))) # (ngf*4, 28, 28)
        x_up_2 = self.upNorm_2(self.upSample_2(self.upRelu_2(torch.cat([x_up_3, x_down_2], 1)))) # (ngf*2, 56, 56)
        x_up_1 = self.upNorm_1(self.upSample_1(self.upRelu_1(torch.cat([x_up_2, x_down_1], 1)))) # (ngf, 112, 112)

        encoded_image = self.activation(self.upSample_0(self.upRelu_0(torch.cat([x_up_1, x_down_0], 1)))) # (3, 224, 224)
        
        return torch.mul(input, encoded_image), encoded_image, latent

class Combined_to_oneNet(nn.Module):
    def __init__(self, spurious_weight, spurious_oc=2, num_verb=14, adv_lambda=1.0, grad_reverse=True, spurious_sigmoid=False, pretrained=True, intermediate=False):
        super(Combined_to_oneNet, self).__init__()
        self.num_verb = num_verb
        self.base_network = models.resnet50(pretrained = pretrained)

        norm_layer = 'batch'
        use_dropout = False
        norm_layer = get_norm_layer(norm_type=norm_layer)
        self.autoencoder = mimic_AutoEncoder(3, 64, \
            norm_layer=norm_layer, use_dropout=use_dropout)

        output_size = self.num_verb
        self.finalLayer = nn.Linear(self.base_network.fc.in_features, output_size)
        self.activation = nn.Sigmoid()
        self.grad_reverse = grad_reverse
        self.spurious_sigmoid = spurious_sigmoid
        self.intermediate = intermediate

        if self.grad_reverse == True:
            self.adv_component = spurious_classifier(oc=spurious_oc)
            pretrained_gender_classifier_path = spurious_weight
            gender_clssifier_checkpoint = torch.load(pretrained_gender_classifier_path)
            self.adv_component.load_state_dict(gender_clssifier_checkpoint["model_state_dict"])
            print("Loaded pretrained classifier from {}".format(pretrained_gender_classifier_path))
            self.adv_lambda = adv_lambda
    
    def forward(self, image):
        autoencoded_image, mask, latent = self.autoencoder(image)

        x = self.base_network.conv1(autoencoded_image)
        x = self.base_network.bn1(x)
        x = self.base_network.relu(x)
        conv1_feature = self.base_network.maxpool(x)

        layer1_feature = self.base_network.layer1(conv1_feature)
        layer2_feature = self.base_network.layer2(layer1_feature)
        layer3_feature = self.base_network.layer3(layer2_feature)
        layer4_feature = self.base_network.layer4(layer3_feature)

        final_feature = self.base_network.avgpool(layer4_feature)
        final_feature = final_feature.view(final_feature.size(0), -1)

        preds = self.finalLayer(final_feature)
        preds = self.activation(preds)        

        # adv component forward pass
        if self.grad_reverse == True:
            adv_feature = ReverseLayerF.apply(autoencoded_image, self.adv_lambda)
            adv_preds = self.adv_component(adv_feature)
            if self.spurious_sigmoid == True:
                adv_preds = self.activation(adv_preds)
            return preds, adv_preds, autoencoded_image, mask

        else:
            if self.intermediate == True:
                return preds, autoencoded_image, mask, final_feature
            return preds, autoencoded_image, mask


class spurious_classifier(nn.Module):
    def __init__(self, oc=2, indpt_grad_reverse=False, in_c = 3, pretrained = True):
        super(spurious_classifier, self).__init__()
        self.classifier = models.resnet18(pretrained=pretrained)
        self.classifier.conv1 = Conv2d(in_c, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.classifier.fc = nn.Linear(in_features=512, out_features=oc, bias=True)
        self.activation = nn.Softmax(dim=1)
        self.indpt_grad_reverse = indpt_grad_reverse

    def forward(self, x):
        output = self.classifier(x)
        if self.indpt_grad_reverse == True:
            norm_output = self.activation(output)
            return output, norm_output

        return output
    
class spurious_classifier_race(nn.Module):
    def __init__(self, oc=2, indpt_grad_reverse=False, in_c = 3, pretrained = True):
        super(spurious_classifier_race, self).__init__()
        self.classifier = models.resnet34(pretrained=pretrained)
        self.classifier.conv1 = Conv2d(in_c, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.classifier.fc = nn.Linear(in_features=512, out_features=oc, bias=True)
        self.activation = nn.Softmax(dim=1)
        self.indpt_grad_reverse = indpt_grad_reverse

    def forward(self, x):
        output = self.classifier(x)
        if self.indpt_grad_reverse == True:
            norm_output = self.activation(output)
            return output, norm_output

        return output

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

## Not combining net
class target_classifier_new(nn.Module):
    def __init__(self, oc=14, in_c=3, pretrained=True, intermediate=False):
        super(target_classifier_new, self).__init__()
        self.base_network = models.resnet50(pretrained = pretrained)
        self.base_network.conv1 = Conv2d(in_c, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.oc = oc
        output_size = self.oc
        self.finalLayer = nn.Linear(self.base_network.fc.in_features, output_size)
        self.activation = nn.Sigmoid()
        self.intermediate = intermediate
    
    def forward(self, x):
        x = self.base_network.conv1(x)
        x = self.base_network.bn1(x)
        x = self.base_network.relu(x)
        conv1_feature = self.base_network.maxpool(x)

        layer1_feature = self.base_network.layer1(conv1_feature)
        layer2_feature = self.base_network.layer2(layer1_feature)
        layer3_feature = self.base_network.layer3(layer2_feature)
        layer4_feature = self.base_network.layer4(layer3_feature)

        final_feature = self.base_network.avgpool(layer4_feature)
        intermediate_preds = final_feature.view(final_feature.size(0), -1)

        preds = self.finalLayer(intermediate_preds)
        preds = self.activation(preds)

        if self.intermediate == True:
            return intermediate_preds, preds
        else:
            return preds

class Autoencoder_new(nn.Module):
    def __init__(self, num_verb=14):
        super(Autoencoder_new, self).__init__()
        self.num_verb = num_verb
        # self.base_network = models.resnet50(pretrained = True)

        norm_layer = 'batch'
        use_dropout = False
        norm_layer = get_norm_layer(norm_type=norm_layer)
        self.autoencoder = mimic_AutoEncoder(3, 64, \
            norm_layer=norm_layer, use_dropout=use_dropout)

        output_size = self.num_verb
        # self.finalLayer = nn.Linear(self.base_network.fc.in_features, output_size)

    def forward(self, image):
        autoencoded_image, mask, latent = self.autoencoder(image)

        return autoencoded_image, mask


class LnL_convnet(nn.Module):
    def __init__(self, oc_target, input_nc=3, ngf=16):
        super(LnL_convnet,self).__init__()
        use_bias = True
        # construct unet structure
        self.downsample_0 = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias)

        self.downRelu_1 = nn.LeakyReLU(0.2, True)
        self.downSample_1 = nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.downNorm_1 = nn.BatchNorm2d(ngf * 2)

        self.downRelu_2 = nn.LeakyReLU(0.2, True)
        # self.encoder = mimic_encoder(input_nc=3, ngf=64,
        #                 norm_layer=nn.BatchNorm2d, use_dropout=False)
        # self.decoder = mimic_decoder(output_nc=3, ngf=64,
        #                 norm_layer=nn.BatchNorm2d, use_dropout=False)
        self.target_classify = target_classifier_new(oc=oc_target, in_c=ngf * 2, pretrained=False)
        

    def forward(self, x):
        x_down_0 = self.downsample_0(x) # (ngf, 112, 112)

        latent = self.downNorm_1(self.downSample_1(self.downRelu_1(x_down_0))) # (ngf*2, 56, 56)
        # latent_map = self.decoder(x_down_0, x_down_1, x_down_2, x_down_3, latent)
        out = self.target_classify(latent)

        return latent, out
    
class LnL_autoencoder(nn.Module):
    def __init__(self, oc_target):
        super(LnL_autoencoder, self).__init__()
        norm_layer = 'batch'
        use_dropout = False
        norm_layer = get_norm_layer(norm_type=norm_layer)
        self.autoencoder = mimic_AutoEncoder(3, 64, \
            norm_layer=norm_layer, use_dropout=use_dropout)
        self.target_classify = target_classifier_new(oc=oc_target, in_c=3, pretrained=True)
        
    def forward(self,x):
        autoencoded_image, mask, latent = self.autoencoder(x)
        out = self.target_classify(autoencoded_image)
        
        return autoencoded_image, out
