import MinkowskiEngine as ME
import torch.nn as nn
from .resnet import ResNetBase
from MinkowskiEngine.modules.resnet_block import BasicBlock
import pkg_resources

class PUResNetV2(ResNetBase):
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1
    def __init__(self, in_channels, out_channels, D=3,p=0.1,block=BasicBlock,planes=(32, 64, 128, 256, 256, 128, 96, 96),
                               layers=(2, 2, 2, 2, 2, 2, 2, 2)):
        PUResNetV2.BLOCK=block
        PUResNetV2.PLANES=planes
        PUResNetV2.LAYERS=layers
        PUResNetV2.INIT_DIM=planes[0]
        PUResNetV2.p=p
        ResNetBase.__init__(self, in_channels, out_channels, D)

    def network_initialization(self, in_channels, out_channels,D):
        
        self.dropout=ME.MinkowskiDropout(self.p)
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=3, dimension=D)

        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0],
                                       self.LAYERS[0])

        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1],
                                       self.LAYERS[1])

        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D)

        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2],
                                       self.LAYERS[2])

        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D)
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3],
                                       self.LAYERS[3])

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[4], kernel_size=3, stride=2, dimension=D)
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4],
                                       self.LAYERS[4])
        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[5], kernel_size=3, stride=2, dimension=D)
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5],
                                       self.LAYERS[5])
        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[6], kernel_size=3, stride=2, dimension=D)
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6],
                                       self.LAYERS[6])
        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[7], kernel_size=3, stride=2, dimension=D)
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7],
                                       self.LAYERS[7])

        self.final = ME.MinkowskiConvolution(
            self.PLANES[7] * self.BLOCK.expansion,
            out_channels,
            kernel_size=1,
            bias=True,
            dimension=D)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)
        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out= self.dropout(out)
        out_b1p2 = self.block1(out)
        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out= self.dropout(out)
        out_b2p4 = self.block2(out)
        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out= self.dropout(out)
        out_b3p8 = self.block3(out)
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out= self.dropout(out)
        out = self.block4(out)
        
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)
        out= self.dropout(out)
        out = ME.cat(out, out_b3p8)
        out = self.block5(out)
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)
        out= self.dropout(out)
        out = ME.cat(out, out_b2p4)
        out = self.block6(out)
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)
        out= self.dropout(out)
        out = ME.cat(out, out_b1p2)
        out = self.block7(out)
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)
        out= self.dropout(out)
        out = ME.cat(out, out_p1)
        out = self.block8(out)
        return self.final(out)
def get_trained_model():
    model_path = pkg_resources.resource_filename('puresnet', 'data/model90 70210.pt')
    p=0.11
    block=BasicBlock
    planes=(32,48,128,128,128,128,48,32)
    layers=(2,3,1,3,3,2,1,3)
    net = PUResNetV2(in_channels=18, out_channels=1,D=3,p=p,block=block,layers=layers,planes=planes)
    net.load_state_dict(ME.torch.load(model_path))
    return net.eval()
def get_model():
    p=0.11
    block=BasicBlock
    planes=(32,48,128,128,128,128,48,32)
    layers=(2,3,1,3,3,2,1,3)
    net = PUResNetV2(in_channels=18, out_channels=1,D=3,p=p,block=block,layers=layers,planes=planes)
    return net