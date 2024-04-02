import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, light=False):
        # ngf=64每层的基本通道数   n_blocks参差块6 图片大小256
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc  # 输入通道数 --> 3
        self.output_nc = output_nc  # 输出通道数 --> 3
        self.ngf = ngf  # 第一层卷积后的通道数 --> 64
        self.n_blocks = n_blocks  # 残差块数 --> 6
        self.img_size = img_size  # 图像size --> 256
        self.light = light  # 是否使用轻量级模型

        n_downsampling = 2  # 两层下采样

        mult = 2 ** n_downsampling  # mult =4
        # 上采样
        UpBlock0 = [nn.ReflectionPad2d(1),
                    nn.Conv2d(int(ngf * mult / 2), ngf * mult, kernel_size=3, stride=1, padding=0, bias=True),
                    ILN(ngf * mult),
                    nn.ReLU(True)]

        self.relu = nn.ReLU(True)

        # Gamma, Beta block --> 生成自适应 L-B Normalization(AdaILN)中的Gamma, Beta
        if self.light:  # 确定轻量级，FC使用的是两个256 --> 256的全连接层
            FC = [nn.Linear(ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True),
                  nn.Linear(ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True)]
        else:  # 不是轻量级，则下面的1024x1024 --> 256的全连接层和一个256 --> 256的全连接层
            FC = [nn.Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True),
                  nn.Linear(ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True)]
        # AdaILN中的Gamma, Beta   用来做图像增强
        self.gamma = nn.Linear(ngf * mult, ngf * mult, bias=False)
        self.beta = nn.Linear(ngf * mult, ngf * mult, bias=False)

        # Up-Sampling Bottleneck     --> 解码器中的自适应残差模块
        for i in range(n_blocks):  ##range（6）即：从0到6，不包含6，即0,1,2.。。。。
            setattr(self, 'UpBlock1_' + str(i + 1), ResnetAdaILNBlock(ngf * mult, use_bias=False))

        # Up-Sampling   --> 解码器中的上采样模块
        UpBlock2 = []
        # 上采样与编码器的下采样对应  两层上采样
        for i in range(n_downsampling):  # range（2） 0,1
            mult = 2 ** (n_downsampling - i)

            UpBlock2 += [nn.ReflectionPad2d(1),
                         nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0, bias=False),
                         ILN(int(ngf * mult / 2)),
                         nn.ReLU(True),
                         nn.Conv2d(int(ngf * mult / 2), int(ngf * mult / 2) * 4, kernel_size=1, stride=1, bias=True),
                         nn.PixelShuffle(2),
                         ILN(int(ngf * mult / 2)),  # 注:只有自适应残差块使用AdaILN
                         nn.ReLU(True)
                         ]
        # 最后一层卷积层，与最开始的卷积层对应
        UpBlock2 += [nn.ReflectionPad2d(3),
                     nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0, bias=False),
                     nn.Tanh()]

        self.FC = nn.Sequential(*FC)  # 生成gamma,beta的全连接层模块
        self.UpBlock0 = nn.Sequential(*UpBlock0)  # 解码器整个模块
        self.UpBlock2 = nn.Sequential(*UpBlock2)  # 只包含上采样后的模块，不包含残差块

    def forward(self, z):
        x = z  # 两层下采样  得到解码器的输出
        x = self.UpBlock0(x)  # 一层上采样

        if self.light:  # 判断是否轻量
            x_ = torch.nn.functional.adaptive_avg_pool2d(x, 1)  #
            x_ = self.FC(x_.view(x_.shape[0], -1))
        else:
            x_ = self.FC(x.view(x.shape[0], -1))
        gamma, beta = self.gamma(x_), self.beta(x_)  # 得到自适应gamma（自适应伽马变换  图像增强序列）和beta  gama和beta都用于图像增强

        for i in range(self.n_blocks):  # 将自适应gamma和beta送入到AdaILN
            x = getattr(self, 'UpBlock1_' + str(i + 1))(x, gamma, beta)

        out = self.UpBlock2(x)  # 通过2层上采样后的模块，得到生成结果

        return out


class ResnetBlock(nn.Module):  # 编码器中的残差块
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetAdaILNBlock(nn.Module):  # 解码器中的自适应残差块   包含6个
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm1 = adaILN(dim)
        self.relu1 = nn.ReLU(True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm2 = adaILN(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out + x


class adaILN(nn.Module):  # Adaptive Layer-Instance Normalization代码
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True, using_bn=False):
        super(adaILN, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.num_features = num_features

        if self.using_bn:
            self.rho = Parameter(torch.Tensor(1, num_features, 3))
            self.rho[:, :, 0].data.fill_(3)
            self.rho[:, :, 1].data.fill_(1)
            self.rho[:, :, 2].data.fill_(1)
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1, 1))
            self.running_mean.zero_()
            self.running_var.zero_()
        else:
            self.rho = Parameter(torch.Tensor(1, num_features, 2))
            self.rho[:, :, 0].data.fill_(3.2)
            self.rho[:, :, 1].data.fill_(1)

    def forward(self, input, gamma, beta):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        softmax = nn.Softmax(2)
        rho = softmax(self.rho)

        if self.using_bn:
            if self.training:
                bn_mean, bn_var = torch.mean(input, dim=[0, 2, 3], keepdim=True), torch.var(input, dim=[0, 2, 3],
                                                                                            keepdim=True)
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * bn_mean.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * bn_var.data)
                else:
                    self.running_mean.add_(bn_mean.data)
                    self.running_var.add_(bn_mean.data ** 2 + bn_var.data)
            else:
                bn_mean = torch.autograd.Variable(self.running_mean)
                bn_var = torch.autograd.Variable(self.running_var)
            out_bn = (input - bn_mean) / torch.sqrt(bn_var + self.eps)
            rho_0 = rho[:, :, 0]
            rho_1 = rho[:, :, 1]
            rho_2 = rho[:, :, 2]

            rho_0 = rho_0.view(1, self.num_features, 1, 1)
            rho_1 = rho_1.view(1, self.num_features, 1, 1)
            rho_2 = rho_2.view(1, self.num_features, 1, 1)
            rho_0 = rho_0.expand(input.shape[0], -1, -1, -1)
            rho_1 = rho_1.expand(input.shape[0], -1, -1, -1)
            rho_2 = rho_2.expand(input.shape[0], -1, -1, -1)
            out = rho_0 * out_in + rho_1 * out_ln + rho_2 * out_bn
        else:
            rho_0 = rho[:, :, 0]
            rho_1 = rho[:, :, 1]
            rho_0 = rho_0.view(1, self.num_features, 1, 1)
            rho_1 = rho_1.view(1, self.num_features, 1, 1)
            rho_0 = rho_0.expand(input.shape[0], -1, -1, -1)
            rho_1 = rho_1.expand(input.shape[0], -1, -1, -1)
            out = rho_0 * out_in + rho_1 * out_ln

        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)
        return out


class ILN(nn.Module):  # 没有加入自适应的Layer-Instance Normalization，  用于上采样
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True, using_bn=False):
        super(ILN, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.num_features = num_features

        if self.using_bn:
            self.rho = Parameter(torch.Tensor(1, num_features, 3))
            self.rho[:, :, 0].data.fill_(1)
            self.rho[:, :, 1].data.fill_(3)
            self.rho[:, :, 2].data.fill_(3)
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1, 1))
            self.running_mean.zero_()
            self.running_var.zero_()
        else:
            self.rho = Parameter(torch.Tensor(1, num_features, 2))
            self.rho[:, :, 0].data.fill_(1)
            self.rho[:, :, 1].data.fill_(3.2)

        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)

        softmax = nn.Softmax(2)
        rho = softmax(self.rho)

        if self.using_bn:
            if self.training:
                bn_mean, bn_var = torch.mean(input, dim=[0, 2, 3], keepdim=True), torch.var(input, dim=[0, 2, 3],
                                                                                            keepdim=True)
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * bn_mean.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * bn_var.data)
                else:
                    self.running_mean.add_(bn_mean.data)
                    self.running_var.add_(bn_mean.data ** 2 + bn_var.data)
            else:
                bn_mean = torch.autograd.Variable(self.running_mean)
                bn_var = torch.autograd.Variable(self.running_var)
            out_bn = (input - bn_mean) / torch.sqrt(bn_var + self.eps)
            rho_0 = rho[:, :, 0]
            rho_1 = rho[:, :, 1]
            rho_2 = rho[:, :, 2]

            rho_0 = rho_0.view(1, self.num_features, 1, 1)
            rho_1 = rho_1.view(1, self.num_features, 1, 1)
            rho_2 = rho_2.view(1, self.num_features, 1, 1)
            rho_0 = rho_0.expand(input.shape[0], -1, -1, -1)
            rho_1 = rho_1.expand(input.shape[0], -1, -1, -1)
            rho_2 = rho_2.expand(input.shape[0], -1, -1, -1)
            out = rho_0 * out_in + rho_1 * out_ln + rho_2 * out_bn
        else:
            rho_0 = rho[:, :, 0]
            rho_1 = rho[:, :, 1]
            rho_0 = rho_0.view(1, self.num_features, 1, 1)
            rho_1 = rho_1.view(1, self.num_features, 1, 1)
            rho_0 = rho_0.expand(input.shape[0], -1, -1, -1)
            rho_1 = rho_1.expand(input.shape[0], -1, -1, -1)
            out = rho_0 * out_in + rho_1 * out_ln

        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)
        return out


class Discriminator(nn.Module,):
    def __init__(self, input_nc, ndf=64, n_layers=7):  # 7层     256    64
        super(Discriminator, self).__init__()
        model = [nn.ReflectionPad2d(1),  # 第一层下采样, 尺寸减半(128)，通道数为64
                 nn.utils.spectral_norm(
                     nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=True)),
                 nn.LeakyReLU(0.2, True)]  # 1+3*2^0 =4

        for i in range(1, 2):  # 1+3*2^0 + 3*2^1 =10     一层下采样 尺寸再缩4倍(64)，通道数为128
            mult = 2 ** (i - 1)
            model += [nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                          nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.LeakyReLU(0.2, True)]

        mult = 2 ** (1)
        self.fc = nn.utils.spectral_norm(nn.Linear(ndf * mult * 2, 1, bias=False))
        # 经过一个1x1卷积和激活函数层得到黄色的a1...an特征图
        self.conv1x1 = nn.Conv2d(ndf * mult * 2, ndf * mult, kernel_size=1, stride=1, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.lamda = nn.Parameter(torch.zeros(1))

        Dis0_0 = []
        # range（3）即：从0到3，不包含3，即0,1,2   range(1,3) 即：从1到3，不包含3，即1,2
        # range（1,3,2）即：从1到3，每次增加2，因为1+2=3，所以输出只有1
        for i in range(2, n_layers - 4):  # range(2,3)    执行一次i=2      1+3*2^0 + 3*2^1 + 3*2^2 =22
            mult = 2 ** (i - 1)  # i=2   mult = 2
            Dis0_0 += [nn.ReflectionPad2d(1),
                       nn.utils.spectral_norm(
                           nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                       nn.LeakyReLU(0.2, True)]

        mult = 2 ** (n_layers - 4 - 1)  # mult = 4

        Dis0_1 = [nn.ReflectionPad2d(1),  # 1+3*2^0 + 3*2^1 + 3*2^2 +3*2^3 = 46
                  nn.utils.spectral_norm(
                      nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, padding=0, bias=True)),
                  nn.LeakyReLU(0.2, True)]
        mult = 2 ** (n_layers - 4)  # mult = 8
        self.conv0 = nn.utils.spectral_norm(  # 1+3*2^0 + 3*2^1 + 3*2^2 +3*2^3 + 3*2^3= 70
            nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))

        Dis1_0 = []
        for i in range(n_layers - 4,
                       n_layers - 2):  # 1+3*2^0 + 3*2^1 + 3*2^2 + 3*2^3=46, 1+3*2^0 + 3*2^1 + 3*2^2 +3*2^3 +3*2^4 = 94
            mult = 2 ** (i - 1)
            Dis1_0 += [nn.ReflectionPad2d(1),
                       nn.utils.spectral_norm(
                           nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                       nn.LeakyReLU(0.2, True)]

        mult = 2 ** (n_layers - 2 - 1)
        Dis1_1 = [nn.ReflectionPad2d(1),  # 1+3*2^0 + 3*2^1 + 3*2^2 +3*2^3 +3*2^4 + 3*2^5= 94 + 96 = 190
                  nn.utils.spectral_norm(
                      nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, padding=0, bias=True)),
                  nn.LeakyReLU(0.2, True)]
        mult = 2 ** (n_layers - 2)
        self.conv1 = nn.utils.spectral_norm(  # 1+3*2^0 + 3*2^1 + 3*2^2 +3*2^3 +3*2^4 + 3*2^5 + 3*2^5 = 286
            nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))

        # self.attn = Self_Attn( ndf * mult)
        self.pad = nn.ReflectionPad2d(1)

        self.model = nn.Sequential(*model)
        self.Dis0_0 = nn.Sequential(*Dis0_0)
        self.Dis0_1 = nn.Sequential(*Dis0_1)
        self.Dis1_0 = nn.Sequential(*Dis1_0)
        self.Dis1_1 = nn.Sequential(*Dis1_1)

    def forward(self, input):
        x = self.model(input)

        x_0 = x  # 经过两层卷积后变成64*64

        # ------------------5.20---------------------

        feature_out = x  # 特征图

        #feature_out = self.adaptive_pool(x)
        #feature_out = feature_out.permute(0, 2, 3, 1)  # 将张量按顺序存储

        # ------------------5.20---------------------

        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)  # 全局平均池化
        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)  # 全局最大池化
        x = torch.cat([x, x], 1)
        cam_logit = torch.cat([gap, gmp], 1)  # 结合两种权重后的特征图
        # 热力图heatmap CAM由于这是基于分类问题的一种可视化技术，并且只有将全连接层改为全局平均池化才能较好的保存图像的空间信息
        cam_logit = self.fc(cam_logit.view(cam_logit.shape[0], -1))
        weight = list(self.fc.parameters())[0]
        x = x * weight.unsqueeze(2).unsqueeze(3)  # 权重计算
        x = self.conv1x1(x)

        x = self.lamda * x + x_0  # 计算attention层
        # print("lamda:",self.lamda)

        x = self.leaky_relu(x)

        heatmap = torch.sum(x, dim=1, keepdim=True)  # 热力图

        z = x
        # feature_out = z

        x0 = self.Dis0_0(x)
        x1 = self.Dis1_0(x0)
        x0 = self.Dis0_1(x0)
        x1 = self.Dis1_1(x1)
        x0 = self.pad(x0)
        x1 = self.pad(x1)
        out0 = self.conv0(x0)
        out1 = self.conv1(x1)
        # 改动
        return out0, out1, cam_logit, heatmap, z, feature_out



#  ————————————————————————————————添加 5.13——————————————————————————————————————————————————


class Attention(nn.Module):
    def __init__(self, feature_out, decoder_dim, converge_vector_channel, converge_vector_dim, attention_dim=256):
        super(Attention, self).__init__()

        self.encoder_att = nn.Linear(feature_out, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.converge = nn.Linear(converge_vector_channel, converge_vector_dim)
        self.converge_att = nn.Linear(converge_vector_dim, attention_dim)
        self.tanh = nn.Tanh()
        self.full_att = nn.Linear(attention_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feature_out, decoder_hidden, converge_vector):
        att1 = self.encoder_att(feature_out)
        att2 = self.decoder_att(decoder_hidden)

        if sum(sum(converge_vector)).item() != 0:
            converge_vector = self.converge(converge_vector)
            att3 = self.converge_att(converge_vector)
            att = self.full_att(self.tanh(att1 + att2.unsqueeze(1) + att3.unsqueeze(1))).squeeze(2)
        else:
            att = self.full_att(self.tanh(att1 + att2.unsqueeze(1))).squeeze(2)

        # att size (batch_size, encoder_feature_length)
        alpha = self.softmax(att)
        context = (feature_out * alpha.unsqueeze(2)).sum(dim=1)

        return context, alpha

class  con_feature(nn.Module,):
    def __init__(self,feature_out):
        super(con_feature, self).__init__()
        ndf = 256
        model = [nn.ReflectionPad2d(1),  # 第一层下采样, 尺寸减半(128)，通道数为64
                 nn.utils.spectral_norm(
                     nn.Conv2d(feature_out, ndf, kernel_size=4, stride=2, padding=3, bias=True)),
                 nn.LeakyReLU(0.2, True)]  # 1+3*2^0 =4

        for i in range(1, 2):  # 1+3*2^0 + 3*2^1 =10     一层下采样 尺寸再缩4倍(64)，通道数为128
            # n = 32
            mult = 2 ** (i - 1)
            model += [nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                          nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=3, bias=True)),
                      nn.LeakyReLU(0.2, True)]

        self.model = nn.Sequential(*model)

    def forward(self, feature_out):
        x_0 = self.model(feature_out)
        feature_out = x_0
        return feature_out


class Decoderrnn(nn.Module):

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=512, encoder_fsize=10,
                 converge_vector_dim=256, dropout=0.5, embedding_dropout=0.1):
        super(Decoderrnn, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.encoder_fsize = encoder_fsize
        self.encoder_fl = encoder_fsize * encoder_fsize
        self.converge_vector_dim = converge_vector_dim
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout

        self.attention = Attention(self.encoder_dim, self.decoder_dim, self.encoder_fl, self.converge_vector_dim,
                                   self.attention_dim)
        self.embeddimg = nn.Embedding(vocab_size, self.embed_dim)
        self.embedding_dropout = nn.Dropout(p=self.embedding_dropout)
        self.dropout = nn.Dropout(p=self.dropout)
        self.gru1 = nn.GRUCell(self.embed_dim, decoder_dim, bias=True)
        self.gru2 = nn.GRUCell(self.encoder_dim, decoder_dim, bias=True)
        self.s = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.fc = nn.Linear(self.decoder_dim, self.vocab_size)
        self.init_weights()

        self.con_feature = con_feature(int(self.encoder_dim/4))
        # --------------------7.9--------

    def con_f(self , feature_out):
        feature_out = self.con_feature(feature_out)
        return feature_out

    def init_weights(self):
        self.embeddimg.weight.data.uniform_(-0.1, 0.1)

        self.fc.bias.data.fill_(0)

        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)

    def init_hidden_state(self, feature_out):
        mean_encoder_out = feature_out.mean(dim=1)
        # s = self.s(mean_encoder_out)
        s = self.s(mean_encoder_out)
        return s

    def decode_step(self, embedding_word, s, feature_out, converge_vector):

        # gru cell
        st_hat = self.gru1(embedding_word, s)
        context, alpha = self.attention(feature_out, s, converge_vector)
        st = self.gru2(context, st_hat)

        # sum of history converge vector
        converge_vector = converge_vector + alpha

        # embedding predict word
        preds = self.fc(self.dropout(st))
        preds_words = preds.topk(1)[1].squeeze()
        embedding_word = self.embeddimg(preds_words)
        embedding_word = self.embedding_dropout(embedding_word)
        embedding_word = embedding_word.view(-1, self.embed_dim)

        return embedding_word, st, converge_vector, preds, alpha


    def forward(self, feature_out, encoded_captions, caption_lengths):



        feature_out = self.con_f(feature_out)
        #  ---------------------6.8------------

        feature_out = feature_out.permute(0, 2, 3, 1)

        batch_size = feature_out.size(0)
        encoder_dim = feature_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image  合并图层
        feature_out = feature_out.view(batch_size, -1, encoder_dim)
        num_pixels = feature_out.size(1)
        # ----------------------------改动--------------------
        # sort input data by decreasing lengths  分类图像长度
        # caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)

        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)

        feature_out = feature_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        # ----------------------------改动--------------------
        # embedding
        start_word = encoded_captions[:, 0]
        embedding_word = self.embeddimg(start_word)
        embedding_word = self.embedding_dropout(embedding_word)

        # initialize GRU state
        # s = self.init_hidden_state(feature_out)
        s = self.init_hidden_state(feature_out)

        # remove <eos> during decoding
        # decode_lengths = (caption_lengths -1).tolist()
        decode_lengths = caption_lengths.tolist()

        # create tensors to hold word prediction scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels)

        # decode by time t
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            embedding_word, s, converge_vector, preds, alpha = self.decode_step(embedding_word[:batch_size_t],
                                                                                s[:batch_size_t],
                                                                                feature_out[:batch_size_t],
                                                                                converge_vector=torch.zeros(
                                                                                    batch_size_t, num_pixels).to('cuda')
                                                                                if t == 0 else converge_vector[
                                                                                               :batch_size_t].to('cuda'))
            predictions[:batch_size_t, t] = preds
            alphas[:batch_size_t, t] = alpha

        # return encoded_captions, decode_lengths
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind


class  delta_feature(nn.Module,):
    def __init__(self,feature_out):
        super(delta_feature, self).__init__()
        ndf = 128
        model = [nn.ReflectionPad2d(1),  # 第一层下采样, 尺寸减半(128)，通道数为64
                 nn.utils.spectral_norm(
                 nn.Conv2d(feature_out, ndf, kernel_size=4, stride=4, padding=0, bias=True)),
                 nn.LeakyReLU(0.2, True)]  # 1+3*2^0 =4

        for i in range(1, 2):  # 1+3*2^0 + 3*2^1 =10     一层下采样 尺寸再缩4倍(64)，通道数为128
            # n = 32
            # mult = 2 ** (i - 1)
            model += [nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                          nn.Conv2d(128, 128, kernel_size=4, stride=4, padding=0, bias=True)),
                      nn.LeakyReLU(0.2, True)]

        self.model = nn.Sequential(*model)

    def forward(self, feature_out):
        x_0 = self.model(feature_out)
        feature_out = x_0
        return feature_out

class DeltaEncoder(nn.Module):
    def __init__(self, feature_dim=2048, encoder_size=[8192], z_dim=16, dropout=0.5, dropout_input=0.0, leak=0.2):
        super(DeltaEncoder, self).__init__()
        self.first_linear = nn.Linear(feature_dim*2, encoder_size[0])
        # self.first_linear = nn.Linear(feature_dim*2, encoder_size[0])

        linear = []
        for i in range(len(encoder_size) - 1):
            linear.append(nn.Linear(encoder_size[i], encoder_size[i+1]))
            linear.append(nn.LeakyReLU(leak))
            linear.append(nn.Dropout(dropout))

        self.linear = nn.Sequential(*linear)
        self.final_linear = nn.Linear(encoder_size[-1], z_dim)
        self.lrelu = nn.LeakyReLU(leak)
        self.dropout_input = nn.Dropout(dropout_input)
        self.dropout = nn.Dropout(dropout)

        self.delta_feature = delta_feature(int(512/ 4))

    # --------------------7.9--------

    def delta_f(self, feature_out):
        feature_out = self.delta_feature(feature_out)
        return feature_out

    def forward(self, features, reference_features):

        features =self.delta_f(features)
        # pred_noise_X = delta_feature(features)
        reference_features = self.delta_f(reference_features)
        # pred_noise_Y = delta_feature(reference_features)

        features = features.view(features.size(0), -1)
        reference_features = reference_features.view(reference_features.size(0), -1)



        features = self.dropout_input(features)
        # features = self.dropout_input(features)
        x = torch.cat([features, reference_features], 1)
        # x = torch.cat([features, reference_features], 1)

        # print("features shape is:", features.shape, reference_features.shape)
        # print(x.shape)

        x = self.first_linear(x)
        x = self.linear(x)

        x = self.final_linear(x)

        return x,features,reference_features


class DeltaDecoder(nn.Module):
    def __init__(self, feature_dim=2048, decoder_size=[8192], z_dim=16, dropout=0.5, leak=0.2):
        super(DeltaDecoder, self).__init__()
        self.first_linear = nn.Linear(z_dim+feature_dim, decoder_size[0])

        linear = []
        for i in range(len(decoder_size) - 1):
            linear.append(nn.Linear(decoder_size[i], decoder_size[i+1]))
            linear.append(nn.LeakyReLU(leak))
            linear.append(nn.Dropout(dropout))

        self.linear = nn.Sequential(*linear)

        self.final_linear = nn.Linear(decoder_size[-1], feature_dim) #【1024,2048】
        self.lrelu = nn.LeakyReLU(leak)
        self.dropout = nn.Dropout(dropout)

    def forward(self, reference_features, code):
        x = torch.cat([reference_features, code], 1)

        x = self.first_linear(x)
        x = self.linear(x)

        x = self.final_linear(x) #【1024,2048】

        return x



cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'vgg19cut': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'N'],
}

class GuidingNet(nn.Module):
    def __init__(self, img_size=256, output_k={'cont': 128, 'disc': 10}):
        super(GuidingNet, self).__init__()
        self.features = make_layers(cfg['vgg11'], True)
        self.disc = nn.Linear(512, output_k['disc'])
        self.cont = nn.Linear(512, output_k['cont'])

    def forward(self, x):
        x = self.features(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        flat = x.view(x.size(0), -1)
        cont = self.cont(flat)

        return cont


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]
            else:
                layers += [conv2d, nn.ReLU(inplace=False)]
            in_channels = v
    return nn.Sequential(*layers)