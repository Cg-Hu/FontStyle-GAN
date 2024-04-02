import time
import itertools
from dataset import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from networks import *
from utils import *
from glob import glob
from tensorboardX import SummaryWriter
# import torch.utils.tensorboard as tensorboardX
from thop import profile
from thop import clever_format

# _______tianjia  5.19______
from torch.nn.utils.rnn import pack_padded_sequence
import torch
import torch.nn as nn

# _______tianjia______


class NICE(object):
    def __init__(self, args):
        self.light = args.light

        if self.light:
            self.model_name = 'NICE_light'
        else:
            self.model_name = 'NICE'

        self.result_dir = args.result_dir
        self.dataset = args.dataset  # 数据集名称

        self.iteration = args.iteration
        self.decay_flag = args.decay_flag

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.ch = args.ch

        """ Weight """
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.recon_weight = args.recon_weight

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis

        self.img_size = args.img_size
        self.img_ch = args.img_ch  # image channel number

        self.device = args.device
        self.benchmark_flag = args.benchmark_flag
        self.resume = args.resume

        self.start_iter = 1

        self.fid = 1000
        self.fid_A = 1000
        self.fid_B = 1000
        if torch.backends.cudnn.enabled and self.benchmark_flag:
            print('set benchmark !')
            torch.backends.cudnn.benchmark = True

        print()

        print("##### Information #####")
        print("# light : ", self.light)
        print("# dataset : ", self.dataset)
        print("# batch_size : ", self.batch_size)
        print("# iteration per epoch : ", self.iteration)
        print("# the size of image : ", self.img_size)
        print("# the size of image channel : ", self.img_ch)
        print("# base channel number per layer : ", self.ch)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)

        print()

        print("##### Discriminator #####")
        print("# discriminator layers : ", self.n_dis)

        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# cycle_weight : ", self.cycle_weight)
        print("# recon_weight : ", self.recon_weight)

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ DataLoader """
        train_transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(), # sujifanzhuan shuiping  bs you
            # transforms.Resize((self.img_size, self.img_size)),
            transforms.Resize(
                (self.img_size + 30, self.img_size + 30)),  # Resize图片
            transforms.RandomCrop(self.img_size),  # suijicaijian  bs you
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        # 数据加载
        #   ----------------------------- 改动 6.3 -----------------------------------------------------
        self.trainA = ImageFolder(os.path.join('dataset', self.dataset, 'trainA.txt'),
                                  vocab_path='radical_alphabet.txt',
                                  dict_path='char2seq_dict.pkl', transform=train_transform)
        self.trainB = ImageFolder(os.path.join('dataset', self.dataset, 'trainB.txt'),
                                  vocab_path='radical_alphabet.txt',
                                  dict_path='char2seq_dict.pkl', transform=train_transform)
        self.testA = ImageFolder(os.path.join('dataset', self.dataset, 'testA.txt'), vocab_path='radical_alphabet.txt',
                                 dict_path='char2seq_dict.pkl', transform=test_transform)
        self.testB = ImageFolder(os.path.join('dataset', self.dataset, 'testB.txt'), vocab_path='radical_alphabet.txt',
                                 dict_path='char2seq_dict.pkl', transform=test_transform)

        self.trainA_loader = DataLoader(
            self.trainA, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        self.trainB_loader = DataLoader(
            self.trainB, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        self.testA_loader = DataLoader(
            self.testA, batch_size=1, shuffle=False, pin_memory=True)
        self.testB_loader = DataLoader(
            self.testB, batch_size=1, shuffle=False, pin_memory=True)

        """ Define Generator, Discriminator """
        self.gen2B = ResnetGenerator(input_nc=self.img_ch, output_nc=self.img_ch, ngf=self.ch, n_blocks=self.n_res,
                                     img_size=self.img_size, light=self.light).to(self.device)
        self.gen2A = ResnetGenerator(input_nc=self.img_ch, output_nc=self.img_ch, ngf=self.ch, n_blocks=self.n_res,
                                     img_size=self.img_size, light=self.light).to(self.device)
        self.disA = Discriminator(
            input_nc=self.img_ch, ndf=self.ch, n_layers=self.n_dis).to(self.device)
        self.disB = Discriminator(
            input_nc=self.img_ch, ndf=self.ch, n_layers=self.n_dis).to(self.device)

        """ Define rnn """
        # ———————————————————————————————————————————rnn-------------------------------#
        # ———————————————————————————————————————改动 5.19————————————————————————————————————----
        vocab_path = 'radical_alphabet.txt'
        # 把预先处理的笔画读进来
        word_map = open(vocab_path, encoding='utf-8').readlines()[0]
        word_map = word_map + 'sep'
        vocab_size = len(word_map)  # 读进来的笔画长度有476个

        #  rnn decoder
        self.rnn2A = Decoderrnn(attention_dim=256, embed_dim=256, decoder_dim=256, vocab_size=vocab_size,
                                encoder_dim=512, encoder_fsize=20,
                                converge_vector_dim=256, dropout=0.5, embedding_dropout=0.5).to(self.device)
        self.rnn2B = Decoderrnn(attention_dim=256, embed_dim=256, decoder_dim=256, vocab_size=vocab_size,
                                encoder_dim=512, encoder_fsize=20,
                                converge_vector_dim=256, dropout=0.5, embedding_dropout=0.5).to(self.device)
        # ———————————————————————————————————————————rnn-------------------------------#

        print('-----------------------------------------------')
        input = torch.randn([1, self.img_ch, self.img_size,
                             self.img_size]).to(self.device)
        macs, params = profile(self.disA, inputs=(input,))
        macs, params = clever_format([macs * 2, params * 2], "%.3f")
        print('[Network %s] Total number of parameters: ' % 'disA', params)
        print('[Network %s] Total number of FLOPs: ' % 'disA', macs)
        print('-----------------------------------------------')
        _, _, _, _, real_A_ae, _ = self.disA(input)
        macs, params = profile(self.gen2B, inputs=(real_A_ae,))
        macs, params = clever_format([macs * 2, params * 2], "%.3f")
        print('[Network %s] Total number of parameters: ' % 'gen2B', params)
        print('[Network %s] Total number of FLOPs: ' % 'gen2B', macs)
        print('-----------------------------------------------')

        """ Define Loss """
        self.L1_loss = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)

        """ Trainer """
        self.G_optim = torch.optim.Adam(itertools.chain(self.gen2B.parameters(), self.gen2A.parameters()), lr=self.lr,
                                        betas=(0.5, 0.999), weight_decay=self.weight_decay)
        self.D_optim = torch.optim.Adam(itertools.chain(self.disA.parameters(), self.disB.parameters()), lr=self.lr,
                                        betas=(0.5, 0.999), weight_decay=self.weight_decay)
        # ----------------------tianjia  5.19--------------------------
        # self.rnn_optim = torch.optim.Adam(itertools.chain(self.rnn2A.parameters(), self.rnn2B.parameters()), lr=self.lr,
        #                                   betas=(0.5, 0.999), weight_decay=self.weight_decay)
        self.rnn_optim = torch.optim.Adam(itertools.chain(self.rnn2A.parameters(), self.rnn2B.parameters()), lr=1e-3,
                                          betas=(0.5, 0.999), weight_decay=self.weight_decay)

        # ----------------------数据加载--------------------------

    def train(self):
        # writer = SummaryWriter(os.path.join(self.result_dir, self.dataset, 'summaries/Allothers'))

        self.gen2B.train(), self.gen2A.train(), self.disA.train(), self.disB.train(),
        # --------------------------tianjia 5.19----------
        self.rnn2A.train(), self.rnn2B.train()

        # --------------------------不加载预训练参数----------

        # --------------------------不加载预训练参数----------
        # --------------------------加载预训练参数----------
        # rnn_params = torch.load(os.path.join(self.result_dir, self.dataset + '_rnn_params_latest.pt'))

        # self.disA.load_state_dict(rnn_params['disA'])
        # self.disB.load_state_dict(rnn_params['disB'])
        # self.D_optim.load_state_dict(rnn_params['D_optimizer'])

        # self.rnn2A.load_state_dict(rnn_params['rnn2A'])
        # self.rnn2B.load_state_dict(rnn_params['rnn2B'])
        # self.rnn_optim.load_state_dict(rnn_params['rnn_optimizer'])

        # self.rnn_optim.load_state_dict(params['rnn_optimizer'])

        # --------------------------tianjia 5.19----------
        # 训练生成B A 训练判别器A B
        self.start_iter = 1  # 迭代数开始为1
        if self.resume:  #
            params = torch.load(os.path.join(
                self.result_dir, self.dataset + '_params_latest.pt'))
            # load加载  os.path.join()函数用于路径拼接文件路径 path：读取文件路径；'%s-labels.idx1-ubyte'：拼接文件名的后半部分；
            self.gen2B.load_state_dict(params['gen2B'])  # 仅加载参数
            self.gen2A.load_state_dict(params['gen2A'])
            self.disA.load_state_dict(params['disA'])
            self.disB.load_state_dict(params['disB'])

            # ------------------------------5.19-----------------------------
            self.rnn2A.load_state_dict(params['rnn2A'])
            self.rnn2B.load_state_dict(params['rnn2B'])
            # ------------------------------5.19-----------------------------

            self.D_optim.load_state_dict(params['D_optimizer'])
            self.G_optim.load_state_dict(params['G_optimizer'])

            # 需要添加rnn梯度更新
            self.rnn_optim.load_state_dict(params['rnn_optimizer'])
            # ------------------------------5.19-----------------------------

            # ------------------------------5.19-----------------------------
            self.start_iter = params['start_iter'] + 1
            if self.decay_flag and self.start_iter > (self.iteration // 2):
                self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (
                    self.start_iter - self.iteration // 2)
                self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (
                    self.start_iter - self.iteration // 2)
                # ------------------------------6.4-------------------------------------------------
            if self.decay_flag and self.start_iter > (self.iteration // 2):
                self.rnn_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (
                    self.start_iter - self.iteration // 2)
                # ------------------------------6.4-------------------------------------------------

            print("ok")

        # training loop
        testnum = 4

        print("self.start_iter", self.start_iter)
        print('training start !')
        start_time = time.time()  # 开始时间
        for step in range(self.start_iter, self.iteration + 1):
            if self.decay_flag and step > (self.iteration // 2):
                self.G_optim.param_groups[0]['lr'] -= (
                    self.lr / (self.iteration // 2))
                self.D_optim.param_groups[0]['lr'] -= (
                    self.lr / (self.iteration // 2))

                # ------------------------------6.4-------------------------------------------------
                self.rnn_optim.param_groups[0]['lr'] -= (
                    self.lr / (self.iteration // 2))
                # ------------------------------6.4-------------------------------------------------

            try:
                real_A, trainA_encoded_captions, trainA_caption_lengths, _ = trainA_iter.next()
                # real_A, _ = trainA_iter.next()
            except:
                trainA_iter = iter(self.trainA_loader)
                # real_A, _ = trainA_iter.next()
                real_A, trainA_encoded_captions, trainA_caption_lengths, _ = trainA_iter.next()
            real_A, trainA_encoded_captions, trainA_caption_lengths = real_A.to(self.device), \
                trainA_encoded_captions.to(self.device),\
                trainA_caption_lengths.to(self.device)

            try:
                real_B, trainB_encoded_captions, trainB_caption_lengths, _ = trainB_iter.next()
                # real_B, _ = trainB_iter.next()
            except:

                trainB_iter = iter(self.trainB_loader)
                real_B, trainB_encoded_captions, trainB_caption_lengths, _ = trainB_iter.next()
            real_B, trainB_encoded_captions, trainB_caption_lengths = real_B.to(self.device),\
                trainB_encoded_captions.to(self.device),\
                trainB_caption_lengths.to(self.device)
            # real_B, _ = trainB_iter.next()

            real_A, real_B = real_A.to(self.device), real_B.to(self.device)

            # Update D
            self.D_optim.zero_grad()  # 判别器梯度清0
            # -------------------updata rnn-------------------------------
            # self.rnn_optim.zero_grad()
            # 先把真的A和B放入判别器A B
            #        -------------------------6.3------------------------
            real_LA_logit, real_GA_logit, real_A_cam_logit, _, real_A_z, _ = self.disA(
                real_A)
            real_LB_logit, real_GB_logit, real_B_cam_logit, _, real_B_z, _ = self.disB(
                real_B)
            #        -------------------------6.3------------------------
            # 先把真的A和B放入生成器生成B  A
            fake_A2B = self.gen2B(real_A_z)
            fake_B2A = self.gen2A(real_B_z)

            fake_B2A = fake_B2A.detach()
            fake_A2B = fake_A2B.detach()
            # 先把假的的A和B放入判别器去判别
            fake_LA_logit, fake_GA_logit, fake_A_cam_logit, _, _, feature_outB2A = self.disA(
                fake_B2A)
            fake_LB_logit, fake_GB_logit, fake_B_cam_logit, _, _, feature_outA2B = self.disB(
                fake_A2B)

            D_ad_loss_GA = self.MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit).to(self.device)) + self.MSE_loss(
                fake_GA_logit, torch.zeros_like(fake_GA_logit).to(self.device))
            D_ad_loss_LA = self.MSE_loss(real_LA_logit, torch.ones_like(real_LA_logit).to(self.device)) + self.MSE_loss(
                fake_LA_logit, torch.zeros_like(fake_LA_logit).to(self.device))
            D_ad_loss_GB = self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit).to(self.device)) + self.MSE_loss(
                fake_GB_logit, torch.zeros_like(fake_GB_logit).to(self.device))
            D_ad_loss_LB = self.MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit).to(self.device)) + self.MSE_loss(
                fake_LB_logit, torch.zeros_like(fake_LB_logit).to(self.device))
            D_ad_cam_loss_A = self.MSE_loss(real_A_cam_logit,
                                            torch.ones_like(real_A_cam_logit).to(self.device)) + self.MSE_loss(
                fake_A_cam_logit, torch.zeros_like(fake_A_cam_logit).to(self.device))
            D_ad_cam_loss_B = self.MSE_loss(real_B_cam_logit,
                                            torch.ones_like(real_B_cam_logit).to(self.device)) + self.MSE_loss(
                fake_B_cam_logit, torch.zeros_like(fake_B_cam_logit).to(self.device))

            D_loss_A = self.adv_weight * \
                (D_ad_loss_GA + D_ad_cam_loss_A + D_ad_loss_LA)
            D_loss_B = self.adv_weight * \
                (D_ad_loss_GB + D_ad_cam_loss_B + D_ad_loss_LB)

            # Discriminator_loss = D_loss_A + D_loss_B
            # Discriminator_loss.backward()
            # self.D_optim.step()  # 更新所有的参数

            # -------------------updata rnn-------------------------------
            self.rnn_optim.zero_grad()

            # # -------------------------------------6.11-------------------------------------
            criterion = nn.CrossEntropyLoss().to('cuda')  # 交叉熵损失函数
            train_scoresB2A, train_caps_sortedB2A, train_decode_lengthsB2A, train_alphasB2A, _ = self.rnn2A(feature_outB2A,
                                                                                                            trainA_encoded_captions,
                                                                                                            trainA_caption_lengths)

            train_scores_padB2A = pack_padded_sequence(
                train_scoresB2A, train_decode_lengthsB2A, batch_first=True).data
            train_targets_padB2A = pack_padded_sequence(
                train_caps_sortedB2A, train_decode_lengthsB2A, batch_first=True).data

            train_scoresA2B, train_caps_sortedA2B, train_decode_lengthsA2B, train_alphasA2B, _ = self.rnn2B(feature_outA2B,
                                                                                                            trainB_encoded_captions,
                                                                                                            trainB_caption_lengths)

            train_scores_padA2B = pack_padded_sequence(
                train_scoresA2B, train_decode_lengthsA2B, batch_first=True).data

            train_targets_padA2B = pack_padded_sequence(
                train_caps_sortedA2B, train_decode_lengthsA2B, batch_first=True).data

            train_scores_padA2B, train_targets_padA2B, train_scores_padB2A, train_targets_padB2A = train_scores_padA2B.to(self.device), \
                train_targets_padA2B.to(self.device),\
                train_scores_padB2A.to(self.device), \
                train_targets_padB2A.to(self.device)

            train_loss_rnn_A = criterion(
                train_scores_padA2B, train_targets_padA2B.long()).to(self.device)

            train_loss_rnn_A += 1 * \
                ((1. - train_alphasA2B.sum(dim=1)) ** 2).mean()

            train_loss_rnn_B = criterion(
                train_scores_padB2A, train_targets_padB2A.long()).to(self.device)

            train_loss_rnn_B += 1 * \
                ((1. - train_alphasB2A.sum(dim=1)) ** 2).mean()

            train_loss_rnn = train_loss_rnn_A + train_loss_rnn_B

            D_loss = D_loss_A + D_loss_B
            # Discriminator_loss = D_loss + loss_rnn
            Discriminator_loss = D_loss_A + D_loss_B + 30*train_loss_rnn

            Discriminator_loss.backward()

            # -----------------7.19------------------
            clip_gradient(self.rnn2A, grad_clip=5.)
            clip_gradient(self.rnn2B, grad_clip=5.)
            # -----------------7.19------------------

            self.D_optim.step()  # 更新所有的参数
            self.rnn_optim.step()
            # loss_rnn_A.backward(retain_graph=True)
            # self.rnn_optim.step()
            # loss_rnn_B.backward(retain_graph=True)
            # self.rnn_optim.step()

            #  ------------------改动    6.4-------------

            # Update G
            self.G_optim.zero_grad()  # 梯度清0

            _, _, _, _, real_A_z, _ = self.disA(real_A)  # 编码A
            _, _, _, _, real_B_z, _ = self.disB(real_B)  # 编码A

            fake_A2B = self.gen2B(real_A_z)  # 假的B
            fake_B2A = self.gen2A(real_B_z)  # 假的A
            # 编码假的A
            fake_LA_logit, fake_GA_logit, fake_A_cam_logit, _, fake_A_z, feature_outB2A = self.disA(
                fake_B2A)
            # 编码假的B
            fake_LB_logit, fake_GB_logit, fake_B_cam_logit, _, fake_B_z, feature_outA2B = self.disB(
                fake_A2B)

            # 循环生成A B
            fake_B2A2B = self.gen2B(fake_A_z)
            fake_A2B2A = self.gen2A(fake_B_z)

            # G—A
            G_ad_loss_GA = self.MSE_loss(
                fake_GA_logit, torch.ones_like(fake_GA_logit).to(self.device))
            G_ad_loss_LA = self.MSE_loss(
                fake_LA_logit, torch.ones_like(fake_LA_logit).to(self.device))
            G_ad_loss_GB = self.MSE_loss(
                fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.device))
            G_ad_loss_LB = self.MSE_loss(
                fake_LB_logit, torch.ones_like(fake_LB_logit).to(self.device))

            G_ad_cam_loss_A = self.MSE_loss(
                fake_A_cam_logit, torch.ones_like(fake_A_cam_logit).to(self.device))

            G_ad_cam_loss_B = self.MSE_loss(
                fake_B_cam_logit, torch.ones_like(fake_B_cam_logit).to(self.device))

            G_cycle_loss_A = self.L1_loss(fake_A2B2A, real_A)
            G_cycle_loss_B = self.L1_loss(fake_B2A2B, real_B)

            fake_A2A = self.gen2A(real_A_z)
            fake_B2B = self.gen2B(real_B_z)
            # 希望重建后和原图一样
            G_recon_loss_A = self.L1_loss(fake_A2A, real_A)
            G_recon_loss_B = self.L1_loss(fake_B2B, real_B)

            G_loss_A = self.adv_weight * (
                G_ad_loss_GA + G_ad_cam_loss_A + G_ad_loss_LA) + self.cycle_weight * G_cycle_loss_A + self.recon_weight * G_recon_loss_A
            G_loss_B = self.adv_weight * (
                G_ad_loss_GB + G_ad_cam_loss_B + G_ad_loss_LB) + self.cycle_weight * G_cycle_loss_B + self.recon_weight * G_recon_loss_B
            Generator_loss = G_loss_A + G_loss_B

            Generator_loss.backward()
            self.G_optim.step()  # 更新参数

            print("[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f, r_loss: %.8f" % (step,
                                                                                      self.iteration, time.time() - start_time, D_loss, Generator_loss, train_loss_rnn))
            # print("[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (step, self.iteration, time.time() - start_time, Discriminator_loss, Generator_loss))

            if step % self.save_freq == 0:  # result_dir  self.dataset写入model文件
                self.save(os.path.join(self.result_dir,
                                       self.dataset, 'model'), step)

            if step % self.print_freq == 0:
                print('current D_learning rate:{}'.format(
                    self.D_optim.param_groups[0]['lr']))
                print('current G_learning rate:{}'.format(
                    self.G_optim.param_groups[0]['lr']))
                print('current rnn_learning rate:{}'.format(
                    self.rnn_optim.param_groups[0]['lr']))
                self.save_path("_params_latest.pt", step)

            if step % self.print_freq == 0:
                train_sample_num = testnum  # 训练集和测试集一起 然后包含4个字  5行     开始训练
                test_sample_num = testnum
                A2B = np.zeros((self.img_size * 5, 0, 3))
                B2A = np.zeros((self.img_size * 5, 0, 3))

                # ---------------5.19--------------
                self.rnn2B.eval(), self.rnn2A.eval(),
                # ---------------5.19--------------
                self.gen2B.eval(), self.gen2A.eval(), self.disA.eval(), self.disB.eval()
                # ---------------5.19--------------

                self.rnn2B, self.rnn2A = self.rnn2B.to(
                    self.device), self.rnn2A.to(self.device)
                # ---------------5.19--------------

                self.gen2B, self.gen2A = self.gen2B.to(
                    self.device), self.gen2A.to(self.device)
                self.disA, self.disB = self.disA.to(
                    self.device), self.disB.to(self.device)
                for _ in range(train_sample_num):
                    try:
                        # real_A, _ = trainA_iter.next()
                        real_A, trainA_encoded_captions, trainA_caption_lengths, _ = trainA_iter.next()
                    except:

                        trainA_iter = iter(self.trainA_loader)
                        real_A, trainA_encoded_captions, trainA_caption_lengths, _ = trainA_iter.next()
                    real_A, trainA_encoded_captions, trainA_caption_lengths = real_A.to(self.device), \
                        trainA_encoded_captions.to(self.device), \
                        trainA_caption_lengths.to(self.device)

                    try:
                        real_B, trainB_encoded_captions, trainB_caption_lengths, _ = trainB_iter.next()
                        # real_B, _ = trainB_iter.next()
                    except:

                        trainB_iter = iter(self.trainB_loader)
                        real_B, trainB_encoded_captions, trainB_caption_lengths, _ = trainB_iter.next()
                    real_B, trainB_encoded_captions, trainB_caption_lengths = real_B.to(self.device), \
                        trainB_encoded_captions.to(self.device), \
                        trainB_caption_lengths.to(self.device)

                    real_A, real_B = real_A.to(
                        self.device), real_B.to(self.device)

                    _, _, _, A_heatmap, real_A_z, feature_outA = self.disA(
                        real_A)
                    _, _, _, B_heatmap, real_B_z, feature_outB = self.disB(
                        real_B)

                    fake_A2B = self.gen2B(real_A_z)
                    fake_B2A = self.gen2A(real_B_z)

                    _, _, _, _, fake_A_z, feature_outB2A = self.disA(fake_B2A)
                    _, _, _, _, fake_B_z, feature_outA2B = self.disB(fake_A2B)

                    #  ---------------------------------6.4--------------------------
                    train_scoresB2A, train_caps_sortedB2A, train_decode_lengthsB2A, train_alphasB2A, _ = self.rnn2A(feature_outB2A,
                                                                                                                    trainA_encoded_captions,
                                                                                                                    trainA_caption_lengths)

                    train_scoresA2B, train_caps_sortedA2B, train_decode_lengthsA2B, train_alphasA2B, _ = self.rnn2B(feature_outA2B,
                                                                                                                    trainB_encoded_captions,
                                                                                                                    trainB_caption_lengths)
                    #  ---------------------------------6.4--------------------------

                    fake_B2A2B = self.gen2B(fake_A_z)
                    fake_A2B2A = self.gen2A(fake_B_z)

                    fake_A2A = self.gen2A(real_A_z)
                    fake_B2B = self.gen2B(real_B_z)

                    A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                               cam(tensor2numpy(
                                                                   A_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(
                                                                   denorm(fake_A2A[0]))),
                                                               RGB2BGR(tensor2numpy(
                                                                   denorm(fake_A2B[0]))),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

                    B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                               cam(tensor2numpy(
                                                                   B_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(
                                                                   denorm(fake_B2B[0]))),
                                                               RGB2BGR(tensor2numpy(
                                                                   denorm(fake_B2A[0]))),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

                for _ in range(test_sample_num):
                    try:
                        real_A, testA_encoded_captions, testA_caption_lengths, _ = testA_iter.next()
                        # real_A, _ = testA_iter.next()
                    except:

                        testA_iter = iter(self.testA_loader)
                        real_A, testA_encoded_captions, testA_caption_lengths, _ = testA_iter.next()
                    real_A, testA_encoded_captions, testA_caption_lengths = real_A.to(self.device), \
                        testA_encoded_captions.to(self.device), \
                        testA_caption_lengths.to(self.device)

                    try:
                        real_B, testB_encoded_captions, testB_caption_lengths, _ = testB_iter.next()
                        # real_B, _ = testB_iter.next()
                    except:

                        testB_iter = iter(self.testB_loader)
                        real_B, testB_encoded_captions, testB_caption_lengths, _ = testB_iter.next()
                    real_B, testB_encoded_captions, testB_caption_lengths = real_B.to(self.device), \
                        testB_encoded_captions.to(self.device), \
                        testB_caption_lengths.to(self.device)

                    real_A, real_B = real_A.to(
                        self.device), real_B.to(self.device)

                    _, _, _, A_heatmap, real_A_z, feature_outA = self.disA(
                        real_A)
                    _, _, _, B_heatmap, real_B_z, feature_outB = self.disB(
                        real_B)

                    fake_A2B = self.gen2B(real_A_z)  # 用经过判别器的A编码后经过生成器解码B

                    fake_B2A = self.gen2A(real_B_z)

                    _, _, _, _, fake_A_z, feature_outB2A = self.disA(
                        fake_B2A)  # 把假的A解码
                    _, _, _, _, fake_B_z, feature_outA2B = self.disB(fake_A2B)
                    #  ---------------------------------6.4--------------------------
                    test_scoresB2A, test_caps_sortedB2A, test_decode_lengthsB2A, test_alphasB2A, _ = self.rnn2A(feature_outB2A,
                                                                                                                testA_encoded_captions,
                                                                                                                testA_caption_lengths)

                    test_scoresA2B, test_caps_sortedA2B, test_decode_lengthsA2B, test_alphasA2B, _ = self.rnn2B(feature_outA2B,
                                                                                                                testB_encoded_captions,
                                                                                                                testB_caption_lengths)
                    #  ---------------------------------6.4--------------------------

                    fake_B2A2B = self.gen2B(fake_A_z)  # 把假的A解码后生成B
                    fake_A2B2A = self.gen2A(fake_B_z)

                    fake_A2A = self.gen2A(real_A_z)  # 编码后的又去生成A
                    fake_B2B = self.gen2B(real_B_z)

                    A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                               # 特征图
                                                               cam(tensor2numpy(
                                                                   A_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(
                                                                   denorm(fake_A2A[0]))),
                                                               RGB2BGR(tensor2numpy(
                                                                   denorm(fake_B2A[0]))),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

                    B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                               cam(tensor2numpy(
                                                                   B_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(
                                                                   denorm(fake_B2B[0]))),
                                                               RGB2BGR(tensor2numpy(
                                                                   denorm(fake_A2B[0]))),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

                cv2.imwrite(os.path.join(self.result_dir, self.dataset,
                                         'img', 'A2B_%07d.png' % step), A2B * 255.0)
                cv2.imwrite(os.path.join(self.result_dir, self.dataset,
                                         'img', 'B2A_%07d.png' % step), B2A * 255.0)

                self.gen2B, self.gen2A = self.gen2B.to(
                    self.device), self.gen2A.to(self.device)
                self.disA, self.disB = self.disA.to(
                    self.device), self.disB.to(self.device)
                # ---------------------------5.19--------------------------
                self.rnn2B, self.rnn2A = self.rnn2B.to(
                    self.device), self.rnn2A.to(self.device)

                self.rnn2B.train(), self.rnn2A.train()
                # ---------------------------5.19--------------------------
                self.gen2B.train(), self.gen2A.train(), self.disA.train(), self.disB.train()
            # 33
            # writer = SummaryWriter('./log')
            # writer.add_scalar('rnn/%s' % 'train_loss_rnn', train_loss_rnn.data.cpu().numpy(), global_step=step)
            # writer.add_scalar('Discriminator/%s' % 'd_loss', D_loss.data.cpu().numpy(), global_step=step)
            # writer.add_scalar('Generator/%s' % 'g_loss', Generator_loss.data.cpu().numpy(), global_step=step)
            #
            # writer.close()
            # 加加

    def save(self, dir, step):
        params = {}
        params['gen2B'] = self.gen2B.state_dict()
        params['gen2A'] = self.gen2A.state_dict()
        params['disA'] = self.disA.state_dict()
        params['disB'] = self.disB.state_dict()
        # -----------------5.19-------------
        params['rnn2A'] = self.rnn2A.state_dict()
        params['rnn2B'] = self.rnn2B.state_dict()
        # params['rnn_optimizer2A'] = self.optimizer.zero_grad()
        params['rnn_optimizer'] = self.rnn_optim.state_dict()
        # -----------------5.19-------------

        params['D_optimizer'] = self.D_optim.state_dict()
        params['G_optimizer'] = self.G_optim.state_dict()
        params['start_iter'] = step
        torch.save(params, os.path.join(
            dir, self.dataset + '_params_%07d.pt' % step))

    def save_path(self, path_g, step):
        params = {}
        params['gen2B'] = self.gen2B.state_dict()
        params['gen2A'] = self.gen2A.state_dict()
        params['disA'] = self.disA.state_dict()
        params['disB'] = self.disB.state_dict()
        # -----------------5.19-------------
        params['rnn2A'] = self.rnn2A.state_dict()
        params['rnn2B'] = self.rnn2B.state_dict()
        # params['rnn2A_optimizer'] = self.optimizer.zero_grad()
        params['rnn_optimizer'] = self.rnn_optim.state_dict()
        # -----------------5.19-------------
        params['D_optimizer'] = self.D_optim.state_dict()
        params['G_optimizer'] = self.G_optim.state_dict()
        params['start_iter'] = step
        torch.save(params, os.path.join(
            self.result_dir, self.dataset + path_g))

    def load(self):
        params = torch.load(os.path.join(
            self.result_dir, self.dataset + '_params_latest.pt'))
        self.gen2B.load_state_dict(params['gen2B'])
        self.gen2A.load_state_dict(params['gen2A'])
        self.disA.load_state_dict(params['disA'])
        self.disB.load_state_dict(params['disB'])
        # ------------------------6.4---------------
        self.rnn2A.load_state_dict(params['rnn2A'])
        self.rnn2B.load_state_dict(params['rnn2B'])
        # ------------------------6.4---------------
        self.D_optim.load_state_dict(params['D_optimizer'])
        self.G_optim.load_state_dict(params['G_optimizer'])
        # ------------------------6.4---------------

        self.rnn_optim.load_state_dict(params['rnn_optimizer'])
        # ------------------------6.4---------------
        self.start_iter = params['start_iter']

    def test(self):
        self.load()
        print(self.start_iter)
        # --------------------------5.19---------------
        self.rnn2B.eval(), self.rnn2A.eval(),
        # --------------------------5.19---------------
        self.gen2B.eval(), self.gen2A.eval(), self.disA.eval(), self.disB.eval()

        for n, (real_A, _, _, real_A_path) in enumerate(self.testA_loader):
            real_A = real_A.to(self.device)
            _, _, _, _, real_A_z, _ = self.disA(real_A)
            fake_A2B = self.gen2B(real_A_z)

            A2B = RGB2BGR(tensor2numpy(denorm(fake_A2B[0])))
            print(real_A_path[0])
            cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'fakeB', real_A_path[0].split('/')[-1]),
                        A2B * 255.0)

        for n, (real_B,  _, _, real_B_path) in enumerate(self.testB_loader):
            real_B = real_B.to(self.device)
            _, _, _, _, real_B_z, _ = self.disB(real_B)
            fake_B2A = self.gen2A(real_B_z)

            B2A = RGB2BGR(tensor2numpy(denorm(fake_B2A[0])))
            print(real_B_path[0])
            cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'fakeA', real_B_path[0].split('/')[-1]),
                        B2A * 255.0)
