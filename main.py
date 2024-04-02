# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
# os.environ['CUDA_VISIBLE_DEVICES']='0,1'
# devices_ids=[0,1]
from torch.nn import DataParallel

from NICE import NICE
import argparse
from utils import *
# -----------------6.25-----------------
#from pretrain import pretrain
"""parsing and configuration"""


def parse_args():
    desc = "Pytorch implementation of NICE-GAN"
    # argparse.ArgumentParser 创建参数
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train',
                        help='[train / test]')  # 解析参数 parser.add_argument
    parser.add_argument('--light', type=str2bool, default=True,
                        help='[NICE-GAN full version / NICE-GAN light version]')
    # parser.add_argument('--dataset', type=str, help='dataset_name')
    parser.add_argument('--dataset', type=str,
                        default='test2', help='dataset_name')

    parser.add_argument('--iteration', type=int, default=50000,
                        help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='The size of batch size')
    parser.add_argument('--print_freq', type=int, default=500,
                        help='The number of image print freq')
    parser.add_argument('--save_freq', type=int, default=500,
                        help='The number of model save freq')
    parser.add_argument('--decay_flag', type=str2bool,
                        default=True, help='The decay_flag')

    parser.add_argument('--lr', type=float, default=0.0001,
                        help='The learning rate')
    parser.add_argument('--weight_decay', type=float,
                        default=0.0001, help='The weight decay')
    parser.add_argument('--adv_weight', type=int,
                        default=1, help='Weight for GAN')
    parser.add_argument('--cycle_weight', type=int,
                        default=10, help='Weight for Cycle')
    parser.add_argument('--recon_weight', type=int,
                        default=10, help='Weight for Reconstruction')

    parser.add_argument('--ch', type=int, default=64,
                        help='base channel number per layer')
    parser.add_argument('--n_res', type=int, default=6,
                        help='The number of resblock')
    parser.add_argument('--n_dis', type=int, default=7,
                        help='The number of discriminator layer')

    parser.add_argument('--img_size', type=int,
                        default=256, help='The size of image')
    parser.add_argument('--img_ch', type=int, default=3,
                        help='The size of image channel')

    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the results')
    # parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Set gpu mode; [cpu, cuda]')   # 用cuda就是用显卡
    parser.add_argument('--device', type=str, default='cuda')   # 用cuda就是用显卡
    parser.add_argument('--benchmark_flag', type=str2bool, default=False)
    parser.add_argument('--resume', type=str2bool, default=False)
    parser.add_argument('--gradient_penalty_weight', type=float, default=10.0)
    parser.add_argument('--n_downsampling', type=int,
                        default=2, help='The number of downsampling')

    return check_args(parser.parse_args())


"""checking arguments"""


def check_args(args):
    check_folder(os.path.join(args.result_dir, args.dataset, 'model'))
    check_folder(os.path.join(args.result_dir, args.dataset, 'rnn_model'))
    check_folder(os.path.join(args.result_dir, args.dataset, 'img'))
    check_folder(os.path.join(args.result_dir, args.dataset, 'fakeA'))
    check_folder(os.path.join(args.result_dir, args.dataset, 'fakeB'))

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args


"""main"""


def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # open session
    gan = NICE(args)

    # -----------------------------6.25------------------
    #pre =pretrain(args)
    # pre.build_model()
    # -----------------------------6.25------------------
    # build graph
    gan.build_model()
# -----------------------------6.25------------------
    if args.phase == 'pretrain':
        pre.pretrain()
        print(" [*] pre-Training finished!")

    if args.phase == 'pretest':
        pre.pretest()
        print(" [*] pre-Training finished!")

# -----------------------------6.25------------------

    if args.phase == 'train':

        gan.train()

        # gan.train()

        print(" [*] Training finished!")

    if args.phase == 'test':
        gan.test()
        print(" [*] Test finished!")


if __name__ == '__main__':

    main()


# python main.py --dataset test2 --phase test  --light True
