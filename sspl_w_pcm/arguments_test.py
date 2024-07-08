"""
Settings.
"""


import argparse


data_path = "path to Datasets"    # e.g., /home/xxx/Datasets/
weights_frame = "./models/pretrain/frame_best.pth"
weights_sound = "./models/pretrain/sound_best.pth"
weights_pcm = "./models/pretrain/pcm_best.pth"
weights_ssl_head = "./models/pretrain/ssl_head_best.pth"
weights_vggish = "./models/torchvggish/torchvggish/vggish_pretrained/vggish-10086976.pth"


class ArgParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description='Sound Source Localization with Existing Visual Backbones')

        # -----------------------------
        # Misc
        # -----------------------------
        parser.add_argument('--mode', default='eval', type=str,
                            help="training or evaluation state: train or eval")
        parser.add_argument('--id', default='',
                            help="name for identifying the model")
        parser.add_argument('--seed', default=1234, type=int,
                            help='manual seed')
        parser.add_argument('--ckpt', default='./test',
                            help='folder to output checkpoints')
        parser.add_argument('--disp_iter', type=int, default=20,
                            help='frequency to display')
        parser.add_argument('--eval_epoch', type=int, default=1,
                            help='frequency to evaluate')
        parser.add_argument('--num_save_samples', type=int, default=10,
                            help='number of samples saved for visualization')

        # -----------------------------
        # Params for data
        # -----------------------------
        parser.add_argument('--trainset', default='flickr', type=str,
                            help='training data set name: {flickr, vggsound}')
        parser.add_argument('--testset', default='flickr', type=str,
                            help='test data set name: {flickr, vggss}')
        parser.add_argument('--data_path', default=data_path, type=str,
                            help='root directory path of data')
        parser.add_argument('--num_train', default=10000, type=int,
                            help='number of training samples: {10000, 144000}')
        parser.add_argument('--imgSize', default=224, type=int,
                            help='height and width of resized image')
        parser.add_argument('--audSec', default=3, type=int,
                            help='sound length (3s)')
        parser.add_argument('--audRate', default=16000, type=int,
                            help='sound sampling rate (16000 for VGGish)')

        # -----------------------------
        # Params for model
        # -----------------------------
        parser.add_argument('--arch_frame', default='vgg16',
                            help="architecture of net_frame")
        parser.add_argument('--arch_sound', default='vggish',
                            help="architecture of net_sound")
        parser.add_argument('--arch_ssl_method', default='simsiam', type=str,
                            help='name of self-supervised learning method')

        parser.add_argument('--train_from_scratch', default=0, type=int,
                            help='whether train frame network from scratch, 1 for True and 0 for False')
        parser.add_argument('--fine_tune', default=0, type=int,
                            help='whether fine-tune frame network, 1 for True and 0 for False')
        parser.add_argument('--weights_frame', default=weights_frame,
                            help="weights of net_frame")
        parser.add_argument('--weights_sound', default=weights_sound,
                            help="weights of net_sound")
        parser.add_argument('--weights_pcm', default=weights_pcm,
                            help="weights of net_pc")
        parser.add_argument('--weights_ssl_head', default=weights_ssl_head,
                            help="weights of net_ssl_head")
        parser.add_argument('--weights_vggish', default=weights_vggish,
                            help="pre-trained weights of vggish (features + embeddings)")

        parser.add_argument('--dim_f_aud', default=128, type=int,
                            help='dimensionality of original sound net')
        parser.add_argument('--out_dim', default=512, type=int,
                            help='output dimension of customized sound net layer')
        parser.add_argument('--cycs_pcm', default=4, type=int,
                            help='recursive cycles in PCM to infer representations (default is 4). '
                                 'Since initializing representations performs one computing step, '
                                 'the actual recursive cycles = cycs_pcm + 1')

        # -----------------------------
        # Distributed Data Parallel
        # -----------------------------
        parser.add_argument('--gpu_ids', default='0,1', type=str)
        parser.add_argument('--num_gpus', default=2, type=int,
                            help='number of gpus to use within a node')
        parser.add_argument('--batch_size_per_gpu', default=128, type=int,
                            help='batch size for each gpu')
        parser.add_argument('--workers', default=8, type=int,
                            help='number of data loading workers')
        parser.add_argument('--nodes', default=1, type=int, metavar='N',
                            help='number of nodes for distributed training')
        parser.add_argument('--nr', default=0, type=int,
                            help='node rank for distributed training')

        parser.add_argument('--optimizer', default='adamw', type=str,
                            help='optimizer')
        parser.add_argument('--num_epoch', default=20, type=int,
                            help='epochs for training')
        parser.add_argument('--lr_frame', default=2e-5, type=float,
                            help='learning rate for pcm network')
        parser.add_argument('--lr_sound', default=2e-5, type=float,
                            help='learning rate for sound network')
        parser.add_argument('--lr_ssl_head', default=5e-5, type=float,
                            help='learning rate for ssl head')
        parser.add_argument('--beta1', default=0.9, type=float,
                            help='momentum for sgd, beta1 for adam')
        parser.add_argument('--weight_decay', default=1e-4, type=float,
                            help='parameter for weight decay')

        self.parser = parser

    def parse_arguments(self):

        args = self.parser.parse_args()

        print("------------------------ Options ------------------------")
        for key, val in vars(args).items():
            print("{:16} {}".format(key, val))
        print("------------------------ Options ------------------------")

        return args
