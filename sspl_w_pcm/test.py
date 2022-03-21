import os
import time
import json
import math
import random
import warnings
import cv2
import numpy as np
from sklearn.metrics import auc

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from models import ModelBuilder
from dataset.videodataset import VideoDataset

from arguments_test import ArgParser
from utils import makedirs, AverageMeter, save_visual_eval, normalize_img, testset_gt

warnings.filterwarnings('ignore')


def main():
    # arguments
    parser = ArgParser()
    args = parser.parse_arguments()
    args.batch_size = args.num_gpus * args.batch_size_per_gpu

    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    args.vis = os.path.join(args.ckpt, 'visualization/')

    args.world_size = args.num_gpus * args.nodes
    os.environ['MASTER_ADDR'] = '172.18.33.22'  # specified by yourself
    os.environ['MASTER_PORT'] = '8899'  # specified by yourself
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    mp.spawn(main_worker, nprocs=args.num_gpus, args=(args,))


def main_worker(gpu, args):
    rank = args.nr * args.num_gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    ################################
    # model
    ################################
    builder = ModelBuilder()
    net_frame = builder.build_frame(
        arch=args.arch_frame,
        train_from_scratch=args.train_from_scratch,
        fine_tune=args.fine_tune
    )
    net_sound = builder.build_sound(
        arch=args.arch_sound,
        weights_vggish=args.weights_vggish,
        weights_vggish_pca=args.weights_vggish_pca,
        out_dim=args.out_dim
    )
    net_pc = builder.build_feat_fusion_pc(cycs_in=args.cycs_pcm,
                                          dim_audio=args.dim_f_aud,
                                          n_fm_out=args.out_dim)
    net_ssl_head = builder.build_selfsuperlearn_head(
        arch=args.arch_ssl_method,
        in_dim_proj=args.out_dim
    )

    # loss function
    loss = builder.build_criterion(args)

    torch.cuda.set_device(gpu)
    net_frame.cuda(gpu)
    # net_sound.cuda(gpu)
    # net_pc.cuda(gpu)
    # net_ssl_head.cuda(gpu)
    # loss.cuda(gpu)

    # net_frame = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net_frame).cuda(gpu)
    net_sound = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net_sound).cuda(gpu)
    net_pc = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net_pc).cuda(gpu)
    net_ssl_head = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net_ssl_head).cuda(gpu)
    loss = torch.nn.SyncBatchNorm.convert_sync_batchnorm(loss).cuda(gpu)

    netWrapper = NetWrapper(net_frame, net_sound, net_pc, net_ssl_head, loss, args)

    # wrap model
    # net_frame = torch.nn.parallel.DistributedDataParallel(net_frame, device_ids=[gpu])
    net_sound = torch.nn.parallel.DistributedDataParallel(net_sound, device_ids=[gpu])
    net_pc = torch.nn.parallel.DistributedDataParallel(net_pc, device_ids=[gpu])
    net_ssl_head = torch.nn.parallel.DistributedDataParallel(net_ssl_head, device_ids=[gpu])

    # load well-trained model
    map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
    net_frame.load_state_dict(torch.load(args.weights_frame, map_location=map_location))
    net_sound.load_state_dict(torch.load(args.weights_sound, map_location=map_location))
    net_pc.load_state_dict(torch.load(args.weights_pcm, map_location=map_location))
    net_ssl_head.load_state_dict(torch.load(args.weights_ssl_head, map_location=map_location))

    ################################
    # data
    ################################
    dataset_test = VideoDataset(args, mode='test')
    args.batch_size_ = int(args.batch_size / args.num_gpus)
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size_,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False)

    # gt for vggss
    if args.testset == 'vggss':
        args.gt_all = {}
        gt_path = args.data_path + 'VGG-Sound/5k_labeled/Annotations/vggss_test_5158.json'
        with open(gt_path) as json_file:
            annotations = json.load(json_file)
        for annotation in annotations:
            args.gt_all[annotation['file']] = annotation['bbox']

    ################################
    # evaluation
    ################################
    if gpu == 0:

        evaluate(netWrapper, loader_test, args)

        print('Evaluation Done!')


class NetWrapper(torch.nn.Module):
    def __init__(self, net_frame, net_sound, net_pc, net_ssl_head, loss, args):
        super(NetWrapper, self).__init__()
        self.net_frame, self.net_sound, self.net_pc, self.net_ssl_head = net_frame, net_sound, net_pc, net_ssl_head
        self.loss = loss
        self.args = args

    def forward(self, batch_data):
        image_view1 = batch_data['frame_view1']
        image_view2 = batch_data['frame_view2']
        spect = batch_data['spect']

        image_view1 = image_view1.cuda(non_blocking=True)
        image_view2 = image_view2.cuda(non_blocking=True)
        spect = spect.cuda(non_blocking=True)

        # 1. forward net_sound
        audio_feat_orig = []
        audio_feat_trans = []
        for i in range(spect.size(0)):
            audio_feat_orig_i, audio_feat_trans_i = self.net_sound(spect[i])
            audio_feat_orig.append(audio_feat_orig_i.mean(dim=0))
            audio_feat_trans.append(audio_feat_trans_i.mean(dim=0))
        audio_feat_orig = torch.stack(audio_feat_orig, dim=0)
        audio_feat_orig = F.normalize(audio_feat_orig, p=2, dim=1)
        audio_feat_trans = torch.stack(audio_feat_trans, dim=0)

        # 2. forward net_frame
        # view 1
        vis_feat_view1 = self.net_frame(image_view1)
        trans_feat_view1 = self.net_pc(vis_feat_view1, audio_feat_orig)
        feat_view1, sim_map_view1, att_map_view1 = self.att_map_weight(trans_feat_view1, audio_feat_trans)

        # view 2
        vis_feat_view2 = self.net_frame(image_view2)
        trans_feat_view2 = self.net_pc(vis_feat_view2, audio_feat_orig)
        feat_view2, sim_map_view2, att_map_view2 = self.att_map_weight(trans_feat_view2, audio_feat_trans)

        # 3. compute projections and predictions
        p1, p2, z1, z2 = self.net_ssl_head(feat_view1, feat_view2)

        # 4. self-supervised learning (SSL) loss
        loss_ssl = self.loss(p1, p2, z1, z2)

        # save original similarity map
        B, H, W = sim_map_view1.size()
        # relu-softmax mask
        sim_map_relu = F.relu(sim_map_view1)
        sim_map_relu_softmax = F.softmax(sim_map_relu.view(B, -1), dim=1)
        sim_map_relu_softmax = sim_map_relu_softmax.view(B, H, W)

        output = {'orig_sim_map': sim_map_view1,
                  'orig_sim_map_rsf': sim_map_relu_softmax}

        return loss_ssl, output

    def att_map_weight(self, vis_feat_map, audio_feat_vec):

        # normalize visual feature
        B, C, H, W = vis_feat_map.size()
        vis_feat_map_trans = F.normalize(vis_feat_map, p=2, dim=1)
        vis_feat_map_trans = vis_feat_map_trans.view(B, C, H * W)
        vis_feat_map_trans = vis_feat_map_trans.permute(0, 2, 1)  # B x (HW) x C

        # normalize audio feature
        audio_feat_vec = F.normalize(audio_feat_vec, p=2, dim=1)
        audio_feat_vec = audio_feat_vec.unsqueeze(2)  # B x C x 1

        # similarity/attention map
        att_map_orig = torch.matmul(vis_feat_map_trans, audio_feat_vec)  # B x (HW) x 1

        # min-max normalization on similarity map
        att_map = torch.squeeze(att_map_orig)  # B x (HW)
        att_map = (att_map - torch.min(att_map, dim=1, keepdim=True).values) / \
                  (torch.max(att_map, dim=1, keepdim=True).values - torch.min(att_map, dim=1,
                                                                              keepdim=True).values + 1e-10)
        att_map = att_map.unsqueeze(2)  # B x (HW) x 1

        # audio-visual representation
        vis_feat_map = vis_feat_map.view(B, C, -1)  # B x C x (HW)
        fav = torch.matmul(vis_feat_map, att_map)  # B x C x 1
        fav = torch.squeeze(fav)  # B x C

        return fav, att_map_orig.view(B, H, W), att_map.view(B, H, W)


def evaluate(netWrapper, loader, args):
    torch.set_grad_enabled(False)

    # remove previous viz results
    makedirs(args.vis, remove=True)

    netWrapper.eval()

    # initialize meters
    loss_meter = AverageMeter()
    ciou_orig_sim = []
    for i, batch_data in enumerate(loader):

        with torch.no_grad():
            loss_ssl, output = netWrapper.forward(batch_data)

            loss_ssl = loss_ssl.mean()

            img = batch_data['frame_view1'].numpy()
            sound = batch_data['sound'].numpy()
            video_id = batch_data['data_id']

            # original similarity map-related
            orig_sim_map = output['orig_sim_map'].detach().cpu().numpy()
            orig_sim_map_rsf = output['orig_sim_map_rsf'].detach().cpu().numpy()

            # convert similarity map into object mask and save for visualization
            for n in range(img.shape[0]):

                gt_map = testset_gt(args, video_id[n])
                ciou, _, _ = eval_cal_ciou(orig_sim_map[n], gt_map, img_size=args.imgSize, thres=0.5)
                ciou_orig_sim.append(ciou)

                save_visual_eval(video_id[n], img[n], sound[n], orig_sim_map[n], orig_sim_map_rsf[n], args)

        loss_meter.update(loss_ssl.item())
        if args.testset == "flickr":
            print('[Eval] iter {}, loss: {}'.format(i, loss_ssl.item()))

        elif args.testset == "vggss" and i % 6 == 0:
            print('[Eval] iter {}, loss: {}'.format(i, loss_ssl.item()))

    # compute cIoU and AUC on whole dataset
    results_orig_sim = []
    for i in range(21):
        # original similarity map
        result_orig_sim = np.sum(np.array(ciou_orig_sim) >= 0.05 * i)
        result_orig_sim = result_orig_sim / len(ciou_orig_sim)
        results_orig_sim.append(result_orig_sim)

    x = [0.05 * i for i in range(21)]
    cIoU_orig_sim = np.sum(np.array(ciou_orig_sim) >= 0.5) / len(ciou_orig_sim)
    AUC_orig_sim = auc(x, results_orig_sim)

    metric_output = '[Eval Summary] Loss: {:.4f}, cIoU_orig_sim: {:.4f}, AUC_orig_sim: {:.4f}'.format(
        loss_meter.average(), cIoU_orig_sim, AUC_orig_sim)
    print(metric_output)


def eval_cal_ciou(heat_map, gt_map, img_size=224, thres=None):

    # preprocess heatmap
    heat_map = cv2.resize(heat_map, dsize=(img_size, img_size), interpolation=cv2.INTER_LINEAR)
    heat_map = normalize_img(heat_map)

    # convert heatmap to mask
    pred_map = heat_map
    if thres is None:
        threshold = np.sort(pred_map.flatten())[int(pred_map.shape[0] * pred_map.shape[1] / 2)]
        pred_map[pred_map >= threshold] = 1
        pred_map[pred_map < 1] = 0
        infer_map = pred_map
    else:
        infer_map = np.zeros((img_size, img_size))
        infer_map[pred_map >= thres] = 1

    # compute ciou
    inter = np.sum(infer_map * gt_map)
    union = np.sum(gt_map) + np.sum(infer_map * (gt_map == 0))
    ciou = inter / union

    return ciou, inter, union


if __name__ == '__main__':
    main()
