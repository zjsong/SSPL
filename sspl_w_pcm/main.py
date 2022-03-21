import os
import cv2
import time
import json
import math
import random
import warnings
import numpy as np
from sklearn.metrics import auc

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from models import ModelBuilder
from dataset.videodataset import VideoDataset

from arguments_train import ArgParser
from utils import makedirs, AverageMeter, save_visual, plot_loss_metrics, normalize_img, testset_gt

warnings.filterwarnings('ignore')


def main():
    # arguments
    parser = ArgParser()
    args = parser.parse_train_arguments()
    args.batch_size = args.num_gpus * args.batch_size_per_gpu

    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    if args.mode == 'train':
        args.id += 'train_{}-test_{}'.format(args.trainset, args.testset)
        args.id += '-{}-{}-{}'.format(args.arch_frame, args.arch_sound, args.arch_ssl_method)
    print('Model ID: {}'.format(args.id))

    args.vis = os.path.join(args.ckpt, 'visualization/')
    args.log = os.path.join(args.ckpt, 'running_log.txt')
    if args.mode == 'train':
        makedirs(args.ckpt, remove=True)
        args_path = os.path.join(args.ckpt, 'args.json')
        args_store = vars(args).copy()
        args_store['device'] = None
        with open(args_path, 'w') as json_file:
            json.dump(args_store, json_file)

    # initialize best cIoU with a small number
    args.best_ciou = -float("inf")

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

    optimizer = create_optimizer(net_sound, net_pc, net_ssl_head, args)

    # wrap model
    # net_frame = torch.nn.parallel.DistributedDataParallel(net_frame, device_ids=[gpu])
    net_sound = torch.nn.parallel.DistributedDataParallel(net_sound, device_ids=[gpu])
    net_pc = torch.nn.parallel.DistributedDataParallel(net_pc, device_ids=[gpu])
    net_ssl_head = torch.nn.parallel.DistributedDataParallel(net_ssl_head, device_ids=[gpu])

    ################################
    # data
    ################################
    dataset_train = VideoDataset(args, mode='train')
    dataset_test = VideoDataset(args, mode='test')

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    args.batch_size_ = int(args.batch_size / args.num_gpus)

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size_,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True)
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

    # history of peroformance
    history = {
        'train': {'epoch': [], 'loss': []},
        'val': {'epoch': [], 'loss': [], 'ciou': [], 'auc': []}
    }

    ################################
    # evaluation first
    ################################
    if gpu == 0:
        args.epoch_iters = len(dataset_train) // args.batch_size
        print('1 Epoch = {} iters'.format(args.epoch_iters))

        history['train']['epoch'].append(0)
        history['train']['loss'].append(11)
        evaluate(netWrapper, loader_test, history, 0, args)

    ################################
    # training
    ################################
    for epoch in range(args.num_epoch):

        train_sampler.set_epoch(epoch)

        train(netWrapper, loader_train, optimizer, history, epoch + 1, gpu, args)

        if (epoch + 1) % args.eval_epoch == 0 and gpu == 0:
            evaluate(netWrapper, loader_test, history, epoch + 1, args)

            # checkpointing
            checkpoint(net_frame, net_sound, net_pc, net_ssl_head, history, epoch + 1, args)

            print('end of display \n')

    if gpu == 0:
        print('Training Done!')


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

        output = {'orig_sim_map': sim_map_view1}

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


def train(netWrapper, loader, optimizer, history, epoch, gpu, args):
    torch.set_grad_enabled(True)
    batch_time = AverageMeter()
    data_time = AverageMeter()

    netWrapper.train()

    torch.cuda.synchronize()
    tic = time.perf_counter()
    for i, batch_data in enumerate(loader):
        torch.cuda.synchronize()
        data_time.update(time.perf_counter() - tic)

        netWrapper.zero_grad()

        loss_ssl, _ = netWrapper.forward(batch_data)
        loss_ssl = loss_ssl.mean()

        loss_ssl.backward()
        optimizer.step()

        torch.cuda.synchronize()
        batch_time.update(time.perf_counter() - tic)
        tic = time.perf_counter()

        if i % args.disp_iter == 0 and gpu == 0:
            if i == 0:
                print('------------------------------------------------------------------------------')
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, lr_frame: {}, lr_sound: {}, '
                  'lr_ssl_head: {}, loss: {:.4f}'.format(
                epoch, i, args.epoch_iters, batch_time.average(), data_time.average(),
                args.lr_frame, args.lr_sound, args.lr_ssl_head, loss_ssl.item()))

            if gpu == 0:
                fractional_epoch = epoch + 1. * i / args.epoch_iters
                history['train']['epoch'].append(fractional_epoch)
                history['train']['loss'].append(loss_ssl.item())


def evaluate(netWrapper, loader, history, epoch, args):
    print('Evaluation at {} epochs...'.format(epoch))
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
            video_id = batch_data['data_id']

            # original similarity map-related
            orig_sim_map = output['orig_sim_map'].detach().cpu().numpy()

            # convert heatmap into object mask and save for visualization
            for n in range(img.shape[0]):

                gt_map = testset_gt(args, video_id[n])
                ciou, _, _ = eval_cal_ciou(orig_sim_map[n], gt_map, img_size=args.imgSize, thres=0.5)
                ciou_orig_sim.append(ciou)

                if i == 0 and (n + 1) % (args.batch_size_per_gpu // 20) == 0:
                    save_visual(video_id[n], img[n], orig_sim_map[n], args)

        loss_meter.update(loss_ssl.item())
        if args.testset == "flickr":
            print('[Eval] iter {}, loss: {}'.format(i, loss_ssl.item()))

        elif args.testset == "vggss" and i % 6 == 0:
            print('[Eval] iter {}, loss: {}'.format(i, loss_ssl.item()))

    # compute cIoU and AUC on whole dataset
    results_orig_sim = []
    for i in range(21):
        result_orig_sim = np.sum(np.array(ciou_orig_sim) >= 0.05 * i)
        result_orig_sim = result_orig_sim / len(ciou_orig_sim)
        results_orig_sim.append(result_orig_sim)

    x = [0.05 * i for i in range(21)]
    cIoU_orig_sim = np.sum(np.array(ciou_orig_sim) >= 0.5) / len(ciou_orig_sim)
    AUC_orig_sim = auc(x, results_orig_sim)

    metric_output = '[Eval Summary] Epoch: {:03d}, Loss: {:.4f}, ' \
                    'cIoU_orig_sim: {:.4f}, AUC_orig_sim: {:.4f}'.format(
        epoch, loss_meter.average(),
        cIoU_orig_sim, AUC_orig_sim)
    print(metric_output)
    with open(args.log, 'a') as F:
        F.write(metric_output + '\n')

    history['val']['epoch'].append(epoch)
    history['val']['loss'].append(loss_meter.average())
    history['val']['ciou'].append(cIoU_orig_sim)
    history['val']['auc'].append(AUC_orig_sim)

    print('Plotting figures...')
    plot_loss_metrics(args.ckpt, history)


def eval_cal_ciou(heat_map, gt_map, img_size=224, thres=None):

    # preprocess heatmap
    heat_map = cv2.resize(heat_map, dsize=(img_size, img_size), interpolation=cv2.INTER_LINEAR)
    heat_map = normalize_img(heat_map)

    # convert heatmap to mask
    pred_map = heat_map
    if thres is None:
        threshold = np.sort(pred_map.flatten())[int(pred_map.shape[0] * pred_map.shape[1] / 2)]
        pred_map[pred_map >= threshold] = 1
        pred_map[pred_map < threshold] = 0
        infer_map = pred_map
    else:
        infer_map = np.zeros((img_size, img_size))
        infer_map[pred_map >= thres] = 1

    # compute ciou
    inter = np.sum(infer_map * gt_map)
    union = np.sum(gt_map) + np.sum(infer_map * (gt_map == 0))
    ciou = inter / union

    return ciou, inter, union


def create_optimizer(net_sound, net_pc, net_ssl_head, args):

    params_pc_convs = [p for p in net_pc.PredProcess.parameters()] + \
                      [p for p in net_pc.ErrorProcess.parameters()] + \
                      [p for p in net_pc.BNPred.parameters()] + \
                      [p for p in net_pc.BNError.parameters()] + \
                      [p for p in net_pc.BNPred_step.parameters()] + \
                      [p for p in net_pc.BNError_step.parameters()]
    params_pc_rates = [p for p in net_pc.b0.parameters()] + \
                      [p for p in net_pc.a0.parameters()]

    params_group = [{'params': net_sound.fc.parameters(), 'lr': args.lr_sound},
                    {'params': params_pc_convs, 'lr': args.lr_frame},
                    {'params': params_pc_rates, 'lr': args.lr_frame, 'weight_decay': 0},
                    {'params': net_ssl_head.parameters(), 'lr': args.lr_ssl_head}]

    return torch.optim.AdamW(params_group, betas=(args.beta1, 0.999), weight_decay=args.weight_decay)


def checkpoint(net_frame, net_sound, net_pc, net_ssl_head, history, epoch, args):
    print('Saving checkpoints at {} epochs.'.format(epoch))
    suffix_best = 'best.pth'

    cur_ciou = history['val']['ciou'][-1]
    if cur_ciou > args.best_ciou:
        args.best_ciou = cur_ciou
        torch.save(net_frame.state_dict(),
                   '{}/frame_{}'.format(args.ckpt, suffix_best))
        torch.save(net_sound.state_dict(),
                   '{}/sound_{}'.format(args.ckpt, suffix_best))
        torch.save(net_pc.state_dict(),
                   '{}/pcm_{}'.format(args.ckpt, suffix_best))
        torch.save(net_ssl_head.state_dict(),
                   '{}/ssl_head_{}'.format(args.ckpt, suffix_best))


if __name__ == '__main__':
    main()
