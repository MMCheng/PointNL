import os
import time
import random
import numpy as np
import logging
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter

from util import dataset
from util.s3dis import S3DIS
from util.util import AverageMeter, intersectionAndUnionGPU


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    
    parser.add_argument('--manual_seed', type=int, default=123, help='manual_seed')
    parser.add_argument('--train_gpu', type=int, nargs='+', default=[1], help='training gpus')
    parser.add_argument('--sync_bn', type=str, default='True', help='sync batchnorm')
    parser.add_argument('--arch', type=str, default='xxx', metavar='N', choices=[
        'PointNL',
    ], help='model choices')
    parser.add_argument('--use_xyz', type=str, default='True', help='use xyz')   
    parser.add_argument('--fea_dim', type=int, default=9, help='input feature dimension')
    parser.add_argument('--classes', type=int, default=13, help='classes')
    parser.add_argument('--data_name', type=str, default='s3dis', metavar='N', choices=['s3dis', 'scannet', 'modelnet40'], help='datasets choices')
    parser.add_argument('--train_full_folder', type=str, default='PATH TO DATASETS', help='train full folder')
    parser.add_argument('--num_point', type=int, default=4096, help='Segmentation with the number of points')
    parser.add_argument('--test_area', type=int, default=5, help='test area')
    parser.add_argument('--block_size', type=float, default=1.0, help='block size')
    parser.add_argument('--sample_rate', type=float, default=1.0, help='sample rate')
    parser.add_argument('--train_workers', type=int, default=16, help='train workers')

    
    parser.add_argument('--epochs', type=int, default=200, help='train epochs')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequence')
    parser.add_argument('--print_freq', type=int, default=1, help='print frequence')
    parser.add_argument('--train_batch_size', type=int, default=16, help='train batchsize')
    parser.add_argument('--base_lr', type=float, default=0.05, help='sgd base_lr')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight_decay')
    parser.add_argument('--ignore_label', type=int, default=255, help='ignore label')    
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')


    parser.add_argument('--opts', type=str, default='sgd', help='optimizer: sgd or adam')
    parser.add_argument('--step_epoch', type=int, default=25, help='epoch')
    parser.add_argument('--multiplier', type=float, default=0.1, help='multiplier')
    
    parser.add_argument('--weight', type=str, default=None, help='weight')
    parser.add_argument('--resume', type=str, default=None, help='resume')
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--minpoints', type=int, default=1024, help='minimal points')


    args = parser.parse_args()
    if args.use_xyz == 'True':
        args.use_xyz = True
    else:
        args.use_xyz = False
    if args.sync_bn == 'True':
        args.sync_bn =  True
    else:
        args.sync_bn = False
    
    if args.arch in ['PointNL']:
        args.fea_dim = 6
    else:
        print('feature dim error!!!')
    return args


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def init():
    global args, logger, writer
    args = get_parser()
    logger = get_logger()
    writer = SummaryWriter(args.save_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    if args.manual_seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
    logger.info(args)


def main():
    init()
    if args.arch == 'PointNL':
        from model.PointNL.pointnet2_seg import PointNet2SSGSeg as Model
    else:
        print(args.arch)
        print(args.arch=='pointgac_seg')
        raise Exception('architecture not supported yet'.format(args.arch))

        exit()
    
    # -------------------------------------------------------------------------------------------------------------------------------------
    if args.arch in ['PointNL']:
        model = Model(c=args.fea_dim, k=args.classes, use_xyz=args.use_xyz)
   
    else:
        print('error!')
        exit()
        model = Model(c=args.fea_dim, k=args.classes, use_xyz=args.use_xyz)

    
    if args.sync_bn:
        from util.util import convert_to_syncbn
        convert_to_syncbn(model) #, patch_replication_callback(model)

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_epoch, gamma=args.multiplier)


    
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))
    logger.info(model)
    model = torch.nn.DataParallel(model.cuda())
    
    if args.sync_bn:
        from lib.sync_bn import patch_replication_callback
        patch_replication_callback(model)
    
    if args.weight is not None:
        if os.path.isfile(args.weight):
            logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume is not None:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    if args.data_name == 's3dis':
        train_data = S3DIS(split='train', data_root=args.train_full_folder, num_point=args.num_point, test_area=args.test_area, 
                            block_size=args.block_size, sample_rate=args.sample_rate, spf_flag=False, K=32)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True, num_workers=args.train_workers, pin_memory=True, drop_last=True)


    for epoch in range(args.start_epoch, args.epochs):
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, criterion, optimizer, epoch)
        epoch_log = epoch + 1
        writer.add_scalar('loss_train', loss_train, epoch_log)
        writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
        writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
        writer.add_scalar('allAcc_train', allAcc_train, epoch_log)

        if epoch_log % args.save_freq == 0:
            filename = args.save_path + '/train_epoch_' + str(epoch_log) + '.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, filename)
        scheduler.step()


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, (input, target, in_comp) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)       # BxNxC
        target = target.cuda(non_blocking=True)     # BxN
        in_comp = in_comp.cuda(non_blocking=True)   # BxN


        output = model(input, in_comp)
        if target.shape[-1] == 1:
            target = target[:, 0]  # for cls
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # calculate remain time
        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0:
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time, data_time=data_time,
                                                          remain_time=remain_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))
        writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
        writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
        writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
        writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch+1, args.epochs, mIoU, mAcc, allAcc))
    return loss_meter.avg, mIoU, mAcc, allAcc



if __name__ == '__main__':
    main()
