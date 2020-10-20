import os
import time
import random
import numpy as np
import logging
import pickle
import argparse
import h5py

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

from util.util import AverageMeter, intersectionAndUnion, check_makedirs

random.seed(123)
np.random.seed(123)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    
    parser.add_argument('--manual_seed', type=int, default=123, help='manual_seed')
    parser.add_argument('--train_gpu', type=int, nargs='+', default=[0], help='training gpus')
    parser.add_argument('--arch', type=str, default='PointNL', metavar='N')
    parser.add_argument('--use_xyz', type=str, default='True', help='use xyz')
    parser.add_argument('--fea_dim', type=int, default=9, help='input feature dimension')
    parser.add_argument('--classes', type=int, default=13, help='classes')
    parser.add_argument('--train_full_folder', type=str, default='PATH TO DATASETS', help='train full folder')
    parser.add_argument('--num_point', type=int, default=4096, help='Segmentation with the number of points')
    parser.add_argument('--test_area', type=int, default=5, help='test area')
    parser.add_argument('--block_size', type=float, default=1.0, help='block size')
    parser.add_argument('--stride_rate', type=float, default=1.0, help='block size')
    parser.add_argument('--print_freq', type=int, default=1, help='print frequence')
    parser.add_argument('--test_batch_size', type=int, default=16, help='test batchsize')
    parser.add_argument('--ignore_label', type=int, default=255, help='ignore label')
    parser.add_argument('--save_path', type=str, default='./', help='evaluate')
    parser.add_argument('--names_path', type=str, default='./dataset/s3dis/s3dis_names.txt', help='evaluate')
    parser.add_argument('--model_path', type=str, default=None, help='evaluate')
    parser.add_argument('--save_folder', type=str, default=None, help='evaluate')
    
    
    args = parser.parse_args()
    if args.use_xyz == 'True':
        args.use_xyz = True
    else:
        args.use_xyz = False
        print('use_xyz: ', args.use_xyz)

    if args.model_path == None:
        print('please write model path')
        exit()
    
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

def main():
    global args, logger
    args = get_parser()
    logger = get_logger()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    
    logger.info(args)
    assert args.classes > 1
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))

    if args.arch == 'PointNL':
        from model.PointNL.pointnet2_seg import PointNet2SSGSeg as Model

    else:
        raise Exception('architecture not supported yet'.format(args.arch))
    
    #-----------------------------------------------------------------------------------------
    if args.arch in ['PointNL']:
        model = Model(c=args.fea_dim, k=args.classes, use_xyz=args.use_xyz)

    print('Total number of parameters: {}'.format(sum([p.numel() for p in model.parameters()])))

    model = torch.nn.DataParallel(model.cuda())
    logger.info(model)
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
    names = [line.rstrip('\n') for line in open(args.names_path)]
    if os.path.isfile(args.model_path):
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        logger.info("=> loaded checkpoint '{}'".format(args.model_path))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))
    test(model, criterion, names)


def load_point_fea_sp(file_name):
    data_file = h5py.File(file_name, 'r')
    #fist get the number of vertices
    n_ver = len(data_file["linearity"])
    has_labels = len(data_file["labels"])
    #the labels can be empty in the case of a test set
    if has_labels:
        labels = np.array(data_file["labels"]).astype(np.int32)
    else:
        labels = []
    #---create the arrays---
    geof = np.zeros((n_ver, 4), dtype='float32')
    #---fill the arrays---
    geof[:, 0] = data_file["linearity"]
    geof[:, 1] = data_file["planarity"]
    geof[:, 2] = data_file["scattering"]
    geof[:, 3] = data_file["verticality"]
    xyz = data_file["xyz"][:]
    rgb = data_file["rgb"][:]

    in_component = np.array(data_file["in_component"])

    data_file.close()
    return xyz, rgb, geof, labels, in_component


def resort_idx(in_component):
    # in_component: n
    sp_idxs = np.unique(in_component)
    num_sp = sp_idxs.shape[0]
    num_point = in_component.shape[0]

    new_incomponent = np.zeros((num_point, ), dtype=np.int32)
    for i in range(num_sp):
        sp_id_o = sp_idxs[i]
        points_idxs = np.nonzero(in_component == sp_id_o)[0]
        new_incomponent[points_idxs] = i
    return new_incomponent



def data_prepare(room_path):
    xyz, rgb, _, labels, in_component = load_point_fea_sp(room_path)
    points = np.concatenate((xyz, rgb), 1)
    
    coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
    stride = args.block_size * args.stride_rate
    grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - args.block_size) / stride) + 1)
    grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - args.block_size) / stride) + 1)
    data_room, label_room, index_room, in_component_room = np.array([]), np.array([]), np.array([]), np.array([])
    for index_y in range(0, grid_y):
        for index_x in range(0, grid_x):
            s_x = coord_min[0] + index_x * stride
            e_x = min(s_x + args.block_size, coord_max[0])
            s_x = e_x - args.block_size
            s_y = coord_min[1] + index_y * stride
            e_y = min(s_y + args.block_size, coord_max[1])
            s_y = e_y - args.block_size
            point_idxs = np.where((points[:, 0] >= s_x - 1e-8) & (points[:, 0] <= e_x + 1e-8) & (points[:, 1] >= s_y - 1e-8) & (points[:, 1] <= e_y + 1e-8))[0]
            if point_idxs.size == 0:
                continue
            num_batch = int(np.ceil(point_idxs.size / args.num_point))
            point_size = int(num_batch * args.num_point)
            replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
            point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
            point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
            np.random.shuffle(point_idxs)
            data_batch = points[point_idxs, :]
            normlized_xyz = np.zeros((point_size, 3))
            normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]
            normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
            normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]
            data_batch[:, 0] = data_batch[:, 0] - (s_x + args.block_size / 2.0)
            data_batch[:, 1] = data_batch[:, 1] - (s_y + args.block_size / 2.0)
            data_batch[:, 3:6] /= 255.0
            data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)
            label_batch = labels[point_idxs]
            in_component_batch = in_component[point_idxs]
            in_component_batch = resort_idx(in_component_batch)

            data_room = np.vstack([data_room, data_batch]) if data_room.size else data_batch
            label_room = np.hstack([label_room, label_batch]) if label_room.size else label_batch
            in_component_room = np.hstack([in_component_room, in_component_batch]) if in_component_room.size else in_component_batch
            index_room = np.hstack([index_room, point_idxs]) if index_room.size else point_idxs

    assert np.unique(index_room).size == labels.size
    return data_room, label_room, in_component_room, index_room, labels


def test(model, criterion, names):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    rooms = sorted(os.listdir(args.train_full_folder))
    rooms_split = [room for room in rooms if 'Area_{}'.format(args.test_area) in room]
    gt_all, pred_all = np.array([]), np.array([])
    check_makedirs(args.save_folder)
    pred_save, gt_save = [], []
    # total_batch = 0
    for idx, room_name in enumerate(rooms_split):
        data_room, label_room, in_component_room, index_room, gt = data_prepare(os.path.join(args.train_full_folder, room_name))
        batch_point = args.num_point * args.test_batch_size
        batch_num = int(np.ceil(label_room.size / batch_point))
        end = time.time()
        output_room = np.array([])

        for i in range(batch_num):
            s_i, e_i = i * batch_point, min((i + 1) * batch_point, label_room.size)
            input, target, in_component, index = data_room[s_i:e_i, :], label_room[s_i:e_i], in_component_room[s_i:e_i], index_room[s_i:e_i]
            input = torch.from_numpy(input).float().view(-1, args.num_point, input.shape[1])
            target = torch.from_numpy(target).long().view(-1, args.num_point)
            in_component = torch.from_numpy(in_component).long().view(-1, args.num_point)


            with torch.no_grad():
                output = model(input.cuda(), in_component.cuda())
            loss = criterion(output, target.cuda())  # for reference
            output = output.transpose(1, 2).contiguous().view(-1, args.classes).data.cpu().numpy()
            pred = np.argmax(output, axis=1)
            intersection, union, target = intersectionAndUnion(pred, target.view(-1).data.cpu().numpy(), args.classes, args.ignore_label)
            accuracy = sum(intersection) / (sum(target) + 1e-10)
            output_room = np.vstack([output_room, output]) if output_room.size else output
            batch_time.update(time.time() - end)
            end = time.time()
            if ((i + 1) % args.print_freq == 0) or (i + 1 == batch_num):
                logger.info('Test: [{}/{}]-[{}/{}] '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f} - {batch_time.sum:.3f})'
                            'Loss {loss:.4f} '
                            'Accuracy {accuracy:.4f} '
                            'Points {gt.size}.'.format(idx + 1, len(rooms_split),
                                                       i + 1, batch_num,
                                                       batch_time=batch_time,
                                                       loss=loss,
                                                       accuracy=accuracy,
                                                       gt=gt))
        pred = np.zeros((gt.size, args.classes))
        for j in range(len(index_room)):
            pred[index_room[j]] += output_room[j]
        pred = np.argmax(pred, axis=1)

        # calculation 1: add per room predictions
        intersection, union, target = intersectionAndUnion(pred, gt, args.classes, args.ignore_label)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)
        # calculation 2
        pred_all = np.hstack([pred_all, pred]) if pred_all.size else pred
        gt_all = np.hstack([gt_all, gt]) if gt_all.size else gt
        pred_save.append(pred), gt_save.append(gt)


    with open(os.path.join(args.save_folder, "pred_{}.pickle".format(args.test_area)), 'wb') as handle:
        pickle.dump({'pred': pred_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.save_folder, "gt_{}.pickle".format(args.test_area)), 'wb') as handle:
        pickle.dump({'gt': gt_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # calculation 1
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU1 = np.mean(iou_class)
    mAcc1 = np.mean(accuracy_class)
    allAcc1 = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    # calculation 2
    intersection, union, target = intersectionAndUnion(pred_all, gt_all, args.classes, args.ignore_label)
    iou_class = intersection / (union + 1e-10)
    accuracy_class = intersection / (target + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection) / (sum(target) + 1e-10)
    logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    logger.info('Val1 result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU1, mAcc1, allAcc1))

    for i in range(args.classes):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, iou_class[i], accuracy_class[i], names[i]))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    return mIoU, mAcc, allAcc, pred_all


if __name__ == '__main__':
    main()
