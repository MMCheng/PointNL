#!/usr/bin/env python
# coding=utf-8

import os
import numpy as np
import h5py
from torch.utils.data import Dataset
import torch

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



class DatasetBase(Dataset):
    """docstring for  Dataset"""
    def __init__(self, data_root, split, num_point=4096, test_area=5, block_size=1.0, transform=None):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        self.split = split

        print('test area is: {}'.format(test_area))
        
        rooms = sorted(os.listdir(data_root))
        rooms = [room for room in rooms if 'Area_' in room]
        if split == 'train':
            rooms_split = [room for room in rooms if not 'Area_{}'.format(test_area) in room]
        else:
            rooms_split = [room for room in rooms if 'Area_{}'.format(test_area) in room]

        self.rooms_split = rooms_split
        print('len of rooms_split: {}'.format(len(self.rooms_split)))
        self.room_points, self.room_labels = [], []
        self.room_geofs, self.in_component = [], []
    
    


class S3DIS(DatasetBase):
    """docstring for RandomBlockData"""
    def __init__(self, data_root, split='train', num_point=4096, test_area=5, block_size=1.0,
        sample_rate=1.0, spf_flag=True, geof_flag=False, K=32):
        super().__init__(data_root=data_root, split = split, test_area=test_area)
        print('{}-S3DIS'.format(split))
        self.spf_flag = spf_flag
        self.geof_flag = geof_flag
        self.room_coord_min, self.room_coord_max = [], []
        self.room_names = []
        self.NUM_POINT = K
        num_point_all = []
        for room_name in self.rooms_split:
            room_path = os.path.join(data_root, room_name)
            xyz, rgb, geof, labels, in_component = load_point_fea_sp(room_path)

            points = np.concatenate((xyz,rgb), 1)
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]

            self.room_points.append(points), self.room_labels.append(labels)
            self.room_geofs.append(geof), self.in_component.append(in_component)
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
            num_point_all.append(labels.size)

        
        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point) # num_batch_all
        room_idxs = []
        for index in range(len(self.rooms_split)):
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
            self.room_names.extend([self.rooms_split[index]] * int(round(sample_prob[index] * num_iter)))
            
        self.room_idxs = np.array(room_idxs)

    
    def resort_idx(self, idx, in_component):
        # in_component: n
        sp_idxs = np.unique(in_component)
        num_sp = sp_idxs.shape[0]
        num_point = in_component.shape[0]

        new_incomponent = np.zeros((num_point, ), dtype=np.int32)
        for i in range(num_sp):
            sp_id_o = sp_idxs[i]
            points_idxs = np.nonzero(in_component == sp_id_o)[0]
            new_incomponent[points_idxs] = i
        return new_incomponent      # n

    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]   # N * 6
        labels = self.room_labels[room_idx]   # N * 14
        geofs = self.room_geofs[room_idx]
        in_component = self.in_component[room_idx]
        room_name = self.room_names[room_idx]
        N_points = points.shape[0]

        while (True):
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 1024:
                break
        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 6
        current_points = np.zeros((self.num_point, 9))  # num_point * 9
        current_points[:, 6] = selected_points[:, 0] / self.room_coord_max[room_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.room_coord_max[room_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.room_coord_max[room_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs] # self.num_point * 14

        current_in_component = in_component[selected_point_idxs]
        current_in_component = self.resort_idx(idx, current_in_component)    # resort super point idx


        current_points = torch.from_numpy(current_points).float()
        current_labels = torch.from_numpy(current_labels.astype(np.int8)).long()
        current_in_component = torch.from_numpy(current_in_component.astype(np.int32)).long()
        
        return current_points, current_labels, current_in_component


    def __len__(self):
        return len(self.room_idxs)
