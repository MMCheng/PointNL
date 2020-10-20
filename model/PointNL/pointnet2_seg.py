from collections import namedtuple

import torch
import torch.nn as nn

from model.PointNL.pointnet2_modules import PointNet2SAModule, PointNet2SAModuleMSG, PointNet2FPModule
from util import pt_util


class PointNet2SSGSeg(nn.Module):
    r"""
        PointNet2 with single-scale grouping
        Semantic segmentation network that uses feature propogation layers
        Parameters
        ----------
        k: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        c: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, c=3, k=13, use_xyz=True):
        super().__init__()
        self.SA_modules = nn.ModuleList()

        self.SA_modules.append(PointNet2SAModule(npoint=1024, nsample=32, sp_knn=20, sp_num=32, mlp=[c, 32, 32, 64], use_xyz=use_xyz))
        self.SA_modules.append(PointNet2SAModule(npoint=256, nsample=32, sp_knn=20, sp_num=16, mlp=[64, 64, 64, 128], use_xyz=use_xyz))
        self.SA_modules.append(PointNet2SAModule(npoint=64, nsample=32, sp_knn=20, sp_num=8, mlp=[128, 128, 128, 256], use_xyz=use_xyz))
        self.SA_modules.append(PointNet2SAModule(npoint=16, nsample=32, sp_knn=20, sp_num=4, mlp=[256, 256, 256, 512], use_xyz=use_xyz))
      
        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointNet2FPModule(mlp=[128 + c, 128, 128, 128]))
        self.FP_modules.append(PointNet2FPModule(mlp=[256 + 64, 256, 128]))
        self.FP_modules.append(PointNet2FPModule(mlp=[256 + 128, 256, 256]))
        self.FP_modules.append(PointNet2FPModule(mlp=[512 + 256, 256, 256]))
        self.FC_layer = nn.Sequential(pt_util.Conv2d(128, 128, bn=True), nn.Dropout(), pt_util.Conv2d(128, k, activation=None))

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None)
        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, in_comp):
        r"""
            Forward pass of the network
            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)
        l_xyz, l_features, l_comp = [xyz], [features], [in_comp]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features, li_comp = self.SA_modules[i](l_xyz[i], l_features[i], l_comp[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
            l_comp.append(li_comp)
        
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i])
        # return self.FC_layer(l_features[0])
        return self.FC_layer(l_features[0].unsqueeze(-1)).squeeze(-1)


