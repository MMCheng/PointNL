from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from lib.pointops.functions import pointops
from util import pt_util


class _PointNet2SAModuleBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.sp_knnquery = None
        self.sp_knngroup = None
        
        self.mlps = None
        self.mlps_d_xyz = None
        self.mlps_d_fea = None
        # self.mlps_sp_fea = None

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None, comp: torch.Tensor = None) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, N, C) tensor of the descriptors of the the features
        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        bs, num_points, _ = xyz.size()
        new_features_list = []
        xyz_trans = xyz.transpose(1, 2).contiguous()    # le: BxNx3 -> Bx3xN
       
        cidx = pointops.furthestsampling(xyz, self.npoint)  # BxM
        new_xyz = pointops.gathering(
            xyz_trans,
            cidx
        ).transpose(1, 2).contiguous() if self.npoint is not None else None     # le: Bx3xM -> BxMx3
        
        
        idx_base = torch.arange(0, bs, dtype=torch.long, device=torch.device('cuda')).view(bs, 1)*num_points
        tmp_idx = cidx.long() + idx_base
        tmp_idx = tmp_idx.view(-1)
        new_comp = comp.view(-1)[tmp_idx].view(bs, -1, 1)
       
        for l in range(len(self.groupers)):

            # -------------------------------------------1st non-local -------------------------------------------------------------------
            features = self.mlps_1d[l](features)  # conv: BxC'xN -> BxCxN
            center_features = pointops.gathering(
                features,
                cidx
            ).unsqueeze(-1) if self.npoint is not None else None     # get center points features: BxCxN -> BxCxMx1
            d_xyz1, o_fea1 = self.groupers[l](xyz, new_xyz, features)  # BxCxMxK, BxCxMxK


            w1 = o_fea1 - center_features      # BxCxMxK
            w1 = torch.cat((d_xyz1, w1), dim=1)            # Bx3xMxK + BxCxMxK -> Bx(C+3)xMxK
            w1 = self.mlps_w[l](w1)                       # Bx(C+3)xMxK -> Bx32xMxK -> Bx64xMxK -> Bx(16+C)xMxK
            c_fea1 = F.max_pool2d(w1, kernel_size=[1, w1.size(3)]).squeeze(-1)   # Bx(C+16)xMxK -> Bx(C+16)xM
            w1 = F.softmax(w1, dim=-1)                    # Bx(16+C)xMxK

            d_xyz1 = self.mlps_delta_xyz[l](d_xyz1)       # Bx16xMxK
            
            g_fea1 = torch.cat((d_xyz1, o_fea1), dim=1)  # Bx16xMxK + BxCxMxK => Bx(16+C)xMxK
            g_fea1 = g_fea1 * w1                        # Bx(16+C)xMxK
            g_fea1 = torch.sum(g_fea1, dim=-1)          # Bx(16+C)xM
            
            
            # -------------------------------------------2nd non-local -------------------------------------------------------------------

            sp_knnidx = self.sp_knnquery[l](new_xyz, new_xyz, new_comp, new_comp)       # (xyz, new_xyz, comp, new_comp) -> BxMxK       
            d_xyz2, d_fea2_xyz2, o_fea2 = self.sp_knngroup[l](new_xyz, g_fea1, sp_knnidx)   # (BxMx3, Bx(16+C)xM, BxMxK) -> Bx3xMxK, Bx(C+16+3)xMxK, Bx(C+16)xMxK

            d_xyz2 = self.mlps_delta_xyz2[l](d_xyz2)    # Bx3xMxK -> Bx16xMxK

            w2 = self.mlps_w2[l](d_fea2_xyz2)           # Bx(C+16+3)xMxk -> Bx64xMxK -> Bx(C+16*2)xMxK
            c_fea2 = F.max_pool2d(w2, kernel_size=[1, w2.size(3)]).squeeze(-1)   # Bx(C+16*2)xMxK -> Bx(C+16*2)xM
            w2 = F.softmax(w2, dim=-1)                  # Bx(C+16*2)xMxK
            
            g_fea2 = torch.cat((d_xyz2, o_fea2), dim=1) # Bx16xMxK + Bx(C+16)xMxK -> Bx(C+16*2)xMxK
            g_fea2 *= w2                                # Bx(C+16*2)xMxK * Bx(C+16*2)xMxK -> Bx(C+16*2)xMxK
            g_fea2 = torch.sum(g_fea2, dim=-1)          # Bx(C+16*2)xMxK -> Bx(C+16*2)xM


            # -------------------------------------------3rd non-local -------------------------------------------------------------------
            sp_idx = pointops.furthestsampling(new_xyz, self.list_sp_num[l])     # BxM -> BxP
            sp_xyz = pointops.gathering(
                new_xyz.transpose(1, 2).contiguous(),   # BxMx3 -> Bx3xM
                sp_idx                                  # BxP
            ).transpose(1, 2).contiguous()              # Bx3xP -> BxPx3

            sp_idx_base = torch.arange(0, bs, dtype=torch.long, device=torch.device('cuda')).view(bs, 1)*self.list_sp_num[l]
            tmp_sp_idx = sp_idx.long() + sp_idx_base                # BxP + Bx1 -> BxP
            tmp_sp_idx = tmp_sp_idx.view(-1)                        # BxP -> BP
            sp_comp = new_comp.view(-1)[tmp_sp_idx].view(bs, -1, 1) # comp: BxM -> BM, BM -> BP -> BxPx1, sp id of super points

            sp_localidx = self.sp_knnquery_for_3rd[l](new_xyz, sp_xyz, new_comp, sp_comp)       # (xyz, sp_xyz, comp, sp_comp) -> BxPxK
            sp_fea = self.sp_group[l](None, c_fea2, sp_localidx)    # Bx(C+16*2)xM with idx BxPxK -> Bx(C+16*2)xPxK 

            sp_fea = F.max_pool2d(sp_fea, kernel_size=[1, sp_fea.size(3)]).squeeze(-1)  # Bx(C+16*2)xPx1 -> Bx(C+16*2)xP

            sp_fea_exp = sp_fea.unsqueeze(2).repeat(1, 1, self.npoint, 1)       # Bx(C+16*2)xP -> Bx(C+16*2)x1xP -> Bx(C+16*2)xMxP
            c_fea2_exp = c_fea2.unsqueeze(-1).repeat(1, 1, 1, self.list_sp_num[l])  # Bx(C+16*2)xM -> Bx(C+16*2)xMx1 -> Bx(C+16*2)xMxP
            d_fea3 = sp_fea_exp - c_fea2_exp    # Bx(C+16*2)xMxP - Bx(C+16*2)xMx1 -> Bx(C+16*2)xMxP

            w3 = self.mlps_w3[l](d_fea3)        # Bx(C+16*2)xMxK --> Bx(C+16*2)xMxK
            c_fea3 = F.max_pool2d(w3, kernel_size=[1, w3.size(3)]).squeeze(-1)   # Bx(C+16*2)xMxK -> Bx(C+16*2)xM
            w3 = F.softmax(w3, dim=-1)          # Bx(C+16*2)xMxP -> Bx(C+16*2)xMxP 

            g_fea3 = c_fea2_exp   # Bx(C+16*2)xMxP
            g_fea3 *= w3        # Bx(C+16*2)xMxP
            g_fea3 = torch.sum(g_fea3, dim=-1)  # Bx(C+16*2)xMxP -> Bx(C+16*2)xM


            sp_knnidx = self.sp_knnquery3[l](new_xyz, new_xyz, new_comp, new_comp)       # (xyz, new_xyz, comp, new_comp) -> BxMxK
            d_xyz2, d_fea2_xyz2, o_fea2 = self.sp_knngroup3[l](new_xyz, g_fea2, sp_knnidx)   # (BxMx3, Bx(C+16*2)xM, BxMxK) -> Bx(C+16*2+3)xMxK # xyz是与中心点相减，feature似乎没有相减
            point_fea = self.mlps_sp_fea3[l](d_fea2_xyz2)  # Bx(C+16*2+3)xMxK --> BxCxMxK
            local_point_fea = F.max_pool2d(point_fea, kernel_size=[1, point_fea.size(3)]).squeeze(-1)   # BxCxMxK -> BxCxM


            fea3 = torch.cat((g_fea3, local_point_fea, g_fea2, c_fea2, g_fea1, c_fea1, center_features.squeeze(-1)), dim=1)
            new_features = self.mlps_new[l](fea3)
            new_features_list.append(new_features)
        return new_xyz, torch.cat(new_features_list, dim=1), new_comp.squeeze(-1)      # BxMx3, BxCxM, BxM


class PointNet2SAModuleMSG(_PointNet2SAModuleBase):
    r"""Pointnet set abstrction layer with multiscale grouping
    Parameters
    ----------
    npoint : int
        Number of features
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet_old before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """
    def __init__(self, *, npoint: int, radii: List[float], nsamples: List[int], sp_knns: List[int], sp_nums: List[int], mlps: List[List[int]], bn: bool = True, use_xyz: bool = True):
        super().__init__()
        assert len(radii) == len(nsamples) == len(mlps)
        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.sp_knnquery = nn.ModuleList()
        self.sp_knngroup = nn.ModuleList()
        self.sp_knnquery_for_3rd = nn.ModuleList()
        self.sp_group = nn.ModuleList() 
        self.sp_knnquery3 = nn.ModuleList() 
        self.sp_knngroup3 = nn.ModuleList() 
        
        self.mlps_1d = nn.ModuleList() 
        self.mlps_delta_xyz = nn.ModuleList() 
        self.mlps_delta_fea = nn.ModuleList()
        self.mlps_w = nn.ModuleList()
        self.mlps_new = nn.ModuleList() 

        self.mlps_delta_xyz2 = nn.ModuleList()
        self.mlps_w2 = nn.ModuleList() 
        self.mlps_w3 = nn.ModuleList()
        self.mlps_sp_fea3 = nn.ModuleList()
        
        self.list_sp_num = []     


        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            sp_knn = sp_knns[i]
            sp_num = sp_nums[i]
            print('radius: {} nsample: {} sp_knn: {} sp_num: {}'.format(radius, nsample, sp_knn, sp_num))
            self.groupers.append(
                pointops.PointNL_QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None else pointops.GroupAll(use_xyz)
            )
            self.sp_knnquery.append(
                pointops.SP_KNNQuery(nsample=sp_knn)
            )
            self.sp_knnquery_for_3rd.append(
                pointops.SP_KNNQuery(nsample=sp_num)
            )
            self.sp_knngroup.append(
                pointops.SP_KNNGroup_V1(use_xyz=use_xyz)
            )
            self.sp_knnquery3.append(
                pointops.SP_KNNQuery(nsample=sp_knn)
            )
            self.sp_knngroup3.append(
                pointops.SP_KNNGroup_V1(use_xyz=use_xyz)
            )
            self.sp_group.append(
                pointops.SP_Group()
            )

            self.list_sp_num.append(sp_num)


            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 0
            
            self.mlps_1d.append(pt_util.SharedMLP_1d(mlp_spec, bn=bn))
            self.mlps_delta_xyz.append(pt_util.SharedMLP([3, 32, 16], bn=bn))
            self.mlps_delta_xyz2.append(pt_util.SharedMLP([3, 32, 16], bn=bn)) 

            self.mlps_w.append(pt_util.SharedMLP([mlp_spec[-1]+3, 32, 64, mlp_spec[-1]+16], bn=bn))
            self.mlps_w2.append(pt_util.SharedMLP([mlp_spec[-1]+16+3, 64, mlp_spec[-1]+16*2], bn=bn)) 
            self.mlps_w3.append(pt_util.SharedMLP([mlp_spec[-1]+16*2, 32, 64, mlp_spec[-1]+16*2], bn=bn))
            
            self.mlps_sp_fea3.append(pt_util.SharedMLP([mlp_spec[-1]+16*2+3, 32, 64, mlp_spec[-1]], bn=bn))

            if i<2:
                self.mlps_new.append(pt_util.SharedMLP_1d([mlp_spec[-1]*7+16*8, 256, mlp_spec[-1]], bn=bn))
            elif i==2:
                self.mlps_new.append(pt_util.SharedMLP_1d([mlp_spec[-1]*7+16*8, 512, mlp_spec[-1]], bn=bn)) 
            else:
                self.mlps_new.append(pt_util.SharedMLP_1d([mlp_spec[-1]*7+16*8, 1024, mlp_spec[-1]], bn=bn)) 



class PointNet2SAModule(PointNet2SAModuleMSG):
    r"""Pointnet set abstrction layer
    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet_old before the global max_pool
    bn : bool
        Use batchnorm
    """
    def __init__(self, *, mlp: List[int], npoint: int = None, radius: float = None, nsample: int = None, sp_knn: int = None, sp_num: int = None, bn: bool = True, use_xyz: bool = True):
        super().__init__(mlps=[mlp], npoint=npoint, radii=[radius], nsamples=[nsample], sp_knns=[sp_knn], sp_nums=[sp_num], bn=bn, use_xyz=use_xyz)
        #print('radius', radius) # le


class PointNet2FPModule(nn.Module):
    r"""Propigates the features of one set to another
    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """
    def __init__(self, *, mlp: List[int], bn: bool = True):
        super().__init__()
        self.mlp = pt_util.SharedMLP(mlp, bn=bn)

    def forward(self, unknown: torch.Tensor, known: torch.Tensor, unknow_feats: torch.Tensor, known_feats: torch.Tensor) -> torch.Tensor:
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propigated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propigated
        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """

        if known is not None:
            dist, idx = pointops.nearestneighbor(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_feats = pointops.interpolation(known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2], unknown.size(1))

        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats
        return self.mlp(new_features.unsqueeze(-1)).squeeze(-1)

