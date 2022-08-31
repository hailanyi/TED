import torch.nn as nn
import numpy as np
from pcdet.datasets.augmentor.X_transform import X_TRANS
import torch

def bilinear_interpolate_torch(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)

    Returns:

    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans

class BEVPool(nn.Module):
    def __init__(self, model_cfg,  voxel_size=None, point_cloud_range=None):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.RANGE = [0, -40, -3, 70.4, 40, 1]
        self.x_trans = X_TRANS()
        self.point_cloud_range = point_cloud_range
        self.voxel_size=voxel_size

    def get_pseudo_points(self, pts_range=[0, -40, -3, 70.4, 40, 1], voxel_size=[0.05, 0.05, 0.05], stride=8):
        x_stride = voxel_size[0] * stride
        y_stride = voxel_size[1] * stride

        min_x = pts_range[0] + x_stride / 2
        max_x = pts_range[3] #+ x_stride / 2
        min_y = pts_range[1] + y_stride / 2
        max_y = pts_range[4] + y_stride / 2

        x = np.arange(min_x, max_x, x_stride)
        y = np.arange(min_y, max_y, y_stride)

        x, y = np.meshgrid(x, y)
        zeo = np.zeros(shape=x.shape)

        grids = torch.from_numpy(np.stack([x, y, zeo]).astype(np.float32)).permute(1,2,0).cuda()

        return grids

    def interpolate_from_bev_features(self, points, bev_features, bev_stride):

        cur_batch_points = points

        x_idxs = (cur_batch_points[:, 0] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (cur_batch_points[:, 1] - self.point_cloud_range[1]) / self.voxel_size[1]
        cur_x_idxs = x_idxs / bev_stride
        cur_y_idxs = y_idxs / bev_stride

        cur_bev_features = bev_features.permute(1, 2, 0)  # (H, W, C)
        point_bev_features = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)

        return point_bev_features

    def bev_align(self, bev_feat, transform_param, stride, stage_i):

        batch_size = len(bev_feat)
        w, h = bev_feat.shape[-2], bev_feat.shape[-1]

        all_feat = []
        for bt_i in range(batch_size):
            cur_bev_feat = bev_feat[bt_i]
            grid_pts = self.get_pseudo_points(self.point_cloud_range, self.voxel_size, stride)

            grid_pts = grid_pts.reshape(-1, 3)
            bt_transform_param = transform_param[bt_i]
            previous_stage_param = bt_transform_param[0]
            current_stage_param = bt_transform_param[stage_i]

            trans_dict = self.x_trans.forward_with_param({'points': grid_pts,
                                                                 'transform_param': current_stage_param})
            trans_dict = self.x_trans.backward_with_param({'points': trans_dict['points'],
                                                          'transform_param': previous_stage_param})

            aligned_feat = self.interpolate_from_bev_features(trans_dict['points'], cur_bev_feat, stride).reshape(w, h, -1)
            aligned_feat=aligned_feat.permute(2,0,1)
            all_feat.append(aligned_feat)

        return torch.stack(all_feat)

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """


        if 'transform_param' in batch_dict:
            trans_param = batch_dict['transform_param']
            rot_num = trans_param.shape[1]
        else:
            rot_num = 1

        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']

        all_feat = []

        for i in range(rot_num):
            if i==0:
                rot_num_id = ''
            else:
                rot_num_id = str(i)

            encoded_spconv_tensor = batch_dict['encoded_spconv_tensor'+rot_num_id]
            spatial_features = encoded_spconv_tensor.dense()
            N, C, D, H, W = spatial_features.shape
            spatial_features = spatial_features.view(N, C * D, H, W)

            batch_dict['spatial_features'+rot_num_id] = spatial_features

            if i==0:
                all_feat.append(spatial_features)
            elif 'transform_param' in batch_dict and i>0:
                aligned_bev_feat = self.bev_align(spatial_features.clone(),
                                                  batch_dict['transform_param'],
                                                  batch_dict['spatial_features_stride'],
                                                  i)
                all_feat.append(aligned_bev_feat)


        if 'transform_param' in batch_dict:
            all_feat = torch.stack(all_feat)

            if self.model_cfg.get('ALIGN_METHOD', 'none') == 'max':
                final_feat = all_feat.max(0)[0]
                batch_dict['spatial_features'] = final_feat
            elif self.model_cfg.get('ALIGN_METHOD', 'none') == 'mean':
                final_feat = all_feat.mean(0)
                batch_dict['spatial_features'] = final_feat
            else:
                raise NotImplementedError


        return batch_dict