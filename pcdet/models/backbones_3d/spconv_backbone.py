from functools import partial
from ...utils.spconv_utils import replace_feature, spconv
import torch.nn as nn
import numpy as np
import torch

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
        relu = nn.ReLU()
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
        relu =nn.ReLU(inplace=True)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
        relu = nn.ReLU()
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        relu,
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out

class BasicBlock(spconv.SparseModule):

    def __init__(self, inplanes, planes,  norm_fn=None, stride=2,  padding=1,  indice_key=None):
        super(BasicBlock, self).__init__()

        assert norm_fn is not None

        block = post_act_block
        self.stride = stride
        if stride >1:
            self.down_conv = block(inplanes,
                                    planes,
                                    3,
                                    norm_fn=norm_fn,
                                    stride=2,
                                    padding=padding,
                                    indice_key=('sp' + indice_key),
                                    conv_type='spconv')
        if stride >1:
            conv_in = planes
        else:
            conv_in = inplanes

        self.conv1 = block(conv_in,
                              planes // 2,
                              3,
                              norm_fn=norm_fn,
                              padding=1,
                              indice_key=('subm1' + indice_key))
        self.conv2 = block(planes//2,
                              planes // 2,
                              3,
                              norm_fn=norm_fn,
                              padding=1,
                              indice_key=('subm2' + indice_key))

        self.conv3 = block(planes//2,
                              planes // 2,
                              3,
                              norm_fn=norm_fn,
                              padding=1,
                              indice_key=('subm3' + indice_key))
        self.conv4 = block(planes//2,
                              planes // 2,
                              3,
                              norm_fn=norm_fn,
                              padding=1,
                              indice_key=('subm4' + indice_key))


    def forward(self, x):

        if self.stride>1:
            x = self.down_conv(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        out = replace_feature(x2, torch.cat([x1.features, x4.features],-1))

        return out

class TeMMVoxelBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size,  **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        self.return_num_features_as_dict = model_cfg.RETURN_NUM_FEATURES_AS_DICT
        self.out_features=model_cfg.OUT_FEATURES

        num_filters = model_cfg.NUM_FILTERS

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, num_filters[0], 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(num_filters[0]),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(num_filters[0], num_filters[0], 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(num_filters[0], num_filters[1], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(num_filters[1], num_filters[1], 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(num_filters[1], num_filters[1], 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(num_filters[1], num_filters[2], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(num_filters[2], num_filters[2], 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(num_filters[2], num_filters[2], 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(num_filters[2], num_filters[3], 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(num_filters[3], num_filters[3], 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(num_filters[3], num_filters[3], 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(num_filters[3], self.out_features, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(self.out_features),
            nn.ReLU(),
        )
        if self.model_cfg.get('MM', False):
            self.conv_input_2 = spconv.SparseSequential(
                spconv.SubMConv3d(input_channels, num_filters[0], 3, padding=1, bias=False, indice_key='subm1_2'),
                norm_fn(num_filters[0]),
                nn.ReLU(),
            )
            block = post_act_block

            self.conv1_2 = spconv.SparseSequential(
                block(num_filters[0], num_filters[0], 3, norm_fn=norm_fn, padding=1, indice_key='subm1_2'),
            )

            self.conv2_2 = spconv.SparseSequential(
                # [1600, 1408, 41] <- [800, 704, 21]
                block(num_filters[0], num_filters[1], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2_2', conv_type='spconv'),
                block(num_filters[1], num_filters[1], 3, norm_fn=norm_fn, padding=1, indice_key='subm2_2'),
                block(num_filters[1], num_filters[1], 3, norm_fn=norm_fn, padding=1, indice_key='subm2_2'),
            )

            self.conv3_2 = spconv.SparseSequential(
                # [800, 704, 21] <- [400, 352, 11]
                block(num_filters[1], num_filters[2], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3_2', conv_type='spconv'),
                block(num_filters[2], num_filters[2], 3, norm_fn=norm_fn, padding=1, indice_key='subm3_2'),
                block(num_filters[2], num_filters[2], 3, norm_fn=norm_fn, padding=1, indice_key='subm3_2'),
            )

            self.conv4_2 = spconv.SparseSequential(
                # [400, 352, 11] <- [200, 176, 5]
                block(num_filters[2], num_filters[3], 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4_2', conv_type='spconv'),
                block(num_filters[3], num_filters[3], 3, norm_fn=norm_fn, padding=1, indice_key='subm4_2'),
                block(num_filters[3], num_filters[3], 3, norm_fn=norm_fn, padding=1, indice_key='subm4_2'),
            )

        self.num_point_features = self.out_features

        if self.return_num_features_as_dict:
            num_point_features = {}
            num_point_features.update({
                'x_conv1': num_filters[0],
                'x_conv2': num_filters[1],
                'x_conv3': num_filters[2],
                'x_conv4': num_filters[3],
            })
            self.num_point_features = num_point_features

    def decompose_tensor(self, tensor, i, batch_size):
        input_shape = tensor.spatial_shape[2]
        begin_shape_ids = i * (input_shape // 4)
        end_shape_ids = (i + 1) * (input_shape // 4)
        x_conv3_features = tensor.features
        x_conv3_coords = tensor.indices

        mask = (begin_shape_ids < x_conv3_coords[:, 3]) & (x_conv3_coords[:, 3] < end_shape_ids)
        this_conv3_feat = x_conv3_features[mask]
        this_conv3_coords = x_conv3_coords[mask]
        this_conv3_coords[:, 3] -= i * (input_shape // 4)
        this_shape = [tensor.spatial_shape[0], tensor.spatial_shape[1], tensor.spatial_shape[2] // 4]

        this_conv3_tensor = spconv.SparseConvTensor(
            features=this_conv3_feat,
            indices=this_conv3_coords.int(),
            spatial_shape=this_shape,
            batch_size=batch_size
        )
        return this_conv3_tensor

    def forward_test(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """

        if 'transform_param' in batch_dict:
            trans_param = batch_dict['transform_param']
            rot_num = trans_param.shape[1]
        else:
            rot_num = 1

        all_lidar_feat = []
        all_lidar_coords = []

        new_shape = [self.sparse_shape[0], self.sparse_shape[1], self.sparse_shape[2] * 4]

        for i in range(rot_num):
            if i==0:
                rot_num_id = ''
            else:
                rot_num_id = str(i)

            voxel_features, voxel_coords = batch_dict['voxel_features'+rot_num_id], batch_dict['voxel_coords'+rot_num_id]

            all_lidar_feat.append(voxel_features)
            new_coord = voxel_coords.clone()
            new_coord[:, 3] += i*self.sparse_shape[2]
            all_lidar_coords.append(new_coord)
        batch_size = batch_dict['batch_size']

        all_lidar_feat = torch.cat(all_lidar_feat, 0)
        all_lidar_coords = torch.cat(all_lidar_coords)

        input_sp_tensor = spconv.SparseConvTensor(
            features=all_lidar_feat,
            indices=all_lidar_coords.int(),
            spatial_shape=new_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        out = self.conv_out(x_conv4)
        for i in range(rot_num):
            if i==0:
                rot_num_id = ''
            else:
                rot_num_id = str(i)

            this_conv3 = self.decompose_tensor(x_conv3, i, batch_size)
            this_conv4 = self.decompose_tensor(x_conv4, i, batch_size)
            this_out = self.decompose_tensor(out, i, batch_size)

            batch_dict.update({
                'encoded_spconv_tensor'+rot_num_id: this_out,
                'encoded_spconv_tensor_stride'+rot_num_id: 8,
            })
            batch_dict.update({
                'multi_scale_3d_features'+rot_num_id: {
                    'x_conv1': None,
                    'x_conv2': None,
                    'x_conv3': this_conv3,
                    'x_conv4': this_conv4,
                },
                'multi_scale_3d_strides'+rot_num_id: {
                    'x_conv1': 1,
                    'x_conv2': 2,
                    'x_conv3': 4,
                    'x_conv4': 8,
                }
            })


        if self.model_cfg.get('MM', False):
            all_mm_feat = []
            all_mm_coords = []
            for i in range(rot_num):
                if i == 0:
                    rot_num_id = ''
                else:
                    rot_num_id = str(i)

                newvoxel_features, newvoxel_coords = batch_dict['voxel_features_mm'+rot_num_id], batch_dict['voxel_coords_mm'+rot_num_id]

                all_mm_feat.append(newvoxel_features)
                new_mm_coord = newvoxel_coords.clone()
                new_mm_coord[:, 3] += i * self.sparse_shape[2]
                all_mm_coords.append(new_mm_coord)
            all_mm_feat = torch.cat(all_mm_feat, 0)
            all_mm_coords = torch.cat(all_mm_coords)

            newinput_sp_tensor = spconv.SparseConvTensor(
                features=all_mm_feat,
                indices=all_mm_coords.int(),
                spatial_shape=new_shape,
                batch_size=batch_size
            )

            newx = self.conv_input_2(newinput_sp_tensor)

            newx_conv1 = self.conv1_2(newx)
            newx_conv2 = self.conv2_2(newx_conv1)
            newx_conv3 = self.conv3_2(newx_conv2)
            newx_conv4 = self.conv4_2(newx_conv3)

            for i in range(rot_num):
                if i == 0:
                    rot_num_id = ''
                else:
                    rot_num_id = str(i)

                this_conv3 = self.decompose_tensor(newx_conv3, i, batch_size)
                this_conv4 = self.decompose_tensor(newx_conv4, i, batch_size)
                batch_dict.update({
                    'encoded_spconv_tensor_stride_mm'+rot_num_id: 8
                })
                batch_dict.update({
                    'multi_scale_3d_features_mm'+rot_num_id: {
                        'x_conv1': None,
                        'x_conv2': None,
                        'x_conv3': this_conv3,
                        'x_conv4': this_conv4,
                    },
                    'multi_scale_3d_strides'+rot_num_id: {
                        'x_conv1': 1,
                        'x_conv2': 2,
                        'x_conv3': 4,
                        'x_conv4': 8,
                    }
                })

        return batch_dict

    def forward_train(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        if 'transform_param' in batch_dict:
            trans_param = batch_dict['transform_param']
            rot_num = trans_param.shape[1]
        else:
            rot_num = 1

        for i in range(rot_num):
            if i==0:
                rot_num_id = ''
            else:
                rot_num_id = str(i)

            voxel_features, voxel_coords = batch_dict['voxel_features'+rot_num_id], batch_dict['voxel_coords'+rot_num_id]

            batch_size = batch_dict['batch_size']
            input_sp_tensor = spconv.SparseConvTensor(
                features=voxel_features,
                indices=voxel_coords.int(),
                spatial_shape=self.sparse_shape,
                batch_size=batch_size
            )
            x = self.conv_input(input_sp_tensor)

            x_conv1 = self.conv1(x)
            x_conv2 = self.conv2(x_conv1)
            x_conv3 = self.conv3(x_conv2)
            x_conv4 = self.conv4(x_conv3)

            # for detection head
            # [200, 176, 5] -> [200, 176, 2]
            out = self.conv_out(x_conv4)

            batch_dict.update({
                'encoded_spconv_tensor'+rot_num_id: out,
                'encoded_spconv_tensor_stride'+rot_num_id: 8,
            })
            batch_dict.update({
                'multi_scale_3d_features'+rot_num_id: {
                    'x_conv1': x_conv1,
                    'x_conv2': x_conv2,
                    'x_conv3': x_conv3,
                    'x_conv4': x_conv4,
                },
                'multi_scale_3d_strides'+rot_num_id: {
                    'x_conv1': 1,
                    'x_conv2': 2,
                    'x_conv3': 4,
                    'x_conv4': 8,
                }
            })

            if self.model_cfg.get('MM', False):
                newvoxel_features, newvoxel_coords = batch_dict['voxel_features_mm'+rot_num_id], batch_dict['voxel_coords_mm'+rot_num_id]

                newinput_sp_tensor = spconv.SparseConvTensor(
                    features=newvoxel_features,
                    indices=newvoxel_coords.int(),
                    spatial_shape=self.sparse_shape,
                    batch_size=batch_size
                )
                newx = self.conv_input_2(newinput_sp_tensor)

                newx_conv1 = self.conv1_2(newx)
                newx_conv2 = self.conv2_2(newx_conv1)
                newx_conv3 = self.conv3_2(newx_conv2)
                newx_conv4 = self.conv4_2(newx_conv3)

                # for detection head
                # [200, 176, 5] -> [200, 176, 2]
                #newout = self.conv_out(newx_conv4)

                batch_dict.update({
                    #'encoded_spconv_tensor_mm': newout,
                    'encoded_spconv_tensor_stride_mm'+rot_num_id: 8
                })
                batch_dict.update({
                    'multi_scale_3d_features_mm'+rot_num_id: {
                        'x_conv1': newx_conv1,
                        'x_conv2': newx_conv2,
                        'x_conv3': newx_conv3,
                        'x_conv4': newx_conv4,
                    },
                    'multi_scale_3d_strides'+rot_num_id: {
                        'x_conv1': 1,
                        'x_conv2': 2,
                        'x_conv3': 4,
                        'x_conv4': 8,
                    }
                })

        return batch_dict

    def forward(self, batch_dict):
        if self.training:
            return self.forward_train(batch_dict)
        else:
            return self.forward_test(batch_dict)


class TeVoxelBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        self.return_num_features_as_dict = model_cfg.RETURN_NUM_FEATURES_AS_DICT
        self.out_features=model_cfg.OUT_FEATURES

        num_filters = model_cfg.NUM_FILTERS

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, num_filters[0], 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(num_filters[0]),
            nn.ReLU(),
        )
        block = post_act_block
        self.conv1 = spconv.SparseSequential(
            block(num_filters[0], num_filters[0], 3, norm_fn=norm_fn, padding=1, indice_key='conv1'),
        )
        self.conv2 = BasicBlock(num_filters[0], num_filters[1], norm_fn=norm_fn,  indice_key='conv2')
        self.conv3 = BasicBlock(num_filters[1], num_filters[2], norm_fn=norm_fn,  indice_key='conv3')
        self.conv4 = BasicBlock(num_filters[2], num_filters[3], norm_fn=norm_fn,  padding=(0, 1, 1),  indice_key='conv4')


        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(num_filters[3], self.out_features, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(self.out_features),
            nn.ReLU(),
        )
        if self.model_cfg.get('MM', False):
            self.conv_input_2 = spconv.SparseSequential(
                spconv.SubMConv3d(input_channels, num_filters[0], 3, padding=1, bias=False, indice_key='subm1_2'),
                norm_fn(num_filters[0]),
                nn.ReLU(),
            )

            self.conv1_2 = spconv.SparseSequential(
                block(num_filters[0], num_filters[0], 3, norm_fn=norm_fn, padding=1, indice_key='conv1_2'),
            )
            self.conv2_2 = BasicBlock(num_filters[0], num_filters[1], norm_fn=norm_fn, indice_key='conv2_2')
            self.conv3_2 = BasicBlock(num_filters[1], num_filters[2], norm_fn=norm_fn, indice_key='conv3_2')
            self.conv4_2 = BasicBlock(num_filters[2], num_filters[3], norm_fn=norm_fn, padding=(0, 1, 1),  indice_key='conv4_2')


        self.num_point_features = self.out_features

        if self.return_num_features_as_dict:
            num_point_features = {}
            num_point_features.update({
                'x_conv1': num_filters[0],
                'x_conv2': num_filters[1],
                'x_conv3': num_filters[2],
                'x_conv4': num_filters[3],
            })
            self.num_point_features = num_point_features


    def decompose_tensor(self, tensor, i, batch_size):
        input_shape = tensor.spatial_shape[2]
        begin_shape_ids = i * (input_shape // 4)
        end_shape_ids = (i + 1) * (input_shape // 4)
        x_conv3_features = tensor.features
        x_conv3_coords = tensor.indices

        mask = (begin_shape_ids < x_conv3_coords[:, 3]) & (x_conv3_coords[:, 3] < end_shape_ids)
        this_conv3_feat = x_conv3_features[mask]
        this_conv3_coords = x_conv3_coords[mask]
        this_conv3_coords[:, 3] -= i * (input_shape // 4)
        this_shape = [tensor.spatial_shape[0], tensor.spatial_shape[1], tensor.spatial_shape[2] // 4]

        this_conv3_tensor = spconv.SparseConvTensor(
            features=this_conv3_feat,
            indices=this_conv3_coords.int(),
            spatial_shape=this_shape,
            batch_size=batch_size
        )
        return this_conv3_tensor

    def forward_test(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """

        if 'transform_param' in batch_dict:
            trans_param = batch_dict['transform_param']
            rot_num = trans_param.shape[1]
        else:
            rot_num = 1

        all_lidar_feat = []
        all_lidar_coords = []

        new_shape = [self.sparse_shape[0], self.sparse_shape[1], self.sparse_shape[2] * 4]

        for i in range(rot_num):
            if i==0:
                rot_num_id = ''
            else:
                rot_num_id = str(i)

            voxel_features, voxel_coords = batch_dict['voxel_features'+rot_num_id], batch_dict['voxel_coords'+rot_num_id]

            all_lidar_feat.append(voxel_features)
            new_coord = voxel_coords.clone()
            new_coord[:, 3] += i*self.sparse_shape[2]
            all_lidar_coords.append(new_coord)
        batch_size = batch_dict['batch_size']

        all_lidar_feat = torch.cat(all_lidar_feat, 0)
        all_lidar_coords = torch.cat(all_lidar_coords)

        input_sp_tensor = spconv.SparseConvTensor(
            features=all_lidar_feat,
            indices=all_lidar_coords.int(),
            spatial_shape=new_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        out = self.conv_out(x_conv4)
        for i in range(rot_num):
            if i==0:
                rot_num_id = ''
            else:
                rot_num_id = str(i)

            this_conv3 = self.decompose_tensor(x_conv3, i, batch_size)
            this_conv4 = self.decompose_tensor(x_conv4, i, batch_size)
            this_out = self.decompose_tensor(out, i, batch_size)

            batch_dict.update({
                'encoded_spconv_tensor'+rot_num_id: this_out,
                'encoded_spconv_tensor_stride'+rot_num_id: 8,
            })
            batch_dict.update({
                'multi_scale_3d_features'+rot_num_id: {
                    'x_conv1': None,
                    'x_conv2': None,
                    'x_conv3': this_conv3,
                    'x_conv4': this_conv4,
                },
                'multi_scale_3d_strides'+rot_num_id: {
                    'x_conv1': 1,
                    'x_conv2': 2,
                    'x_conv3': 4,
                    'x_conv4': 8,
                }
            })


        if self.model_cfg.get('MM', False):
            all_mm_feat = []
            all_mm_coords = []
            for i in range(rot_num):
                if i == 0:
                    rot_num_id = ''
                else:
                    rot_num_id = str(i)

                newvoxel_features, newvoxel_coords = batch_dict['voxel_features_mm'+rot_num_id], batch_dict['voxel_coords_mm'+rot_num_id]

                all_mm_feat.append(newvoxel_features)
                new_mm_coord = newvoxel_coords.clone()
                new_mm_coord[:, 3] += i * self.sparse_shape[2]
                all_mm_coords.append(new_mm_coord)
            all_mm_feat = torch.cat(all_mm_feat, 0)
            all_mm_coords = torch.cat(all_mm_coords)

            newinput_sp_tensor = spconv.SparseConvTensor(
                features=all_mm_feat,
                indices=all_mm_coords.int(),
                spatial_shape=new_shape,
                batch_size=batch_size
            )

            newx = self.conv_input_2(newinput_sp_tensor)

            newx_conv1 = self.conv1_2(newx)
            newx_conv2 = self.conv2_2(newx_conv1)
            newx_conv3 = self.conv3_2(newx_conv2)
            newx_conv4 = self.conv4_2(newx_conv3)

            for i in range(rot_num):
                if i == 0:
                    rot_num_id = ''
                else:
                    rot_num_id = str(i)

                this_conv3 = self.decompose_tensor(newx_conv3, i, batch_size)
                this_conv4 = self.decompose_tensor(newx_conv4, i, batch_size)
                batch_dict.update({
                    'encoded_spconv_tensor_stride_mm'+rot_num_id: 8
                })
                batch_dict.update({
                    'multi_scale_3d_features_mm'+rot_num_id: {
                        'x_conv1': None,
                        'x_conv2': None,
                        'x_conv3': this_conv3,
                        'x_conv4': this_conv4,
                    },
                    'multi_scale_3d_strides'+rot_num_id: {
                        'x_conv1': 1,
                        'x_conv2': 2,
                        'x_conv3': 4,
                        'x_conv4': 8,
                    }
                })

        return batch_dict

    def forward_train(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        if 'transform_param' in batch_dict:
            trans_param = batch_dict['transform_param']
            rot_num = trans_param.shape[1]
        else:
            rot_num = 1


        for i in range(rot_num):
            if i==0:
                rot_num_id = ''
            else:
                rot_num_id = str(i)

            voxel_features, voxel_coords = batch_dict['voxel_features'+rot_num_id], batch_dict['voxel_coords'+rot_num_id]

            batch_size = batch_dict['batch_size']
            input_sp_tensor = spconv.SparseConvTensor(
                features=voxel_features,
                indices=voxel_coords.int(),
                spatial_shape=self.sparse_shape,
                batch_size=batch_size
            )
            x = self.conv_input(input_sp_tensor)

            x_conv1 = self.conv1(x)
            x_conv2 = self.conv2(x_conv1)
            x_conv3 = self.conv3(x_conv2)
            x_conv4 = self.conv4(x_conv3)

            # for detection head
            # [200, 176, 5] -> [200, 176, 2]
            out = self.conv_out(x_conv4)

            batch_dict.update({
                'encoded_spconv_tensor'+rot_num_id: out,
                'encoded_spconv_tensor_stride'+rot_num_id: 8,
            })
            batch_dict.update({
                'multi_scale_3d_features'+rot_num_id: {
                    'x_conv1': x_conv1,
                    'x_conv2': x_conv2,
                    'x_conv3': x_conv3,
                    'x_conv4': x_conv4,
                },
                'multi_scale_3d_strides'+rot_num_id: {
                    'x_conv1': 1,
                    'x_conv2': 2,
                    'x_conv3': 4,
                    'x_conv4': 8,
                }
            })

            if self.model_cfg.get('MM', False):
                newvoxel_features, newvoxel_coords = batch_dict['voxel_features_mm'+rot_num_id], batch_dict['voxel_coords_mm'+rot_num_id]

                newinput_sp_tensor = spconv.SparseConvTensor(
                    features=newvoxel_features,
                    indices=newvoxel_coords.int(),
                    spatial_shape=self.sparse_shape,
                    batch_size=batch_size
                )
                newx = self.conv_input_2(newinput_sp_tensor)

                newx_conv1 = self.conv1_2(newx)
                newx_conv2 = self.conv2_2(newx_conv1)
                newx_conv3 = self.conv3_2(newx_conv2)
                newx_conv4 = self.conv4_2(newx_conv3)

                # for detection head
                # [200, 176, 5] -> [200, 176, 2]
                #newout = self.conv_out(newx_conv4)

                batch_dict.update({
                    #'encoded_spconv_tensor_mm': newout,
                    'encoded_spconv_tensor_stride_mm'+rot_num_id: 8
                })
                batch_dict.update({
                    'multi_scale_3d_features_mm'+rot_num_id: {
                        'x_conv1': newx_conv1,
                        'x_conv2': newx_conv2,
                        'x_conv3': newx_conv3,
                        'x_conv4': newx_conv4,
                    },
                    'multi_scale_3d_strides'+rot_num_id: {
                        'x_conv1': 1,
                        'x_conv2': 2,
                        'x_conv3': 4,
                        'x_conv4': 8,
                    }
                })

        return batch_dict

    def forward(self, batch_dict):
        if self.training:
            return self.forward_train(batch_dict)
        else:
            return self.forward_test(batch_dict)



