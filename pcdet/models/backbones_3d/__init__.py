from .pointnet2_backbone import PointNet2Backbone, PointNet2MSG
from .spconv_backbone import TeMMVoxelBackBone8x,TeVoxelBackBone8x
__all__ = {
    'TeMMVoxelBackBone8x': TeMMVoxelBackBone8x,
    'TeVoxelBackBone8x': TeVoxelBackBone8x,
    'PointNet2Backbone': PointNet2Backbone,
    'PointNet2MSG': PointNet2MSG,
}
