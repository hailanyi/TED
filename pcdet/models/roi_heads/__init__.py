from .roi_head_template import RoIHeadTemplate

from .ted_head import TEDMHead, TEDSHead

__all__ = {
    'RoIHeadTemplate': RoIHeadTemplate,
    'TEDSHead': TEDSHead,
    'TEDMHead': TEDMHead
}
