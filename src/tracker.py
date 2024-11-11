from torch import Tensor
from torch.nn import Module

from src.nn.backbone import AlexNetBackbone
from src.nn.branch import OutputBranch


class SiamRPN(Module):
    """An implementation of SiamRPN network.

    Parameters
    ----------
    n_anchors : int, default=20
        Number of anchors in the framework.

    Notes
    -----
    Implementation of the SiamRPN from the paper: "High Performance Visual Tracking
    with Siamese Region Proposal Network" -
    (https://ieeexplore.ieee.org/document/8579033).
    """

    def __init__(self, n_anchors: int = 20) -> None:
        super(SiamRPN, self).__init__()
        self.template_input: Module = AlexNetBackbone()
        self.search_input: Module = AlexNetBackbone()
        self.classification_branch: Module = OutputBranch(
            n_anchors=n_anchors, n_groups=2, n_channels=256
        )
        self.regression_branch: Module = OutputBranch(
            n_anchors=n_anchors, n_groups=4, n_channels=256
        )

    def forward(self, template: Tensor, search: Tensor) -> tuple[Tensor, Tensor]:
        template_features = self.template_input(template)
        search_features = self.search_input(search)
        return (
            self.classification_branch(template_features, search_features),
            self.regression_branch(template_features, search_features),
        )
