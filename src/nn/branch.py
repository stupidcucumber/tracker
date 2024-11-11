from torch import Tensor
from torch.nn import LazyConv2d, Module

from src.nn.similarity import Similarity


class OutputBranch(Module):
    """Output branch that provides info either on each of the anchor boxes
    on the transition and scale or classification whether it is foreground or
    background.

    Parameters
    ----------
    n_anchors : int, default=20
        Number of anchor boxes in each cell.
    """

    def __init__(
        self,
        n_anchors: int = 20,
        n_channels: int = 256,
        n_groups: int = 4,
    ) -> None:
        super(OutputBranch, self).__init__()
        self.n_groups = n_groups
        self.n_anchors = n_anchors
        self.template_conv = LazyConv2d(
            out_channels=n_groups * n_anchors * n_channels, kernel_size=(3, 3)
        )
        self.search_conv = LazyConv2d(out_channels=n_channels, kernel_size=(3, 3))
        self.similarity = Similarity(n_anchors_group=n_anchors * n_groups)

    def forward(self, template_features: Tensor, search_features: Tensor) -> Tensor:
        template_features = self.template_conv(template_features)
        search_features = self.search_conv(search_features)
        aggregation: Tensor = self.similarity(template_features, search_features)
        return aggregation.reshape((self.n_groups * self.n_anchors, 17, 17))
