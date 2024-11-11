from torch import Tensor
from torch.nn import Module, functional


class Similarity(Module):
    """Layer that using Cross-Correlation to compute
    similarity between features.

    Parameters
    ----------
    n_anchors_group : int
        Number of anchors in one group.
    """

    def __init__(self, n_anchors_group: int) -> None:
        super(Similarity, self).__init__()
        self.n_anchors_group = n_anchors_group

    def forward(self, template_features: Tensor, search_features: Tensor) -> Tensor:
        return functional.conv2d(
            search_features,
            template_features.reshape((self.n_anchors_group, 256, 4, 4)),
        )
