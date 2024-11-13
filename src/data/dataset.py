from itertools import batched
from pathlib import Path
from typing import Literal

import albumentations
import albumentations.pytorch
import cv2
import numpy as np
import torch
from got10k.datasets import GOT10k
from torch import Tensor
from torch.utils.data import Dataset


class DatasetGot10k(Dataset):
    """Class provides an implementation of gathering data
    from the Got10k dataset and preparing it for the
    training of the Tracker.

    Parameters
    ----------
    root_dir : Path
        Path to the root directory of the Got10k dataset.
    subset : Literal["train", "val"], default="train"
        Subset to be used.
    template_image_size: tuple[int, int], default=(127, 127)
        Image size of a template in the end.
    search_image_size: tuple[int, int], default=(255, 255)
        Image size of a search image in the end.
    """

    def __init__(
        self,
        root_dir: Path,
        subset: Literal["train", "val"] = "train",
        template_image_size: tuple[int, int] = (127, 127),
        search_image_size: tuple[int, int] = (255, 255),
    ) -> None:
        super(DatasetGot10k, self).__init__()

        self.template_transform = albumentations.Compose(
            [
                albumentations.Resize(
                    height=template_image_size[1], width=template_image_size[0]
                ),
                albumentations.Normalize(),
                albumentations.pytorch.ToTensorV2(),
            ]
        )
        self.search_transform = albumentations.Compose(
            [
                albumentations.Resize(
                    height=search_image_size[1], width=search_image_size[0]
                ),
                albumentations.Normalize(),
                albumentations.pytorch.ToTensorV2(),
            ],
            bbox_params=albumentations.BboxParams(
                format="coco", label_fields=["classes"]
            ),
        )

        self.data = self._prepare_data(root_dir=root_dir, subset=subset)

    def _read_image(self, image_path: str) -> cv2.Mat:
        image = cv2.imread(image_path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _prepare_data(
        self, root_dir: Path, subset: Literal["train", "val"]
    ) -> list[tuple[tuple[str, np.ndarray], tuple[str, np.ndarray]]]:
        got10k_dataset = GOT10k(root_dir=root_dir, subset=subset)

        result = []
        for video_index in range(len(got10k_dataset)):
            video = got10k_dataset[video_index]
            _temp = [
                batch for batch in batched(zip(video[0], video[1]), 2) if len(batch) > 1
            ]
            result.extend(_temp)

        return result

    def __len__(self) -> int:
        return len(self.data)

    def _prepare_template_image(
        self, image_path: str, annotation: np.ndarray
    ) -> torch.Tensor:
        """Reads image and translates it to the template
        image.

        Parameters
        ----------
        image_path : str
            Path to the image on the local computer.
        annotation : np.ndarray
            Annotation in the format [xmin, ymin, width, height].

        Returns
        -------
        torch.Tensor
            Template image.
        """
        image = self._read_image(image_path)
        image = image[
            int(annotation[1]) : int(annotation[1] + annotation[3]),
            int(annotation[0]) : int(annotation[0] + annotation[2]),
        ]
        return self.template_transform(image=image)["image"]

    def _prepare_search_image(
        self, image_path: str, annotation: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reads image and translates it to the search
        image.

        Parameters
        ----------
        image_path : str
            Path to the image on the local computer.
        annotation : np.ndarray
            Annotation in the format [xmin, ymin, width, height].

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Search image and corresponding ground truth
            box.
        """
        image = self._read_image(image_path)
        _transform_result = self.search_transform(
            image=image, bboxes=np.expand_dims(annotation, axis=0), classes=[0]
        )
        return (
            _transform_result["image"],
            torch.as_tensor(_transform_result["bboxes"][0]),
        )

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        """Extracts next pair of images to train the similarity.

        Parameters
        ----------
        index : int
            Index of the template image.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            In order: template image, search image and ground-truth
            on the search image.
        """
        (
            (template_image_path, template_annotation),
            (search_image_path, search_annotation),
        ) = self.data[index]

        template_image = self._prepare_template_image(
            image_path=template_image_path, annotation=template_annotation
        )

        search_image, ground_truth_annotation = self._prepare_search_image(
            image_path=search_image_path, annotation=search_annotation
        )

        return (template_image, search_image, ground_truth_annotation)
