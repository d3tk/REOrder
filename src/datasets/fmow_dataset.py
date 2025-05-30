import os
import json
from typing import Any, Dict, Optional, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import Dataset

# Increase PIL's image size limit to handle large satellite images
Image.MAX_IMAGE_PIXELS = 800 * 10**6  # should be > 620 MPix


class FMoWDataset(Dataset):
    """FMoW (Functional Map of the World) dataset.

    This dataset reads a text file containing relative paths to images, with one path per line.
    The category is inferred from the top-level folder name. For each image, there is an
    associated JSON file containing metadata.

    Example path format:
        airport/airport_0/airport_0_0_rgb.jpg
    """

    def __init__(
        self,
        root_dir: Optional[str] = None,
        split_file: Optional[str] = None,
        transform: Optional[Any] = None,
        return_metadata: bool = False,
    ) -> None:
        """Initialize FMoW dataset.

        Args:
            root_dir (Optional[str]): Base directory containing the dataset.
            split_file (Optional[str]): Path to text file with image paths.
            transform (Optional[Any]): Transform to apply to each image.
            return_metadata (bool): Whether to load and return JSON metadata.

        Raises:
            FileNotFoundError: If split_file or root_dir doesn't exist.
            ValueError: If split_file is empty or has invalid paths.
        """
        self.split_file = split_file
        self.root_dir = root_dir
        self.transform = transform
        self.return_metadata = return_metadata

        # Read and validate paths
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(self.split_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            raise ValueError(f"Split file is empty: {split_file}")

        # Sort for reproducibility
        self.samples = sorted(lines)

        # Build category mappings
        all_cats = set()
        for rel_path in self.samples:
            if not os.path.exists(os.path.join(root_dir, rel_path)):
                raise ValueError(f"Image not found: {rel_path}")
            category = rel_path.split("/")[0]
            all_cats.add(category)

        self.categories = sorted(all_cats)
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        self.idx_to_category = {idx: cat for cat, idx in self.category_to_idx.items()}

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int, Dict[str, Any]]]:
        """Get a dataset item.

        Args:
            idx (int): Index of the item to get.

        Returns:
            Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int, Dict[str, Any]]]:
                If return_metadata is False: (image, label)
                If return_metadata is True: (image, label, metadata)

        Raises:
            FileNotFoundError: If image or metadata file doesn't exist.
            ValueError: If image path is invalid.
        """
        rel_path = self.samples[idx]
        abs_img_path = os.path.join(self.root_dir, rel_path)

        # Get category and label
        category = rel_path.split("/")[0]
        label = self.category_to_idx[category]

        # Get metadata path
        rel_json_path = rel_path.replace("_rgb.jpg", "_rgb.json")
        abs_json_path = os.path.join(self.root_dir, rel_json_path)

        # Load image
        try:
            image = Image.open(abs_img_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"Failed to load image {abs_img_path}: {e}")

        # Load metadata if requested
        metadata = None
        if self.return_metadata:
            try:
                with open(abs_json_path, "r") as f:
                    metadata = json.load(f)
            except Exception as e:
                raise FileNotFoundError(f"Failed to load metadata {abs_json_path}: {e}")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        if self.return_metadata:
            return image, label, metadata
        return image, label
