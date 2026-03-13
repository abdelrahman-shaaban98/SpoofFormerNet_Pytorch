from pathlib import Path
from typing import Optional, List, Tuple
import random
from PIL import Image

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from visualize import visualize_batch


IMG_EXTENSION = ".jpg"
LABEL_MAP = {"real": 1, "fake": 0}  

color_mean = (0.5, 0.5, 0.5)
color_std  = (0.5, 0.5, 0.5)
depth_mean = (0.5,)
depth_std  = (0.5,)
image_size = 256


def _extract_label(filename: str) -> int:
    """
    Get label (fake or real) from the image name.
    """
    for token, label in LABEL_MAP.items():
        if token in filename:
            return label

    raise ValueError(
        f"Cannot determine label from filename '{filename}'. "
        "Filename must contain 'fake' or 'real'."
    )


def _collect_pairs(
    color_dir: str,
    depth_dir: str,
) -> List[Tuple[Path, Path, int]]:
    """
    Walk color_dir, find matching depth files, extract labels.
    """
    color_dir = Path(color_dir)
    depth_dir = Path(depth_dir)

    if not color_dir.exists():
        raise FileNotFoundError(f"Color directory not found: {color_dir}")
    if not depth_dir.exists():
        raise FileNotFoundError(f"Depth directory not found: {depth_dir}")

    pairs = []
    missing_depth = []
    bad_labels    = []

    for color_path in sorted(color_dir.iterdir()):
        if IMG_EXTENSION not in color_path.suffix.lower():
            continue

        depth_path = depth_dir / color_path.name
        if not depth_path.exists():
            missing_depth.append(color_path.name)
            continue

        try:
            label = _extract_label(color_path.name)
        except ValueError as exc:
            bad_labels.append(str(exc))
            continue

        pairs.append((color_path, depth_path, label))

    if missing_depth:
        print(f"[WARNING] {len(missing_depth)} color images have no matching depth file "
              f"and were skipped. First few: {missing_depth[:5]}")
    
    if bad_labels:
        print(f"[WARNING] {len(bad_labels)} files had unrecognisable labels and were "
              f"skipped. First few: {bad_labels[:5]}")
    
    if not pairs:
        raise RuntimeError(
            f"No valid (color, depth, label) pairs found in:\n"
            f"  color: {color_dir}\n  depth: {depth_dir}"
        )

    return pairs


class PairedTransform:
    """
    Applies identical random spatial augmentations to both the color image
    and the depth map, then applies separate photometric transforms to color.
    """
    def __init__(self, augment: bool = True,):
        self.augment    = augment
        self.color_normalize = T.Normalize(mean=color_mean, std=color_std)
        self.depth_normalize = T.Normalize(mean=depth_mean, std=depth_std)

    def __call__(
        self,
        color_img: Image.Image,
        depth_img: Image.Image,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Resize both to a slightly larger size before random crop
        resize_to = int(image_size * 1.21875)  # e.g. 312 when image_size=256

        color_img = TF.resize(color_img, (resize_to, resize_to), interpolation=Image.BILINEAR)
        depth_img = TF.resize(depth_img, (resize_to, resize_to), interpolation=Image.BILINEAR)

        if self.augment:
            i, j, h, w = T.RandomCrop.get_params(
                color_img, output_size=(image_size, image_size)
            )
            color_img = TF.crop(color_img, i, j, h, w)
            depth_img = TF.crop(depth_img, i, j, h, w)

            if random.random() > 0.5:
                color_img = TF.hflip(color_img)
                depth_img = TF.hflip(depth_img)

            if random.random() > 0.5:
                color_img = TF.vflip(color_img)
                depth_img = TF.vflip(depth_img)

            angle = random.uniform(-15, 15)
            color_img = TF.rotate(color_img, angle)
            depth_img = TF.rotate(depth_img, angle)

        else:
            color_img = TF.center_crop(color_img, image_size)
            depth_img = TF.center_crop(depth_img, image_size)

        color_t = TF.to_tensor(color_img)        # 3, H, W  in [0, 1]
        depth_t = TF.to_tensor(depth_img)        # C, H, W  in [0, 1]

        # Keep depth as single channel (grayscale or first channel)
        # print(depth_t.shape[0])
        if depth_t.shape[0] > 1:
            depth_t = depth_t.mean(dim=0, keepdim=True)

        # Photometric augmentation on color only
        if self.augment:
            color_t = T.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1
            )(color_t)

        color_t = self.color_normalize(color_t)
        depth_t = self.depth_normalize(depth_t)

        return color_t, depth_t
    

class SpoofDataset(Dataset):
    """
    Dataset for paired (color, depth) face anti-spoofing data.
    """
    def __init__(
        self,
        color_dir: str,
        depth_dir: str,
        transform: Optional[PairedTransform] = None,
        verbose: bool = True,
    ):
        self.transform = transform
        self.samples   = _collect_pairs(color_dir, depth_dir)

        if verbose:
            n_real = sum(1 for _, _, lbl in self.samples if lbl == 1)
            n_fake = len(self.samples) - n_real
            print(
                f"[SpoofDataset] Loaded {len(self.samples)} pairs from '{color_dir}'\n"
                f"  Real (genuine): {n_real} | Fake (spoof): {n_fake}"
            )


    def __len__(self) -> int:
        return len(self.samples)


    def get_labels(self) -> List[int]:
        """Return all labels as a plain list (useful for WeightedRandomSampler)."""
        return [label for _, _, label in self.samples]


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        color_path, depth_path, label = self.samples[idx]

        color_img = Image.open(color_path).convert("RGB")
        depth_img = Image.open(depth_path).convert("RGB")   # will be collapsed to 1 ch in the dataloader transformer

        if self.transform is not None:
            color_t, depth_t = self.transform(color_img, depth_img)
        else:
            color_t = TF.to_tensor(color_img)
            # Keep depth as single channel (grayscale or first channel)
            depth_t = TF.to_tensor(depth_img).mean(dim=0, keepdim=True)

        return color_t, depth_t, label
    

def _make_weighted_sampler(dataset: SpoofDataset) -> WeightedRandomSampler:
    """
    Creates a WeightedRandomSampler so that each class is sampled
    with equal probability regardless of dataset imbalance.
    """
    labels        = dataset.get_labels()
    class_counts  = [labels.count(c) for c in range(2)]          
    class_weights = [1.0 / max(c, 1) for c in class_counts]
    sample_weights = torch.tensor([class_weights[label] for label in labels], dtype=torch.float)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def get_dataloaders(
    train_color_dir: str,
    train_depth_dir: str,
    test_color_dir: str,
    test_depth_dir: str,
    batch_size: int    = 32,
    num_workers: int   = 4,
    balance_classes: bool = True,
    pin_memory: bool   = True,
    verbose: bool      = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build and return (train_loader, test_loader).
    """
    train_transform = PairedTransform(augment=True)
    test_transform = PairedTransform(augment=False)

    train_dataset = SpoofDataset(
        color_dir=train_color_dir,
        depth_dir=train_depth_dir,
        transform=train_transform,
        verbose=verbose,
    )
    test_dataset = SpoofDataset(
        color_dir=test_color_dir,
        depth_dir=test_depth_dir,
        transform=test_transform,
        verbose=verbose,
    )

    # Sampler
    if balance_classes:
        sampler   = _make_weighted_sampler(train_dataset)
        train_shuffle = False  # sampler is mutually exclusive with shuffle
    else:
        sampler       = None
        train_shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=train_shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,       
        persistent_workers=num_workers > 0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    if verbose:
        print(
            f"\n[DataLoaders]\n"
            f"  Train batches : {len(train_loader)}  "
            f"  Test  batches : {len(test_loader)}"
        )

    return train_loader, test_loader


if __name__ == "__main__":

    train_color = "data/train_img/color"
    train_depth = "data/train_img/depth_generated"
    test_color  = "data/test_img/color"
    test_depth  = "data/test_img/depth_generated"

    train_loader, test_loader = get_dataloaders(
        train_color_dir = train_color,
        train_depth_dir = train_depth,
        test_color_dir  = test_color,
        test_depth_dir  = test_depth,
        batch_size      = 4,
        num_workers     = 8,   
        verbose         = True,
    )

    rgb, depth, labels = next(iter(train_loader))

    print(f"\nBatch check:")
    print(f"  RGB   shape : {tuple(rgb.shape)}")
    print(f"  Depth shape : {tuple(depth.shape)}")
    print(f"  Labels      : {labels.tolist()}")

    visualize_batch(rgb, depth, labels, save_path="images/batch_visualization.png")



