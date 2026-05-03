import os
from typing import Tuple, List, Optional

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms, datasets


def get_dataloaders(
	data_dir: str = "SkinDisease",
	train_folder: str = "train",
	test_folder: str = "test",
	batch_size: int = 32,
	val_split: float = 0.15,
	num_workers: int = 0,
	img_size: int = 224,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], List[str]]:
	"""Create train/val/test DataLoaders using ImageFolder and WeightedRandomSampler.

	Returns (train_loader, val_loader, test_loader_or_None, class_names)
	"""
	train_dir = os.path.join(data_dir, train_folder)
	test_dir = os.path.join(data_dir, test_folder)

	if not os.path.isdir(train_dir):
		raise FileNotFoundError(f"Train directory not found: {train_dir}")

	# ImageNet normalization (matches pretrained models)
	imagenet_mean = [0.485, 0.456, 0.406]
	imagenet_std = [0.229, 0.224, 0.225]

	# Training augmentations
	transform_train = transforms.Compose([
		transforms.Resize(256),
		transforms.RandomResizedCrop(img_size, scale=(0.80, 1.0)),
		transforms.RandomHorizontalFlip(),
		transforms.RandomRotation(20),
		transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
		transforms.ToTensor(),
		transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
	])

	# Validation/test preprocessing (no augmentation)
	transform_val = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(img_size),
		transforms.ToTensor(),
		transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
	])

	# Separate datasets to apply different transforms
	dataset_train = datasets.ImageFolder(train_dir, transform=transform_train)
	dataset_val = datasets.ImageFolder(train_dir, transform=transform_val)

	pin_memory = torch.cuda.is_available()
	num_samples = len(dataset_train)

	# Create reproducible train/validation split
	split = int(val_split * num_samples)
	generator = torch.Generator().manual_seed(42)
	perm = torch.randperm(num_samples, generator=generator).tolist()

	val_idx = perm[:split]
	train_idx = perm[split:]

	train_subset = Subset(dataset_train, train_idx)
	val_subset = Subset(dataset_val, val_idx)

	# Compute class distribution for weighted sampling
	targets = [dataset_train.samples[i][1] for i in train_idx]

	class_counts = {}
	for t in targets:
		class_counts[t] = class_counts.get(t, 0) + 1

	num_classes = len(dataset_train.classes)

	# Avoid division by zero for missing classes
	class_counts_list = [class_counts.get(i, 0) for i in range(num_classes)]
	class_weights = [0.0 if c == 0 else 1.0 / c for c in class_counts_list]

	# Assign weight to each sample
	sample_weights = [class_weights[t] for t in targets]

	sampler = WeightedRandomSampler(
		weights=sample_weights,
		num_samples=len(sample_weights),
		replacement=True
	)

	# Training loader uses sampler instead of shuffle
	train_loader = DataLoader(
		train_subset,
		batch_size=batch_size,
		sampler=sampler,
		num_workers=num_workers,
		pin_memory=pin_memory,
	)

	# Validation loader (no shuffle)
	val_loader = DataLoader(
		val_subset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=pin_memory,
	)

	# Optional test loader if test folder exists
	test_loader = None
	if os.path.isdir(test_dir):
		test_dataset = datasets.ImageFolder(test_dir, transform=transform_val)
		test_loader = DataLoader(
			test_dataset,
			batch_size=batch_size,
			shuffle=False,
			num_workers=num_workers,
			pin_memory=pin_memory,
		)

	return train_loader, val_loader, test_loader, dataset_train.classes


if __name__ == "__main__":
	# Quick local check
	try:
		t, v, te, classes = get_dataloaders()
		print("Created DataLoaders:")
		print(" - train batches:", len(t))
		print(" - val batches:", len(v))
		if te is not None:
			print(" - test batches:", len(te))
		print("Classes:", classes)
	except FileNotFoundError as e:
		print(e)