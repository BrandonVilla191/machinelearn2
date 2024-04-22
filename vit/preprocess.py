import torch
from torchvision.transforms import v2
from torchvision import datasets
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader


def create_training_transformations(image_path):
    augmented_transform_1 = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32),
            v2.RandomResizedCrop(224),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    augmented_transform_2 = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32),
            v2.RandomResizedCrop(224),
            v2.RandomRotation(10),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    augmented_transform_3 = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32),
            v2.RandomResizedCrop(224),
            v2.RandomRotation(10),
            v2.RandomHorizontalFlip(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    original_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32)])

    dataset_path = image_path
    augmented_dataset_1 = datasets.ImageFolder(
        root=dataset_path, transform=augmented_transform_1
    )
    augmented_dataset_2 = datasets.ImageFolder(
        root=dataset_path, transform=augmented_transform_2
    )
    augmented_dataset_3 = datasets.ImageFolder(
        root=dataset_path, transform=augmented_transform_3
    )
    original_dataset = datasets.ImageFolder(
        root=dataset_path, transform=original_transform
    )

    combined_dataset = ConcatDataset(
        [
            original_dataset,
            augmented_dataset_1,
            augmented_dataset_2,
            augmented_dataset_3,
        ]
    )
    batch_size = 32
    data_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    return data_loader
