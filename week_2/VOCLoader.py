import torchvision.transforms as transforms
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader


def load(train_batch_size=4, test_batch_size=1) -> tuple[DataLoader, DataLoader]:
    """
    Load Pascal VOC 2012 dataset and return DataLoaders for training and testing.

    This function applies transformations to the images and masks, and loads the Pascal VOC dataset.
    It returns two PyTorch DataLoaders, one for training data and one for testing data.
    
    Note: The masks are not one-hot encoded, and the pixel values are normalized from [0, 255] to [0, 1].

    Parameters:
        train_batch_size (int): The batch size for the training DataLoader.
        test_batch_size (int): The batch size for the testing DataLoader.

    Returns:
        tuple[DataLoader, DataLoader]: A tuple containing the training DataLoader and testing DataLoader.
    """
    # Define data transformation
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Load Pascal VOC dataset
    train_dataset = VOCSegmentation(root="./data", year='2012', image_set='train', download=True, transform=image_transform, target_transform=mask_transform)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    test_dataset = VOCSegmentation(root="./data", year='2012', image_set='val', download=True, transform=image_transform, target_transform=mask_transform)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)
    
    print("Successfully loaded Pascal VOC dataset")
    print(f"Image shape: {train_dataset[0][0].shape}")
    print(f"Mask shape: {train_dataset[0][1].shape}")
    
    return train_loader, test_loader
