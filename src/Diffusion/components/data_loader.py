from torch.utils.data import DataLoader
import torchvision
from Diffusion.components.data_transformation import transforms



def get_data(dataset_dir, transforms=transforms):
    """
    Returns a DataLoader object for the dataset at the specified directory.

    This function takes the path to a dataset directory and a set of transformations,
    and returns a DataLoader object for the dataset. The DataLoader object is created
    using the ImageFolder class from torchvision.datasets module.

    Parameters:
        dataset_dir (str): The path to the dataset directory.
        transforms (torchvision.transforms.Compose): A composition of image transformations.

    Returns:
        DataLoader: A DataLoader object for the dataset.
    """

    dataset = torchvision.datasets.ImageFolder(dataset_dir, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    return dataloader
