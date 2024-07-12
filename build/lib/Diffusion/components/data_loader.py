import torchvision
from torch.utils.data import DataLoader
from Diffusion.components.data_transformation import transforms
from Diffusion.utils.logger import logger
from Diffusion.utils.utils import read_config

data_config = read_config("/home/amzad/Desktop/diffusion/config/config.yaml")["Train_config"]

def get_data(dataset_dir, batch_size=data_config['batch_size'], shuffle=True, transforms=transforms):
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
    logger.info(f"...dataloader....")
    logger.info(f"current transforms {transforms}")

    dataset = torchvision.datasets.ImageFolder(dataset_dir, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    logger.info("data loading done ")

    return dataloader
