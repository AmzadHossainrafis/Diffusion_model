import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import logging
from Diffusion.utils.utils import save_images
class Trainer:
    """
    A class to encapsulate the training process of a diffusion model.

    Attributes:
        device (str): The device (CPU or CUDA) on which the training will run.
        dataloader (DataLoader): The DataLoader object providing the dataset.
        model (torch.nn.Module): The neural network model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer for training the model.
        mse (torch.nn.MSELoss): Mean Squared Error loss function.
        diffusion (Diffusion): The diffusion process object.
        logger (SummaryWriter): TensorBoard logger for tracking training progress.
        l (int): The length of the DataLoader (number of batches).

    Methods:
        setup_logging(name): Sets up logging with a specified name.
        train(epochs): Trains the model for a specified number of epochs.
        save_images(images, path): Saves images to a specified path. Placeholder for implementation.
    """

    def __init__(self, model, dataloader, diffusion, device=None):
        """
        Initializes the Trainer class with model, dataloader, diffusion process, and device.

        Parameters:
            model (torch.nn.Module): The model to be trained.
            dataloader (DataLoader): The DataLoader for the dataset.
            diffusion (Diffusion): The diffusion process object.
            device (str, optional): The device to run the training on. Defaults to CUDA if available, else CPU.
        """
        self.setup_logging("diffusion_unconditional")
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.dataloader = dataloader
        self.model = model.to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=3e-4)
        self.mse = nn.MSELoss()
        self.diffusion = diffusion
        self.logger = SummaryWriter(os.path.join("runs", "diffusion_unconditional"))
        self.l = len(dataloader)

    def setup_logging(self, name):
        """
        Sets up logging with a specified filename.

        Parameters:
            name (str): The name of the log file.
        """
        logging.basicConfig(filename=f'{name}.log', level=logging.INFO)

    def train(self, epochs=500):
        """
        Trains the model for a specified number of epochs.

        Parameters:
            epochs (int, optional): The number of epochs to train the model. Defaults to 500.
        """
        for epoch in range(epochs):
            logging.info(f"Starting epoch {epoch}:")
            pbar = tqdm(self.dataloader)
            for i, (images, _) in enumerate(pbar):
                images = images.to(self.device)
                t = self.diffusion.sample_timesteps(images.shape[0]).to(self.device)
                x_t, noise = self.diffusion.noise_images(images, t)
                predicted_noise = self.model(x_t, t)
                loss = self.mse(noise, predicted_noise)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pbar.set_postfix(MSE=loss.item())
                self.logger.add_scalar("MSE", loss.item(), global_step=epoch * self.l + i)

            sampled_images = self.diffusion.sample(self.model, n=images.shape[0])
            save_images(sampled_images, os.path.join("results", "diffusion_unconditional", f"{epoch}.jpg"))
            torch.save(self.model.state_dict(), os.path.join("models", "diffusion_unconditional", f"ckpt_{epoch}.pt"))

    
if __name__ == '__main__':
    import torchvision
    from Diffusion.components.data_transformation import transforms
    from Diffusion.components.models import UNet
    from Diffusion.components.diffusion import Diffusion
    from Diffusion.components.data import get_celeba_dataloader

    dataset = torchvision.datasets.CelebA(root="data", download=True, transform=transforms)
    dataloader = get_celeba_dataloader(dataset, batch_size=16, num_workers=4)
    model = UNet(c_in=3, c_out=3, time_dim=256)
    diffusion = Diffusion()
    trainer = Trainer(model, dataloader, diffusion)
    trainer.train(epochs=10)
    