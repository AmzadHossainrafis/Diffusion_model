import os
import sys
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
from Diffusion.utils.utils import read_config
from Diffusion.utils.logger import logger
from Diffusion.utils.exception import CustomException
from Diffusion.components.noise_sheduler import Diffusion
from Diffusion.components.custom import EMA


train_config =read_config(f"{Path.cwd().parents[2]}/config/config.yaml")[
    "Train_config"
]


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
        save_images(images, path): Saves images to a specified path. Placeholder for
        implementation.
    """

    def __init__(
        self, model, dataloader, diffusion, ema=train_config["ema_enable"], device=None
    ):
        """
        Initializes the Trainer class with model, dataloader, diffusion process, and device.

        Parameters:
            model (torch.nn.Module): The model to be trained.
            dataloader (DataLoader): The DataLoader for the dataset.
            diffusion (Diffusion): The diffusion process object.
            device (str, optional): The device to run the training on. Defaults to CUDA if available, else CPU.
        """

        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dataloader = dataloader
        self.model = model.to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=train_config["lr"])
        self.mse = nn.MSELoss()
        self.diffusion = diffusion
        self.l = len(dataloader)
        self.ema = ema
        if self.ema:
            self.Ema = EMA(0.99)
            self.Ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)

    def train(self, epochs=train_config["epochs"]):
        """
        Trains the model for a specified number of epochs.

        Parameters:
            epochs (int, optional): The number of epochs to train the model. Defaults to 500.
        """

        try:
            logger.info("Initializing Trainer")
            logger.info(f"train_config {train_config}")

            for epoch in range(epochs):
                # logger.info(f"Starting epoch {epoch}:")
                pbar = tqdm(self.dataloader)
                for i, (images, _) in enumerate(pbar):
                    images = images.to(self.device)
                    t = self.diffusion.sample_timesteps(images.shape[0]).to(self.device)
                    x_t, noise = self.diffusion.noise_images(images, t)
                    predicted_noise = self.model(x_t, t)
                    loss = self.mse(noise, predicted_noise)
                    # logger.info(f"Epoch {epoch}, Batch {i}: MSE Loss = {loss.item()}")
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    if self.ema:
                        self.Ema.step_ema(self.Ema_model, self.model)

                    pbar.set_postfix(MSE=loss.item())

                # if epoch % train_config["intrable"] == 0:
                #     sampled_images = self.diffusion.sample(
                #         self.model, n=train_config["num_sample"]
                #     )
                #     save_images(
                #         sampled_images,
                #         os.path.join(
                #             train_config["figs"], f"prediction_on_{epoch}.jpg"
                #         ),
                #     )
                if self.ema:
                    torch.save(
                        self.Ema_model.state_dict(),
                        os.path.join(
                            train_config["model_ckpt"],
                            f"{train_config['train_name']}_ema_ckpt_.pt",
                        ),
                    )
                else:
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(
                            train_config["model_ckpt"],
                            f"{train_config['train_name']}_ckpt_.pt",
                        ),
                    )
        except Exception as e:
            logger.info(f"Error  occared {e}")
            raise CustomException(e, sys)
