import torch
from tqdm import tqdm
from Diffusion.utils.logger import logger
from Diffusion.utils.exception import CustomException
import sys
from Diffusion.utils.utils import read_config

noise_schedule_config = read_config("/home/amzad/Desktop/diffusion/config/config.yaml")[
    "Noise_schedule"
]


class Diffusion:
    """
    A class representing a diffusion model for image generation.

    Attributes:
        noise_steps (int): The number of steps to perform the noise schedule over.
        beta_start (float): The starting value of beta for the noise schedule.
        beta_end (float): The ending value of beta for the noise schedule.
        img_size (int): The size of the images to be generated.
        device (str): The device (e.g., 'cuda' or 'cpu') to perform computations on.
        beta (torch.Tensor): The noise schedule, a tensor of betas linearly spaced between beta_start and beta_end.
        alpha (torch.Tensor): 1 - beta, representing the opposite of the noise schedule.
        alpha_hat (torch.Tensor): The cumulative product of alpha, used in the denoising process.

    Methods:
        prepare_noise_schedule(): Prepares the noise schedule as a linear interpolation between beta_start and beta_end.
        noise_images(x, t): Applies noise to images x at time steps t, based on the noise schedule.
        sample_timesteps(n): Randomly samples n timesteps within the noise schedule.
        sample(model, n): Generates n new images using the provided model by reversing the diffusion process.
    """

    def __init__(
        self,
        noise_steps=noise_schedule_config["steps"],
        beta_start=noise_schedule_config["start"],
        beta_end=noise_schedule_config["end"],
        img_size=256,
        device= noise_schedule_config["device"]
    ):
        """
        Initializes the Diffusion model with the specified parameters and computes the noise schedule.

        Parameters:
            noise_steps (int): The number of steps for the noise schedule.
            beta_start (float): The starting beta value for the noise schedule.
            beta_end (float): The ending beta value for the noise schedule.
            img_size (int): The size of the images to generate.
            device (str): The device to use for computations.
        """
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        """
        Prepares the noise schedule as a linear interpolation of beta values.

        Returns:
            torch.Tensor: A tensor of beta values linearly spaced between beta_start and beta_end.
        """
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        """
        Applies noise to images x at time steps t according to the noise schedule.

        Parameters:
            x (torch.Tensor): The images to apply noise to.
            t (torch.Tensor): The time steps at which to apply the noise.

        Returns:
            tuple: A tuple containing the noised images and the noise itself.
        """
        logger.info("sapling new noisy images ")
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[
            :, None, None, None
        ]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        """
        Samples n random timesteps within the noise schedule.

        Parameters:
            n (int): The number of timesteps to sample.

        Returns:
            torch.Tensor: A tensor of randomly sampled timesteps.
        """
        logger.info('creating new timestep ')
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        """
        Generates n new images by reversing the diffusion process using the provided model.

        Parameters:
            model (torch.nn.Module): The model to use for generating images.
            n (int): The number of images to generate.

        Returns:
            torch.Tensor: A tensor of generated images.
        """
        logger.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            try:  
                logger.info(f'sampaling image : {n}')
                x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
                for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                    t = (torch.ones(n) * i).long().to(self.device)
                    predicted_noise = model(x, t)
                    alpha = self.alpha[t][:, None, None, None]
                    alpha_hat = self.alpha_hat[t][:, None, None, None]
                    beta = self.beta[t][:, None, None, None]
                    if i > 1:
                        noise = torch.randn_like(x)
                    else:
                        noise = torch.zeros_like(x)
                    x = (
                        1
                        / torch.sqrt(alpha)
                        * (
                            x
                            - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise
                        )
                        + torch.sqrt(beta) * noise
                    )
                model.train()
                x = (x.clamp(-1, 1) + 1) / 2
                x = (x * 255).type(torch.uint8)
            except Exception as e:
                logger.error(f"Error while sampling images: {e}")
                raise CustomException(e, sys)
        return x
