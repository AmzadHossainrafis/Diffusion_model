import sys
import torch
from tqdm import tqdm
from Diffusion.utils.logger import logger
from Diffusion.utils.exception import CustomException


class PredictionPipeline:
    def __init__(
        self,
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        img_size=256,
        device="cuda",
    ):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def result(self, model, n):
        # clear cache
        # torch.cuda.empty_cache()
        # #clear memory
        # torch.cuda.memory_summary(device=None, abbreviated=False)
        logger.info(f"Sampling {n} new images....")
        model.eval()
        try:
            with torch.no_grad():
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
                            - ((1 - alpha) / (torch.sqrt(1 - alpha_hat)))
                            * predicted_noise
                        )
                        + torch.sqrt(beta) * noise
                    )
            x = (x.clamp(-1, 1) + 1) / 2
            x = (x * 255).type(torch.uint8)
        except Exception as e:
            logger.info(f"Error  occared {e}")
            raise CustomException(e, sys)
        return x


if __name__ == "__main__":
    # clear cache memory
    torch.cuda.empty_cache()

    from Diffusion.components.models import *

    model = UNet().to("cuda")
    model_ckpt = "/home/amzad/Desktop/diffusion/artifacts/model_ckpt/flower.pt"
    model.load_state_dict(torch.load(model_ckpt))
    # model = model.to("cuda")

    pipeline = PredictionPipeline(img_size=64)
    result = pipeline.result(model, 7)

    from Diffusion.utils.utils import save_images, plot_images

    fig_path = "/home/amzad/Desktop/diffusion/fig/result_14.jpg"
    save_images(result, fig_path)
