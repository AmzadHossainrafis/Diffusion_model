
from Diffusion.components.models import *
from Diffusion.components.data_loader import get_data
from Diffusion.components.noise_sheduler import Diffusion
from Diffusion.components.trainer import Trainer

data = get_data(train_config["dataset"])
model = UNet()
diffusion = Diffusion()
trainer = Trainer(model, data, diffusion)
trainer.train()

