import torchvision
from pathlib import Path
from Diffusion.utils.utils import read_config


transform_config = read_config(f"{Path.cwd().parents[2]}/config/config.yaml")[
    "Transform_config"
]


transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(
            transform_config["resize"]
        ),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(
            transform_config["crop"], scale=(0.8, 1.0)
        ),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            transform_config["mean"], transform_config["std"]
        ),
    ]
)
