import sys
import torch
from tqdm import tqdm
from Diffusion.utils.logger import logger
from Diffusion.utils.exception import CustomException
from Diffusion.utils.utils import read_config


def prection(model , number_of_sample):
    
