# Diffusion model implimentation 
paper link : [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

## Description

Diffusion models are a class of generative models that excel in image generation by transforming random noise into detailed images through a process inspired by physical diffusion. This involves a forward process where images are gradually noised, and a reverse process where a neural network learns to denoise, effectively generating images from noise.

Key points about diffusion models include:

- **High-Quality Images:** They produce high-resolution, realistic images.
- **Versatile Applications:** Useful for tasks like image synthesis, super-resolution, inpainting, and conditional generation.
- **Stable Training:** More stable and less prone to issues like mode collapse compared to other generative models, such as GANs.
- **Computational Intensity:** The iterative denoising process requires significant computational resources.

Despite their computational demands, diffusion models' ability to generate diverse and high-quality images makes them a powerful tool in the field of image generation.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. create conda enviroment
 
```bash
 $ conda create --name myenv python=3.9 -y 
```
2. clone the project 
```bash
 $ git clone [link]

```


3. install requirements 
please install pytorch from 
```bash
#activate the conda env 
$ conda activate [env_name]
# cd into root of the project 
$ cd [root of the projcet ]
#then run 
$ pip install . 
# install the requirements 
$ pip install -r requirments.txt 

```
## How to train on your own data 

just changer the dataset dir path in the config.yaml file 


```bash
 $ python train_pipeline.py

```



## Results 


![alt text](static/result_2.jpeg)
![alt text](static/result_1.jpeg)
![alt text](static/result_8.jpg)
![alt text](static/result_6.jpg)
![alt text](static/result_10.jpg)


## pre train wights (unconditional)
https://drive.google.com/drive/folders/11Ej5R4HSvo1naScvmAoTaf80FgpEl36d?usp=sharing
download pre-train model weight from this like and replace  model_ckpt = "{weight_dir}"
```bash
 $ python predicton_pipe;ine.py

```

## Usage

Instructions on how to use the project can be found here.

## Contributing

Guidelines on how to contribute to the project can be found here.

## License

Information about the project's license can be found here.
