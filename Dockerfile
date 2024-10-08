FROM python:3.10.13

# Set the working directory
WORKDIR /diffusion

RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Copy the requirements file
COPY requirements.txt .
RUN pip install -r requirements_dev.txt

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install gunicorn

COPY . .

CMD [ "python", "app.py" ]