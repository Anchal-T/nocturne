FROM anyscale/ray:2.55.1-py311-cu128

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libsfml-dev \
    xvfb \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Pre-install Python dependencies into the image for faster node startup.
# Anyscale containerfile does not allow COPY from local directories, so
# packages are listed inline rather than sourced from requirements-anyscale.txt.
RUN pip install --no-cache-dir \
    torch \
    hydra-core \
    omegaconf \
    "gym==0.21.0" \
    wandb \
    tensorboard \
    boto3 \
    pyvirtualdisplay \
    dm-tree \
    tabulate
