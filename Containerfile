FROM anyscale/ray:2.47.0-py311-gpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libsfml-dev \
    xvfb \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-anyscale.txt /tmp/requirements-anyscale.txt
RUN pip install --no-cache-dir -r /tmp/requirements-anyscale.txt

# Build and install the nocturne C++ extension into site-packages.
# At runtime Ray prepends working_dir to sys.path, so modified Python files
# override the installed package while the compiled .so stays in site-packages.
COPY . /nocturne-build/
RUN cd /nocturne-build && pip install . --no-build-isolation && rm -rf /nocturne-build
