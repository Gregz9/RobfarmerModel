# RobfarmerModel Docker Setup

This guide explains how to build and run the RobfarmerModel using Docker.

## Prerequisites

- Docker installed on your system
- NVIDIA Docker runtime (for GPU support)
- Copy the Robofarmer-II directory from 

## Building the Docker Image

Build the Docker image with the following command:

```bash
podman build -t robofarmer:latest .
```

**Note:** The build process will:
- Use NVIDIA PyTorch base image with CUDA support
- Install system dependencies and build tools
- Set up Anaconda environment
- Clone the repository and install dependencies
- Build OpenCV with CUDA support
- Extract the dataset from `Robofarmer-II.tar.gz`

## Running the Container

### Basic Run Command

```bash
podman run -it --rm robofarmer:latest
```

### With GPU Support

```bash
podman run -it --rm --gpus all robofarmer:latest
```

### With Volume Mounting (for data persistence)

```bash
podman run -it --rm --device nvidia.com/gpu=0 \
  -v $(pwd)/Robofarmer-II:/app/data/datasets/Robofarmer-II  \
  robofarmer:latest
```

### Interactive Shell Access (Recommended for running all scripts)

```bash
podman run -dit --shm-size=24g --device nvidia.com/gpu=0 --name robofarmer -v $HOME/Robofarmer-II:/app/data/datasets/Robofarmer-II bash
```

### Enter the container 
```bash
podman attach robofarmer
```

## Working Directory

The container's working directory is `/app`, which contains:
- `/app/src` - The cloned RobfarmerModel repository
- `/app/data` - Dataset directory
- `/app/opencv` - OpenCV source code

## Conda Environment

The container uses a conda environment called `samclip`. To activate it within the container:

```bash
conda activate robrobofarmer
```

## Common Commands

Once inside the container, you can run the model scripts:

```bash
# Activate the environment
conda activate robofarmer

# Navigate to source directory
cd src/interaction_hotspots
```

### How to train and evaluate models
In the interaction_hotspots directory, there are README files explaining how to train and evaluate models and to visualize heatmaps using trained models.

