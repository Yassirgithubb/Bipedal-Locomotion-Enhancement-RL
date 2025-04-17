FROM nvidia/cuda:11.4.2-runtime-ubuntu20.04 AS builder

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

LABEL maintainer="Filip Bjelonic" mail="filipb@leggedrobotics.com"

# To avoid tzdata asking for geographic location...
ENV LANG C.UTF-8
ARG DEBIAN_FRONTEND=noninteractive
ENV DEBIAN_frontend noninteractive

# Create folders for cloning
ENV WORKSPACE=/isaac_ws
RUN mkdir -p $WORKSPACE && mkdir -p $WORKSPACE/isaacgym && mkdir -p $WORKSPACE/rsl_rl && mkdir -p $WORKSPACE/legged_gym

ENV CXX=/usr/bin/g++
ENV CC=/usr/bin/gcc

# Install dependencies
RUN apt-get update --fix-missing && apt-get install -y apt-utils git wget vim openscad python3-dev python3-distutils python3-pip libpython3-dev
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install matplotlib tensorboard trimesh warp-lang trimesh pytest GitPython numpy==1.23.0 imageio ninja pillow pyyaml scipy

# Install IsaacGym
ADD isaacgym $WORKSPACE/isaacgym
RUN cd $WORKSPACE/isaacgym/python && pip install -e .

# Install rsl_rl
ADD rsl_rl $WORKSPACE/rsl_rl
RUN cd $WORKSPACE/rsl_rl && pip install -e .

# Install legged_gym
ADD legged_gym $WORKSPACE/legged_gym
RUN cd $WORKSPACE/legged_gym && pip install -e .

# delete folders for mounting your changes on the cluster 
# (uncomment the next line if you have changes in rsl_rl that needs to be mounted on the cluster)
#RUN rm -r $WORKSPACE/rsl_rl/* && rm -rf rsl_rl/{,.[!.],..?}*  # deletes normal files && hidden files
RUN rm -r $WORKSPACE/legged_gym/* && rm -rf legged_gym/{,.[!.],..?}*  # deletes normal files && hidden files

RUN mkdir -p $WORKSPACE/legged_gym/logs
