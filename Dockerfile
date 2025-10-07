FROM nvcr.io/nvidia/pytorch:24.12-py3

ARG DEBIAN_FRONTEND=noninteractive
RUN echo "APT::Install-Recommends False;" > /etc/apt/apt.conf.d/60norecommends

RUN apt-get update && apt-get install -y --no-install-recommends \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxtst6 \
    libxi6 \
    libglib2.0-0 \
    #libgl1-mesa-glx \
    nfs-common \
    libopenblas-dev \
    python3-all-dev \
    nvidia-cuda-toolkit \
    ninja-build \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    libatlas-base-dev \
    gfortran \
    libblas-dev \
    liblapack-dev \
    screen \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Choose which GPU to build for
RUN export CUDA_VISIBLE_DEVICES=all

# Check CUDA toolkit installation
RUN nvcc --version && \
    python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"

WORKDIR /app
RUN curl -O https://repo.anaconda.com/archive/Anaconda3-2025.06-0-Linux-x86_64.sh

RUN sh Anaconda3-2025.06-0-Linux-x86_64.sh -b -p /opt/anaconda3

# Remove hte installation script
# RUN rm -rf Anaconda3-2025.06-0-Linux-x86_64.sh

# Add anaconda to the path
ENV PATH="/opt/anaconda3/condabin:$PATH"
ENV WORKDIR="/app"

# CLONE the project repository 
RUN git clone https://github.com/Gregz9/RobfarmerModel.git src

# Initialize conda and create environment
RUN conda init bash && \
    conda config --add channels conda-forge && \
    conda config --set channel_priority flexible && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda env create -f src/environment.yml

# Install setup in conda environment  
RUN cd src && conda run -n samclip pip install -e .

# CLONE OpenCV repositories
RUN git clone -b 4.x --depth 1 https://github.com/opencv/opencv.git
RUN git clone -b 4.x --depth 1 https://github.com/opencv/opencv_contrib.git

# INSTALL OpenCV with extra modules in the conda env
RUN cd opencv && \
    eval "$(/opt/anaconda3/bin/conda shell.bash hook)" && \
    conda activate samclip && \
    cmake -S . -B build -G Ninja -Wno-dev \
                        -D BUILD_LIST="core,cudev,features2d,imgproc,imgcodecs,highgui,stereo,videoio,viz,xfeatures2d" \
                        -D CMAKE_BUILD_TYPE=RELEASE \
                        -D CMAKE_INSTALL_PREFIX=/usr/local \
                        -D OPENCV_EXTRA_MODULES_PATH=/app/opencv_contrib/modules \
                        -D PYTHON_EXECUTABLE=$CONDA_PREFIX/bin/python \
                        -D PYTHON3_EXECUTABLE=$CONDA_PREFIX/bin/python \
                        -D PYTHON_DEFAULT_EXECUTABLE=$CONDA_PREFIX/bin/python \
                        -D PYTHON3_LIBRARY=$CONDA_PREFIX/lib/python3.12 \
                        -D PYTHON3_INCLUDE_DIR=$CONDA_PREFIX/include/python3.12 \
                        -D PYTHON3_PACKAGES_PATH=$CONDA_PREFIX/lib/python3.12/site-packages \
                        -D BUILD_OPENCV_PYTHON2=OFF \
                        -D BUILD_OPENCV_PYTHON3=ON \
                        -D INSTALL_PYTHON_EXAMPLES=ON \
                        -D INSTALL_C_EXAMPLES=ON \
                        -D OPENCV_ENABLE_NONFREE=ON \
                        -D BUILD_EXAMPLES=ON \
                        -D WITH_CUDA=ON \
                        -D CUDA_ARCH_BIN=8.6 \
                        -D CUDA_ARCH_PTX=8.6 \
                        -D WITH_CUBLAS=ON \
                        -D WITH_CUDNN=ON \
                        -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
                        -D OPENCV_DNN_CUDA=ON \
                        -D CUDA_FAST_MATH=ON \
                        -D BUILD_opencv_cudacodec=ON \
                        -D BUILD_opencv_cudaimgproc=ON \
                        -D BUILD_opencv_cudawarping=ON \
                        -D BUILD_opencv_cudev=ON \
                        -D BUILD_opencv_cudafilters=ON \
                        -D BUILD_opencv_cudaarithm=ON \
                        -D BUILD_opencv_cudaobjdetect=ON \
                        -D BUILD_opencv_cudastereo=ON \
                        -D BUILD_opencv_cudafeatures2d=ON \
                        -D BUILD_opencv_cudaoptflow=ON \
                        -D BUILD_opencv_cudabgsegm=ON \
                        -D CUDA_GENERATION=Auto \
                        -D WITH_GDAL=ON \
                        -D WITH_GPHOTO2=ON \
                        -D WITH_GSTREAMER=OFF \
                        -D WITH_PROTOBUF=ON \
                        -D BUILD_PROTOBUF=ON \
                        -D WITH_QT=OFF \
                        -D WITH_TBB=ON \
                        -D BUILD_TBB=ON && \
    cd build && ninja -j 8 && ninja install

# CREATE dataset dirs 
RUN mkdir $WORKDIR/data
RUN mkdir $WORKDIR/data/datasets

# COPY dataset from OneDrive to its location
COPY Robofarmer-II.tar.gz $WORKDIR/data/datasets

# UNPACK The dataset
RUN cd data/datasets && tar -xzvf Robofarmer-II.tar.gz

# Install additional OpenGL libraries for cv2
RUN apt-get update && apt-get install -y libgl1-mesa-dev libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# RUN scripts to create dataset structure and annotation file
# RUN cd /app/src && conda run -n samclip python create_robofarmer2.py

# RUN script to create annotation.json file
# RUN cd /app/src && conda run -n samclip python data_loading.py
