## OPENCV BUILD FROM SOURCE FOR PYTHON

```
export CPLUS_INCLUDE_PATH=~/anaconda3/envs/<env_name>/lib/python3.12

# Run from the top dir 
cmake -S . -B build2 -G Ninja \
      -D BUILD_LIST="core,cudev,features2d,imgproc,imgcodecs,highgui,stereo,videoio,viz,xfeatures2d" \
      -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=~/Desktop/MasterThesis/opencv_contrib/modules \
      -D PYTHON_EXECUTABLE=/home/gregz9/anaconda3/envs/<env_name>/bin/python \
      -D PYTHON3_EXECUTABLE=/home/gregz9/anaconda3/envs/<env_name>/bin/python \
      -D PYTHON_DEFAULT_EXECUTABLE=/home/gregz9/anaconda3/envs/<env_name>/bin/python \
      -D PYTHON3_LIBRARY=/home/gregz9/anaconda3/envs/<env_name>/lib/python3.12 \
      -D PYTHON3_INCLUDE_DIR=/home/gregz9/anaconda3/envs/<env_name>/include/python3.12 \
      -D PYTHON3_PACKAGES_PATH=/home/gregz9/anaconda3/envs/<env_name>/lib/python3.12/site-packages \
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
      <!-- -D BUILD_TBB=OFF -->
```

cmake -S . -B build -G Ninja \
      -D BUILD_LIST="core,cudev,features2d,imgproc,cudaimageproc, imgcodecs,highgui,stereo,videoio,viz,xfeatures2d" \
      -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=~/Desktop/MasterThesis/opencv_contrib/modules \
      -D PYTHON3_LIBRARY=~/anaconda3/envs/yolo11/lib/python3.12 \
      -D PYTHON3_INCLUDE_DIR=~/anaconda3/envs/yolo11/include/python3.12 \
      -D PYTHON3_EXECUTABLE=~/anaconda3/envs/yolo11/bin/python \
      -D PYTHON3_PACKAGES_PATH=~/anaconda3/envs/yolo11/lib/python3.12/site-packages \
      -D BUILD_DOCS=OFF \
      -D BUILD_EXAMPLES=OFF \
      -D BUILD_JAVA=OFF \
      -D BUILD_opencv_python2=OFF \
      -D BUILD_opencv_python3=ON \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_TESTS=OFF \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D WITH_CUDA=ON \
      -D WITH_CUBLAS=ON \
      -D WITH_CUDNN=ON \
      -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
      -D CUDA_ARCH_BIN=8.6 \
      -D CUDA_ARCH_PTX=8.6 \
      -D OPENCV_DNN_CUDA=ON \
      -D CUDA_SEPARABLE_COMPILATION=OFF \
      -D CUDA_NVCC_FLAGS="-Xcompiler -fPIC -O3" \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      -D BUILD_opencv_world=OFF \
      -D CMAKE_POSITION_INDEPENDENT_CODE=ON \
      -D CMAKE_CXX_FLAGS="-fPIC" \
      -D CUDA_NVCC_FLAGS="--default-stream per-thread --expt-relaxed-constexpr -O3" \
      -D BUILD_SHARED_LIBS=ON

cmake -S . -B build -G Ninja \
    -D BUILD_LIST="core,cudev,features2d,imgproc,imgcodecs,highgui,stereo,videoio,viz,xfeatures2d" \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
    -D OPENCV_EXTRA_MODULES_PATH=~/Desktop/MasterThesis/opencv_contrib/modules \
    -D PYTHON3_LIBRARY=$CONDA_PREFIX/lib/libpython3.12.so \
    -D PYTHON3_INCLUDE_DIR=$CONDA_PREFIX/include/python3.12 \
    -D PYTHON3_EXECUTABLE=$CONDA_PREFIX/bin/python \
    -D PYTHON3_NUMPY_INCLUDE_DIRS=$CONDA_PREFIX/lib/python3.12/site-packages/numpy/core/include \
    -D WITH_CUDA=ON \
    -D WITH_CUBLAS=ON \
    -D WITH_CUDNN=ON \
    -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    -D CUDA_ARCH_BIN=8.6 \
    -D CUDA_ARCH_PTX=8.6 \
    -D OPENCV_DNN_CUDA=ON \
    -D CUDA_SEPARABLE_COMPILATION=OFF \
    -D BUILD_opencv_cudacodec=ON \
    -D BUILD_opencv_cudaimgproc=ON \
    -D BUILD_opencv_cudafeatures2d=ON \
    -D BUILD_opencv_cudafilters=ON \
    -D BUILD_opencv_cudaarithm=ON \
    -D CUDNN_INCLUDE_DIR=/usr/include \
    -D CUDNN_LIBRARY=/usr/lib/x86_64-linux-gnu/libcudnn.so \
    -D BUILD_SHARED_LIBS=ON
