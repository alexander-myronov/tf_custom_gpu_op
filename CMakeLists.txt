cmake_minimum_required(VERSION 3.5)

#found from running python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())'
include_directories(/home/ardigen/anaconda3/envs/tf_gpu/lib/python3.6/site-packages/tensorflow/include)
link_directories(/home/ardigen/anaconda3/envs/tf_gpu/lib/python3.6/site-packages/tensorflow)
find_package(CUDA)

#set flags based on tutorial
set (CMAKE_CXX_FLAGS "--std=c++11 -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=0")

#pass flags to c++ compiler
SET(CUDA_PROPAGATE_HOST_FLAGS ON)

#create library
cuda_add_library(
    cuda_op SHARED
    cuda_op_kernel.cu
    cuda_op_kernel.cc)

#copy test file to build folder
#configure_file(src/test.py test.py COPYONLY)