TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
nvcc -std=c++11 -c -o cuda_op_kernel.cu.o cuda_op_kernel.cu ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 -shared cuda_op_kernel.cc cuda_op_kernel.cu.o -o cuda_op_kernel.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
