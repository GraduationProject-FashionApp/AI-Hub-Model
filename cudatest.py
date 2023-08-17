import tensorflow as tf

print("cuDNN version:", tf.sysconfig.get_build_info()['cuda_version'])
print('loading')