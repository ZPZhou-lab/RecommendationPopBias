import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # close tensorflow logging
import tensorflow as tf

# set GPU memory growth
def set_gpu_memory_limitation(memory: int=10):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory*1024)]
            )
        except RuntimeError as e:
            print(e)
    else:
        print('No GPU available')