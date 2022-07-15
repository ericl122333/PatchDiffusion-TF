import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

#visualization utils
def show_images(images, scale=5, savepath=None, dims = None):
    if isinstance(images, tf.Tensor):
        if len(images.shape) == 4:
            images = tf.split(images, images.shape[0], axis=0)
            for i in range(len(images)):
                images[i] = tf.squeeze(images[i])
    
    if not isinstance(images[0], np.ndarray):
        for i in range(len(images)):
            images[i] = float_to_image(images[i])
    
    if dims is None:
        m = len(images)//10 + 1
        n = 10
    else:
        m, n = dims

    plt.figure(figsize=(scale*n, scale*m))

    for i in range(len(images)):
        plt.subplot(m, n, i+1)
        plt.imshow(images[i])
        plt.axis('off')

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()

    
#general utils
def inv_snr_function(logsnr):
    t = tf.sqrt((tf.math.softplus(-logsnr) - 1e-4)/10)
    t = tf.cast(t * 1000, tf.int32)
    return t
    
class WarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr, warmup_steps):
        super().__init__()
        self.lr = lr
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        warmup_lr = step * self.lr / float(self.warmup_steps)
        return tf.math.minimum(self.lr, warmup_lr)
        

def make_mapfn(is_labeled, res):
    def read_tfrecord_labeled(example):
        features = {
            "x": tf.io.FixedLenFeature([], tf.string),
            "y": tf.io.FixedLenFeature([], tf.int64),
        }
        example = tf.io.parse_single_example(example, features)
        image, label = example['x'], example["y"]
        image = tf.io.decode_png(image, dtype=tf.uint8)
        label = tf.cast(label, tf.int32)

        image = tf.reshape(image, [res, res, 3]) 
        image = tf.cast(image, tf.float32) / 127.5 - 1.0
        return image, label

    def read_tfrecord_unlabeled(example):
        features = {
            "x": tf.io.FixedLenFeature([], tf.string)
        }
        example = tf.io.parse_single_example(example, features)
        image = example['x']
        image = tf.io.decode_raw(image, tf.uint8)
        image = tf.reshape(image, [res, res, 3]) 
        image = tf.cast(image, tf.float32) / 127.5 - 1.0
        return image
    
    if is_labeled:
        return read_tfrecord_labeled
    else:
        return read_tfrecord_unlabeled


#device utils
def set_policy(policy_name, device_type):
    from tensorflow.keras import mixed_precision
    if device_type == 'TPU' and policy_name == 'mixed_bfloat16':
        policy = mixed_precision.Policy('mixed_bfloat16')
        mixed_precision.set_global_policy(policy)
    elif policy_name == 'mixed_float16':
        raise NotImplementedError("mixed_float16 NOT YET SUPPORTED bc of loss scale optimizer not implemented.")
        
def Colab_TPU_Strategy():
    import tensorflow_gcs_config
    from google.colab import auth

    os.environ['USE_AUTH_EPHEM'] = '0'
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    TPU_ADDRESS = tpu.get_master()
    print('Running on TPU:', TPU_ADDRESS)

    auth.authenticate_user()
    tf.config.experimental_connect_to_host(TPU_ADDRESS)
    tensorflow_gcs_config.configure_gcs_from_colab_auth()

    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    print("All devices: ", tf.config.list_logical_devices('TPU'))
    strategy = tf.distribute.TPUStrategy(tpu)
    return strategy

def get_strategy(device=None):
    if isinstance(device, str):
        if "CPU" in device.upper():
            return tf.distribute.get_strategy(), "CPU"
        elif "colab" in device.lower() and "tpu" in device.lower():
            return Colab_TPU_Strategy(), "TPU"

    #get this to work on single GPU
    if tf.config.list_physical_devices('GPU'):
        return tf.distribute.MirroredStrategy(devices=device), "GPU"
    else:
        try:
            return Colab_TPU_strategy(), "TPU"
        except:
            return tf.distribute.get_strategy(), "CPU"


