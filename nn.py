import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Activation, Dense, Layer, Dropout
from tensorflow_addons.layers import GroupNormalization
import numpy as np

def get_timestep_encoding(timesteps, embedding_dim: int):
    #
    
    assert len(timesteps.shape) == 1 
    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
    
    emb = tf.cast(timesteps, dtype=tf.float32)[:, None] * emb[None, :]
    emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = tf.pad(emb, [[0, 0], [0, 1]])

    assert emb.shape == [timesteps.shape[0], embedding_dim]
    return emb

def get_2d_positional_encoding(resolution, embedding_dim, dtype=tf.float32):
    #encodes spatial information for a 2D image by creating Transformer sinusoidal embeddings 
    #for the height and width, and concatting them along the channel dimension
    
    #args: resolution -- length/width of the (patched) image; only square images are supported
    #      embedding_dim -- the number of channels, should be equal to the number of channels in the first layer
    #      dtype -- the output dtype of the positional encoding, used for mixed precision training
    
    omega = 64 / resolution   #higher resolutions need longer wavelengths
    half_dim = embedding_dim // 2
    arange = tf.range(resolution, dtype=tf.float32)

    emb = np.log(10000) / (half_dim - 1)
    emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -emb) 
    emb = arange[:, None] * emb[None, :] 
    emb = tf.sin(emb * omega)
    
    emb_x = tf.repeat(emb[None, ...], resolution, axis=0)
    emb_y = tf.repeat(emb[:, None, :], resolution, axis=1)
    emb = tf.concat([emb_x, emb_y], axis=-1)
    return tf.cast(emb[None, ...], dtype=dtype)
    
class Downsample(Layer):
    #biggan-style downsampling block
    #this version is slightly different from the one in ADM because tensorflow's padding
    #is different from PyTorch
    
    #args: c -- integer, the number of output channels 
    #      use_conv -- whether to use a strided convolution instead of average pooling
    
    def __init__(self, c, use_conv):
        super().__init__()
        if use_conv: self.down = Conv2D(c, 3, padding='same', strides=2)
        else: self.down = AveragePooling2D()
    
    def call(self, x, timesteps):
        return self.down(x)
      
class Upsample(Layer):
    #biggan-style upsampling block, as used in ADM
    #args: c -- integer, the number of output channels 
    #      use_conv -- whether to use a strided convolution instead of average pooling
    
    def __init__(self, c, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if self.use_conv: self.up = Conv2D(c, 3, padding='same')

    def call(self, x, timesteps):
        B, H, W, C = x.shape
        x = tf.image.resize(x, size=[H * 2, W * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        if self.use_conv:
            x = self.up(x)
        return x


class AttentionBlock(Layer):
    #self attention block with fused q, k, and v matmuls
    #unlike the ADM attention block, this one uses GroupNorm 
    
    #args: c -- integer, number of input/output channels
    #      resolution -- resolution at this layer; in an earlier version this arg had functionality
    #      channels_per_head -- integer
    #      zero_init -- whether to initialize the output weight matrix to zero
    

    def __init__(self, c, resolution, channels_per_head=64, zero_init=True):
        super().__init__()
        self.c = c
        self.resolution = resolution
        self.num_heads = c//channels_per_head
        self.channels_per_head = channels_per_head
        self.factor = 1/np.sqrt(np.sqrt(self.channels_per_head)).astype('float32')
        
        self.norm = GroupNormalization(groups=32)
        self.qkv_proj = Dense(c*3)
        if zero_init: self.out_proj = Dense(c, kernel_initializer='zeros')
        else: self.out_proj = Dense(c)
          
    def split_heads(self, x):
        B, L, C = x.shape
        x = tf.reshape(x, [B, L, self.num_heads, self.channels_per_head])
        return tf.transpose(x, perm=[0,2,1,3])
          
    def attention(self, qkv):
        q, k, v = tf.split(qkv, 3, axis=-1)
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        
        qk = tf.matmul(q*self.factor, k*self.factor, transpose_b=True) #batch_size, d_model, seq_len_q, seq_len_k
        attention_weights = tf.nn.softmax(qk, axis=-1)  # (..., seq_len_q, seq_len_k)
        output = tf.matmul(attention_weights, v)     
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        return tf.reshape(output, [q.shape[0], -1, self.c])
        
    def call(self, x, timesteps):
        B, H, W, C = x.shape
        x = tf.reshape(x, [B, H*W, C])
        qkv = self.qkv_proj(self.norm(x))
        
        h = self.attention(qkv)
        h = self.out_proj(h)
        x = tf.reshape(x + h, [B, H, W, C])
        return x
        
class resblock_se(Layer):
    #class-conditional residual block that uses a sigmoid gate for class-conditioning

    #args: c -- integer, number of input/output channels
    #      droprate -- float, probability that a channel will be zero
    #      use_nin_shortcut -- whether to use skip connection, 
    #                          should be true if input channel =/= output channel
    
    def __init__(self, c, droprate, use_nin_shortcut=False):
        super().__init__()
        self.c = c
        self.droprate = droprate
        self.use_nin_shortcut = use_nin_shortcut
        
        self.in_layers = Sequential([
            GroupNormalization(groups=32),
            Activation('swish'),
            Conv2D(c, 3, padding='same')
        ])
        
        self.classemb_proj = Dense(c)
        self.temb_proj = Dense(c * 2, kernel_initializer='zeros')
        self.out_norm = GroupNormalization(groups=32, center=False, scale=False)
        
        out_layers = [Activation('swish'),
            Conv2D(c, 3, padding='same', kernel_initializer='zeros')]
        if droprate>0.01: out_layers.insert(1, Dropout(droprate))
        self.out_layers = Sequential(out_layers)
        if use_nin_shortcut: self.skip_connection = Dense(c) 
        
    def call(self, x, conditioning):
        c_emb, t_emb = conditioning
        scale, shift = tf.split(self.temb_proj(t_emb), 2, axis=-1)  
        
        h = self.in_layers(x)       
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_layers(h) * tf.nn.sigmoid(self.classemb_proj(c_emb) + 2.)
        
        if self.use_nin_shortcut: return self.skip_connection(x) + h
        else: return x + h

class resblock(Layer):
    #ADM-style residual block

    #args: c -- integer, number of input/output channels
    #      droprate -- float, probability that a channel will be zero
    #      use_nin_shortcut -- whether to use skip connection, 
    #                          should be true if input channel =/= output channel
    
    def __init__(self, c, droprate, use_nin_shortcut=False):
        super().__init__()
        self.c = c
        self.droprate = droprate
        self.use_nin_shortcut = use_nin_shortcut
        
        self.in_layers = Sequential([
            GroupNormalization(groups=32),
            Activation('swish'),
            Conv2D(c, 3, padding='same')
        ])
        
        self.emb_layers = Dense(c * 2, kernel_initializer='zeros')
        self.out_norm = GroupNormalization(groups=32, center=False, scale=False)
        
        out_layers = [Activation('swish'),
            Conv2D(c, 3, padding='same', kernel_initializer='zeros')]
        if droprate>0.01: out_layers.insert(1, Dropout(droprate))
        self.out_layers = Sequential(out_layers)
        if use_nin_shortcut: self.skip_connection = Dense(c) 
        
    def call(self, x, emb, use_nin_shortcut=False):
        h = self.in_layers(x)            
        emb = self.emb_layers(emb)
        scale, shift = tf.split(emb, 2, axis=-1)
            
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        if self.use_nin_shortcut: return self.skip_connection(x) + h
        else: return x + h
        
        
def ResBlock(c, droprate, use_nin_shortcut=False, se=False):
    #returns either a normal residual block (if se=False), 
    #or a gated residual block (if se=True)

    #args: c -- integer, number of input/output channels
    #      droprate -- float, probability that a channel will be zero
    #      use_nin_shortcut -- whether to use skip connection, 
    #                          should be true if input channel =/= output channel
    #      se -- whether to use a class-conditional sigmoid gate
     
    if se: return resblock_se(c, droprate, use_nin_shortcut=use_nin_shortcut)
    else: return resblock(c, droprate, use_nin_shortcut=use_nin_shortcut)
     

class TimestepEmbedSequential(Layer):
    #args: layers -- list of Layer instances that must take in x, timesteps as input

    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def call(self, x, timesteps):
        for layer in self.layers:
            x = layer(x, timesteps)
        return x
        

