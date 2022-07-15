import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model, Sequential
import numpy as np
from tensorflow_addons.layers import GroupNormalization
from nn import get_timestep_encoding, get_2d_positional_encoding, Downsample, Upsample, AttentionBlock, TimestepEmbedSequential, ResBlock



class UNet(Model):
    '''
    Patched U-Net architecture
    args: image_resolution -- integer, resolution before patching
          c -- integer, number of base channels, will be multiplied by c_mult
          num_res_blocks -- integer or list, number of blocks per resolution on one half (up/down) of the U-Net
          droprate -- float, dropout probability, must be greater than 0.01 to be used
          attn_resos -- list (integer) of resolutions to use attention at
          c_mult -- list (integer) of channel multipliers, to be multiplied by c (base_channels)
          num_classes -- integer, for class conditional, defaults to zero
          policy_name -- string, 'mixed_bfloat16' for bfloat16 training, float16 not supported
          patchsize -- integer, side length of the square patch
          se -- boolean, whether to inject class conditioning via a sigmoid gate
    '''

    def __init__(self, image_resolution, c, num_res_blocks, droprate=0., attn_resos=[16,8], c_mult=[1,2,3,4], num_classes=0, policy_name='float32', patchsize=1, se=False):
        super().__init__()
        self.resolution = resolution = image_resolution // patchsize
        self.c = c
        self.c_out = 3 * patchsize**2
        self.num_res_blocks = num_res_blocks
        
        if isinstance(num_res_blocks, int):
            num_res_blocks = [num_res_blocks] * len(c_mult)
        elif isinstance(num_res_blocks, list) or isinstance(num_res_blocks, tuple):
            num_res_blocks = list(num_res_blocks)
            assert len(num_res_blocks) == len(c_mult), "when passing in a list of depths at each resolution, make sure the list length is equal to the number of resolutions."
        else:
            raise ValueError("n_blocks should be an integer specifying depth (how many blocks) or a list of integers specifying depth at each resolution.")
            
        self.attn_resos = attn_resos
        self.droprate = droprate
        self.c_mult = c_mult
        self.network_dtype = tf.bfloat16 if policy_name=='mixed_bfloat16' else tf.float32 #we only support mixed precision training on TPU
        self.se = (se and num_classes)
        self.patch_size = patchsize
        
        self.num_classes = num_classes
        self.num_classesp1 = num_classes + 1 #+1 is for classifier-free guidance, where the LAST embedding is the unconditional label
        self.inp_shape = [image_resolution, image_resolution, 3]
        self.positional_emb = get_2d_positional_encoding(resolution, c, dtype=self.network_dtype)
        
        if num_classes > 1: 
            if self.se: self.label_embs = Embedding(num_classes+1, c)
            else: self.label_embs = Embedding(num_classes+1, c*c_mult[-1]) #here, we add c_emb and t_emb, so they must be same channel size
        self.temb_network = Sequential([Dense(c*4, activation='swish'), Dense(c*c_mult[-1], activation='swish')])
        
        self.in_proj = Conv2D(c, 3, padding='same', dtype=tf.float32) 
        current_reso = resolution
        self.input_layers = []
        
        for level, mult in enumerate(c_mult):
            C = int(c*mult)
            
            for _ in range(num_res_blocks[level]):
                layers = [ResBlock(C, droprate, se=self.se)]
                if current_reso in self.attn_resos: 
                    layers.append(AttentionBlock(C, current_reso))
                self.input_layers.append(TimestepEmbedSequential(layers))
            
            if level != len(c_mult)-1:
                num_out_channels = int(c*c_mult[level+1])
                self.input_layers.append(Downsample(num_out_channels, use_conv=True))
                current_reso /= 2
            
        C = int(c*c_mult[-1])
        self.middle_layers = [
            ResBlock(C, droprate, se=self.se),
            AttentionBlock(C, current_reso),
            ResBlock(C, droprate, se=self.se)
        ]
        
        self.output_layers = []
        for level, mult in list(enumerate(c_mult))[::-1]:
            C = int(c*mult)
            for i in range(num_res_blocks[level]+1):
                layers = [ResBlock(C, droprate, use_nin_shortcut=True, se=self.se)]
                if current_reso in self.attn_resos:
                    layers.append(AttentionBlock(C, current_reso))
                  
                if level and i==num_res_blocks[level]: 
                    num_out_channels = int(c*c_mult[level-1])
                    layers.append(Upsample(num_out_channels, use_conv=True))
                    current_reso *= 2
                self.output_layers.append(TimestepEmbedSequential(layers))
                
        self.out_proj = Sequential([
            GroupNormalization(groups=32, dtype=tf.float32),
            Activation('swish', dtype=tf.float32),
            Conv2D(self.c_out, 3, padding='same', kernel_initializer='zeros', dtype=tf.float32)
        ])
        
        self.variance_output_t_emb = Dense(c*2)
        self.variance_output_model = Sequential([
                Conv2D(c, 3, padding='same', activation='swish'),
                Conv2D(self.c_out, 3, padding='same', activation='sigmoid', dtype=tf.float32), #0 for beta, 1 for b_tilde
            ])
    
    def to_patches(self, x):
        p = self.patch_size
        return tf.image.extract_patches(x, [1,p,p,1], [1,p,p,1], [1,1,1,1], "SAME")
    
    def from_patches(self, x):
        B, H, W, C = x.shape   #horizontal assemble patches
        x = tf.transpose(x, [0,2,1,3])
        x = tf.reshape(x, [B, W, H*self.patch_size, C//self.patch_size])
        x = tf.transpose(x, [0,2,1,3])

        B, H, W, C = x.shape  #horizontal assemble patches
        x = tf.reshape(x, [B, H, W*self.patch_size, C//self.patch_size])
        return x
    
    def get_classifier_free_label(self, label):
        mask = tf.cast(tf.math.greater(tf.random.uniform(label.shape), 0.9), tf.int32)
        empty_label = tf.fill(label.shape, self.num_classes)
        return (1-mask) * label + mask * empty_label

    def build_network(self, verbose=False):
        z, t, c = tf.random.normal([1]+self.inp_shape), tf.random.uniform([1], maxval=1000, dtype=tf.int32), tf.random.uniform([1], maxval=self.num_classesp1, dtype=tf.int32)
        _ = self(z, t, c)
        if verbose:
            self.summary()
            
    def call_var_network(self, x, h, t_emb):
        #we concatenate the input, hidden state, and timestep embedding
        #then call a stop_gradient on this, to ensure that L_sigma does not affect the rest of the model
        concat = tf.concat([x, h], axis=-1)
        concat = tf.stop_gradient(concat) + self.variance_output_t_emb(tf.nn.swish(t_emb))
        return self.variance_output_model(concat)
        
    def call(self, x, timesteps, y=None):
        #args: x -- of shape [B, image_resolution, image_resolution, C], representing a sample from q(zt|x)
        #      timesteps -- of shape [B,]
        #      y -- of shape [B, ]
        
        t_emb = get_timestep_encoding(timesteps, self.c)
        t_emb = self.temb_network(t_emb)
        t_emb = tf.cast(t_emb, self.network_dtype)[:, None, None, :]
        
        if self.num_classes > 1:
            c_emb = self.label_embs(y)
            c_emb = tf.cast(c_emb, self.network_dtype)[:, None, None, :]
            if self.se: conditioning = (c_emb, t_emb)
            else: conditioning = c_emb + t_emb
        else: conditioning = t_emb
        
        x = self.to_patches(x)
        
        x = self.in_proj(x)
        x = tf.cast(x, self.network_dtype)
        x += self.positional_emb
        h = tf.identity(x)
        activations = [h]
        
        for layer in self.input_layers:
            h = layer(h, conditioning)
            activations.append(h)
        for layer in self.middle_layers:
            h = layer(h, conditioning)
        for layer in self.output_layers:
            h = tf.concat([h, activations.pop()], axis=-1)
            h = layer(h, conditioning)
            
        mean_output = self.out_proj(tf.cast(h, tf.float32))
        mean_output = self.from_patches(mean_output)
        var_output = self.call_var_network(x, h, t_emb)
        var_output = self.from_patches(var_output)
        return mean_output, var_output
