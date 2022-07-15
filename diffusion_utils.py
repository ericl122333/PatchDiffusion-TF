import tensorflow as tf
import numpy as np

def hybrid_lossfn(x0, pred_x0, var_output, timesteps, diffusion):
    alphas = diffusion.gather(timesteps, 'alphas')
    betas = diffusion.gather(timesteps, 'betas')
    b_tilde = diffusion.gather(timesteps, 'b_tilde')
    root_snr = tf.sqrt(tf.squeeze(alphas/(1-alphas)))
    vlb_mask = tf.math.minimum(tf.cast(timesteps, tf.float32), 1.0) #do not learn variance for t=0

    squared_distances = tf.reduce_sum(tf.square(pred_x0 - x0), axis=[-1,-2,-3])
    L_simple = tf.reduce_mean(squared_distances * root_snr) 
    
    pred_x0 = tf.stop_gradient(pred_x0)
    pred_var = diffusion.get_pred_var(var_output, b_tilde, betas)
    gamma_vlb = alphas * betas**2 / (pred_var * (1-betas) * (1-alphas)**2)
    Constant_term = tf.math.log(pred_var / b_tilde) + (b_tilde / pred_var) - 1
    
    KL = 0.5 * (gamma_vlb * tf.square(pred_x0 - x0) + Constant_term)
    KL = tf.reduce_sum(KL, axis=[-1,-2,-3]) * vlb_mask
    L_vlb = tf.reduce_sum(KL)
    return L_simple, L_vlb

def l2_lossfn(x0, pred_x0, timesteps, diffusion):
    alphas = diffusion.gather(timesteps, 'alphas')
    betas = diffusion.gather(timesteps, 'betas')
    b_tilde = diffusion.gather(timesteps, 'b_tilde')
    root_snr = tf.sqrt(tf.squeeze(alphas/(1-alphas)))

    squared_distances = tf.reduce_sum(tf.square(pred_x0 - x0), axis=[-1,-2,-3])
    L_simple = tf.reduce_mean(squared_distances * root_snr) 
    return L_simple

class Diffusion():
    def __init__(self, rng, beta_max=0.02, schedule='linear'):
        self.rng = rng

        self.beta_set = tf.linspace(1e-4, beta_max, 1000)
        self.alpha_set = tf.math.cumprod(1-self.beta_set)

        #alpha_prev[0] should actually be 1.0 instead of 1-beta[0], but this will cause NaN errors when taking log(1-alpha_prev[0])
        self.alpha_prev = tf.concat([self.alpha_set[0:1], self.alpha_set[:-1]], axis=0)  
        self.b_tilde = self.beta_set * (1-self.alpha_prev) / (1-self.alpha_set)
        self.T = len(self.beta_set)
        self.snrs = self.alpha_set / (1-self.alpha_set)

    def get_timestep_from_snr(self, snr):
        log_snrs = np.log(self.snrs)
        t = np.argmin(np.abs(log_snrs - np.log(snr)))
        return t

    def gather(self, timesteps, gather_from='alphas'):
        if gather_from=='betas': arr = self.beta_set
        elif gather_from=='b_tilde': arr = self.b_tilde
        elif gather_from=='alpha_prev': arr = self.alpha_prev
        else: arr = self.alpha_set
        return tf.gather_nd(arr, timesteps[:, None])[:, None, None, None] 
    
    def draw_from_posterior(self, x0, t):
        alphas = self.gather(t, 'alphas')
        eps = self.rng.normal(x0.shape)
        xt = tf.sqrt(alphas)*x0 + tf.sqrt(1-alphas) * eps
        return xt, eps

    def get_eps_from_x0(self, xt, x0, timesteps):
        alphas = self.gather(timesteps, 'alphas')
        return (xt - tf.sqrt(alphas)*x0) / tf.sqrt(1-alphas)
    
    def get_pred_x0(self, xt, model_out, timesteps, output_type):
        alphas = self.gather(timesteps, 'alphas')
        if output_type == 'x0': return model_out
        elif output_type == 'v': return tf.sqrt(alphas) * xt - tf.sqrt(1-alphas) * model_out
        elif output_type == 'eps': (xt - tf.sqrt(1-alphas) * model_out) / tf.sqrt(alphas)
        else: raise Exception("output type must be one of 'x0', 'v', 'eps'")
    
    def to_timestep(self, bs, t):
        return tf.cast(tf.ones([bs])*t, tf.int32)

    def sample_timesteps(self, bs, minval, maxval):
        return self.rng.randint([bs], minval=minval, maxval=maxval)

    def get_pred_var(self, var_output, b_tilde, beta):
        log_btilde = tf.math.log(b_tilde)
        log_beta = tf.math.log(beta)
        pred_var = tf.exp(var_output*log_btilde + (1-var_output)*log_beta)
        return pred_var

class RNG:
    def __init__(self, seed=0):
        self.generator = tf.random.Generator.from_seed(seed)
        self.normal = self.generator.normal
        self.uniform = self.generator.uniform
    
    def get_state(self):
        return self.generator.state.numpy()
        
    def set_state(self, state):
        tf.compat.v1.assign(self.generator.state, tf.constant(state))
    
    def randint(self, shape, minval, maxval):
        return self.uniform(shape, minval=minval, maxval=maxval, dtype=tf.int32)
    