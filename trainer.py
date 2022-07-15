import tensorflow as tf
import os
from unet import UNet
from utils import get_strategy, WarmupSchedule, make_mapfn, set_policy
from diffusion_utils import Diffusion, RNG, hybrid_lossfn
from configs import get_config

import numpy as np
import pickle
from time import time
import random

class Trainer():
    def __init__(self, training_args, model_args):
        self.targs = targs = training_args 
        self.margs = margs = model_args

        self.strategy, self.device_type = get_strategy(targs.device)

        set_policy(margs.policy, self.device_type)
        
        self.dataset = targs.dataset
        self.model_dir = targs.model_dir
        self.data_dir = targs.data_dir
        self.prev_model_dir = targs.prev_model_dir
        self.output_type = targs.output_type
        self.schedule = targs.schedule
    
        self.num_classes = 1000 if targs.class_cond else 0
        self.p = margs.patch_size
        if not os.path.exists(self.model_dir): os.mkdir(self.model_dir)

        self.make_training_objects()
        self.make_dataset()

        #get range of timesteps.
        start_snr, end_snr = targs.start_snr, targs.end_snr
        if start_snr is None: start_snr = 1e4
        if end_snr is None: 
            a = np.cumprod(1-np.linspace(1e-4,targs.beta_max,1000))
            end_snr = a[-1] / (1-a[-1])

        self.lowest_t = self.diffusion.get_timestep_from_snr(start_snr)
        self.highest_t = self.diffusion.get_timestep_from_snr(end_snr)
        print(f"Lowest seen timestep: {self.lowest_t}, highest seen timestep: {self.highest_t}")
        print(f"Output type: {self.diffusion.output_type}")
        print(f"Number of Classes: {self.num_classes}")
        #make logfile
        self.logfile_path = os.path.join(self.model_dir, 'logfile.txt')
        if not os.path.exists(self.logfile_path):
            with open(self.logfile_path, 'w') as f:
                f.write(f"Logging file for model training on snrs of [{start_snr}, {end_snr}]")
        
    def model_function(self): #generates a new unet
        margs = self.margs
        return UNet(image_resolution=margs.res, c=margs.c, num_res_blocks=margs.n_blocks, droprate=margs.droprate, attn_resos=margs.attn_resos, c_mult=margs.c_mult, 
                    num_classes=self.num_classes, policy_name=margs.policy, patchsize=self.p, se=margs.se)
    
    def make_optimizer(self):
        oargs = self.targs.optimizer_args
        learnrate = WarmupSchedule(oargs.lr, oargs.warmup_steps)
        optimizer = tf.keras.optimizers.Adam(learnrate, beta_1=oargs.beta_1, beta_2=oargs.beta_2)
        return optimizer

    def make_training_objects(self):
        with self.strategy.scope():
            self.model = self.model_function()
            self.model.build_network()
            self.optimizer = self.make_optimizer()

            self.rng = RNG(seed=0)
            self.diffusion = Diffusion(self.rng, self.targs.beta_max, schedule=self.schedule)

            self.loss_metric = tf.keras.metrics.Mean()
            self.vlb_metric = tf.keras.metrics.Mean()
            self.grad_norm = tf.keras.metrics.Mean()
        
        self.ema_weights = self.model.get_weights()
        assert isinstance(self.ema_weights[0], np.ndarray)
        print('model dtype', self.model.compute_dtype)
        print('Model Parameters', self.model.count_params())
    
    def make_dataset(self):
        margs, targs = self.margs, self.targs
        mapfn = make_mapfn(margs.class_cond, margs.res)
        
        if 'imagenet' in self.dataset_name.lower():
            dataset_size = 401  #1281167 // 3200
        elif 'ffhq' in self.dataset_name.lower():
            dataset_size = 274
        elif 'church' in self.dataset_name.lower(): 
            dataset_size = 50
        else:
            raise NotImplementedError()

        filenames = [os.path.join(self.data_dir, f"example{i}.tfrecords") for i in range(dataset_size)]
        random.shuffle(filenames)
        raw_dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=tf.data.AUTOTUNE)
        raw_dataset = raw_dataset.map(mapfn)

        raw_dataset = raw_dataset.shuffle(targs.batch_size*5).batch(targs.batch_size).repeat()
        self.dataset = self.strategy.experimental_distribute_dataset(raw_dataset)

    def make_forward_pass(self):
        def forward_pass_fn(x, y):
            model, diffusion = self.model, self.diffusion

            batch_size = x.shape[0]
            timesteps = diffusion.sample_timesteps(batch_size, minval=self.lowest_t, maxval=self.highest_t)
            xt, eps = diffusion.draw_from_posterior(x, timesteps)
            
            mean_output, var_output = model(xt, timesteps, y) #may be either x prediction or eps prediction
            x0_output = diffusion.get_pred_x0(xt, mean_output, timesteps, self.output_type)

            L_simple, L_vlb = hybrid_lossfn(x, x0_output, var_output, timesteps, diffusion)
            return L_simple, L_vlb
            
        return forward_pass_fn

    @tf.function
    def train_step(self, x, y):
        forward_pass = self.make_forward_pass()
        model, optimizer, loss_metric, vlb_metric, grad_norm = self.model, self.optimizer, self.loss_metric, self.vlb_metric, self.grad_norm
        def train_step_fn(x, y):
            y = model.get_classifier_free_label(y)
            with tf.GradientTape() as tape:
                loss, L_vlb = forward_pass(x, y)
                combined_loss = loss + L_vlb
            gradients = tape.gradient(combined_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            gnorm = tf.linalg.global_norm(gradients)

            loss_metric(loss)
            vlb_metric(L_vlb)
            grad_norm(gnorm)
        return self.strategy.run(train_step_fn, args=(x, y))
    
    def get_names(self):
        m_dir = self.model_dir
        nonema_mpath = os.path.join(m_dir, 'nonema_model.p')
        opt_path = os.path.join(m_dir, 'opt.p')
        ema_path = os.path.join(m_dir, 'ema_model.p')
        rng_path = os.path.join(m_dir, 'rng.p')
        return nonema_mpath, opt_path, ema_path, rng_path
    
    def save_checkpoint(self, verbose=True):
        checkpoint_path = os.path.join(self.model_dir, "checkpoint.p")
        checkpoint = {
            "nonema_model": self.model.get_weights(),
            "opt_weights": self.optimizer.get_weights(),
            "ema_model": self.ema_weights,
            "rng_state": self.rng.get_state()
        }
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f, pickle.HIGHEST_PROTOCOL)
        
        if verbose:
            self.print_and_log(f"Saved new checkpoint at location {checkpoint_path}")
        
    def load_checkpoint(self, verbose=True):
        checkpoint_path = os.path.join(self.model_dir, "checkpoint.p")
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.model.set_weights(checkpoint["nonema_model"])
        self.optimizer.set_weights(checkpoint["opt_weights"])
        self.ema_weights = checkpoint["ema_model"]
        self.rng.set_state(checkpoint["rng_state"])

        if verbose:
            self.print_and_log(f"loaded previous checkpoint from {checkpoint_path}")

    def make_strategy_fn(self, function):
        @tf.function
        def strategy_fn(x, y):
            return self.strategy.run(function, args=(x, y))
        return strategy_fn
    
    def print_and_log(self, string, end='\n'):
        #a tool to print and write to a logfile using one convenience function. will be used to restore the lowest loss EMA model. replace the print statements in training with this when done.
        print(string, end=end)
        with open(self.logfile_path, 'a') as f:
            f.write(string+end)
    
    def update_ema(self):
        ema_rate = self.targs.optimizer_args.ema_rate
        new_weights = self.model.get_weights()
        self.ema_weights = [ema_rate * w + (1-ema_rate) * w2 for w, w2 in zip(self.ema_weights, new_weights)]

    def train(self):
        strategy, targs = self.strategy, self.targs
        printl = self.print_and_log

        model, optimizer, loss_metric, vlb_metric, grad_norm = self.model, self.optimizer, self.loss_metric, self.vlb_metric, self.grad_norm

        checkpoint_path = os.path.join(self.model_dir, "checkpoint.p")
        if self.prev_model_dir is not None: prev_model_path = os.path.join(self.prev_model_dir, "ema_weights.h5")

        if os.path.isfile(checkpoint_path):
            for x, y in self.dataset:
                self.train_step(x, y) #trains one step to initialize model weights
                loss_metric.reset_states()
                vlb_metric.reset_states()
                grad_norm.reset_states()
                break
            self.load_checkpoint()
        elif os.path.isfile(prev_model_path) and self.prev_model_dir is not None: 
            printl(f"No previous checkpoints found for this run, starting new training run by initializing from the model at {prev_model_path}")
            self.model.load_weights(prev_model_path) #this line will fail if the 2 models have different configs
        else:
            printl(f"No previous checkpoints or old models found, starting new training run by initializing from scratch")
        
        s = time()
        current_opt_iters = optimizer.iterations.numpy()
        printl("Current Optimizer iterations: %d" % current_opt_iters)
        
        max_iter = targs.iterations
        for x, y in self.dataset:
            self.train_step(x, y)

            if current_opt_iters%targs.optimizer_args.ema_freq==0: self.update_ema()
        
            if current_opt_iters%targs.print_every == 0:
                printl(f"Optimizer iterations {optimizer.iterations.numpy()},  Time {round(time()-s)} sec", end= ', ')
                printl(f"train loss: {loss_metric.result().numpy()}, L_vlb: {vlb_metric.result().numpy()}, gradient norms: {grad_norm.result().numpy()}")
                loss_metric.reset_states()
                vlb_metric.reset_states()
                grad_norm.reset_states()

            if current_opt_iters%targs.save_every == 0:
                self.save_checkpoint()


            if current_opt_iters >= max_iter:
                printl(f"Training is completed. The model was trained for a maximum of {optimizer.iterations.numpy()} iterations")
                ema_mpath = os.path.join(self.model_dir, "ema_weights.h5") 
                self.model.set_weights(self.ema_weights) 
                self.model.save_weights(ema_mpath) #use h5 instead of pickle when saving the final model for more secure sharing
                printl(f"The EMA Model is available at {ema_mpath}")
                break
            
            current_opt_iters += 1
        
        
if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("dataset", type=str, help="Which dataset's images to write")
    args = parser.parse_args()

    model_args, training_args = get_config(args.dataset)
    trainer = Trainer(training_args, model_args)
    trainer.train()
