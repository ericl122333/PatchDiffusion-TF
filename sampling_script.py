import tensorflow as tf
import os
from unet import UNet
from utils import get_strategy
import pickle
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from diffusion_utils import Diffusion, RNG
import tqdm.auto as tqdm
import numpy as np
from IPython import display
from configs import get_config
import imageio

strategy, _ = get_strategy(device="Colab TPU")
display.clear_output()

def get_model(expm_name, policy_name, strategy):
    if 'imagenet' in expm_name:
        margs, targs1 = get_config('imagenet_top')
        _, targs2 = get_config('imagenet_bottom')

        with strategy.scope():
            model1 = UNet(image_resolution=margs.res, 
                        c=margs.c, 
                        num_res_blocks=margs.n_blocks, 
                        droprate=margs.droprate, 
                        attn_resos=margs.attn_resos, 
                        c_mult=margs.c_mult, 
                        num_classes=1000, 
                        policy_name=policy_name, patchsize=margs.patch_size, 
                        se=margs.se)
            model1.build_network()
        ema_mpath1 = os.path.join(targs1.model_dir, "ema_weights.h5") 
        model1.load_weights(ema_mpath1)

        with strategy.scope():
            model2 = UNet(image_resolution=margs.res, 
                        c=margs.c, 
                        num_res_blocks=margs.n_blocks, 
                        droprate=margs.droprate, 
                        attn_resos=margs.attn_resos, 
                        c_mult=margs.c_mult, 
                        num_classes=1000, 
                        policy_name=policy_name, patchsize=margs.patch_size, 
                        se=margs.se)
            model2.build_network()
        ema_mpath2 = os.path.join(targs1.model_dir, "ema_weights.h5") 
        model2.load_weights(ema_mpath2)
        return model1, model2
    else:
        margs, targs = get_config(expm_name)
        with strategy.scope():
            model = UNet(image_resolution=margs.res, 
                        c=margs.c, 
                        num_res_blocks=margs.n_blocks, 
                        droprate=margs.droprate, 
                        attn_resos=margs.attn_resos, 
                        c_mult=margs.c_mult, 
                        num_classes=0, 
                        policy_name=policy_name, patchsize=margs.patch_size, 
                        se=margs.se)
            model.build_network()
        ema_mpath = os.path.join(targs.model_dir, "ema_weights.h5") 
        model.load_weights(ema_mpath)
        return model

def space_timesteps(num_timesteps, section_counts):
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return list(all_steps)

print(space_timesteps(1000, [250]))

def create_last_step(strategy, network, diffusion, output_type):
    @tf.function
    def last_step(xt, t, y):
        def run_last_step(xt, t, y):
            mean_output, _ = network(xt, t, y)
            x0 = diffusion.get_pred_x0(xt, mean_output, t, output_type=output_type)
            return x0
        return strategy.run(run_last_step, args=(xt, t, y))
    return last_step

def create_reverse_step(strategy, network, diffusion, output_type):
    @tf.function 
    def cf_reverse_step(xt, t, alpha, alpha_next, y):
        def run_reverse_step(xt, t, alpha, alpha_next, y):
            beta = (1 - alpha / alpha_next)
            b_tilde = beta * (1 - alpha_next) / (1 - alpha)

            mean_output, var_output = network(xt, t, y)
            x0 = diffusion.get_pred_x0(xt, mean_output, t, output_type=output_type)
            sigma = tf.sqrt(diffusion.get_pred_var(var_output, b_tilde, beta))

            x_t_minus1 = (xt - beta*(xt - tf.sqrt(alpha)*x0) / (1-alpha)) / tf.sqrt(1-beta)
            x_t_minus1 += sigma * diffusion.rng.normal(xt.shape)
            return x_t_minus1
        return strategy.run(run_reverse_step, args=(xt, t, alpha, alpha_next, y))
    return cf_reverse_step

def create_cf_reverse_step(strategy, network, W, diffusion, output_type):
    @tf.function 
    def cf_reverse_step(xt, t, alpha, alpha_next, y):
        def run_reverse_step(xt, t, alpha, alpha_next, y):
            beta = (1 - alpha / alpha_next)
            b_tilde = beta * (1 - alpha_next) / (1 - alpha)

            mean_cond, var_output = network(xt, t, y)
            mean_uncond, _ = network(xt, t, tf.fill(y.shape, 1000))
            
            x0_cond = diffusion.get_pred_x0(xt, mean_cond, t, output_type=output_type)
            x0_uncond = diffusion.get_pred_x0(xt, mean_uncond, t, output_type=output_type)

            x0 = W * x0_cond - (W - 1.)*x0_uncond
            s = tfp.stats.percentile(tf.abs(x0), 99.5, axis=(-1,-2,-3), keepdims=True)
            s = tf.math.maximum(s, 1.0)
            x0 = tf.clip_by_value(x0, -s, s) / s
            
            sigma = tf.sqrt(diffusion.get_pred_var(var_output, b_tilde, beta))
            x_t_minus1 = (xt - beta*(xt - tf.sqrt(alpha)*x0) / (1-alpha)) / tf.sqrt(1-beta)
            x_t_minus1 += sigma * diffusion.rng.normal(xt.shape)
            return x_t_minus1
        return strategy.run(run_reverse_step, args=(xt, t, alpha, alpha_next, y))
    return cf_reverse_step


def sampling_script(device, action, dataset, num_images, save_path, batch_size, timesteps, modified_stride, guidance, policy, seed):
    if device=='colabtpu': device = 'Colab TPU'
    strategy, strategy_type = get_strategy(device=device)
    display.clear_output()
    
    with strategy.scope():
        if seed=='random': rng = RNG(seed=np.random.randint(0, 1000000))
        else: rng = RNG(seed=seed)


    if modified_stride: seq = space_timesteps(1000, [90, 60, 60, 20, 20])
    else: seq = space_timesteps(1000, [timesteps])
    seq_next = [-1] + list(seq[:-1])

    if dataset=='imagenet':
        model1, model2 = get_model('imagenet', policy, strategy)
        _, targs1 = get_config('imagenet_top')
        _, targs2 = get_config('imagenet_top')
        diffusion = Diffusion(rng, beta_max=targs1.diffusion_beta_max, schedule='linear')

        if guidance > 1.0:
            network1_call = create_cf_reverse_step(strategy, model1, guidance, diffusion, targs1.output_type)
            network2_call = create_cf_reverse_step(strategy, model2, guidance, diffusion, targs2.output_type)
        else:
            network1_call = create_reverse_step(strategy, model1, diffusion, targs1.output_type)
            network2_call = create_reverse_step(strategy, model2, diffusion, targs2.output_type)
        final_call = create_last_step(strategy, model1, diffusion)
        xshape = [batch_size] + model1.inp_shape
    else:
        model = get_model(dataset, policy, strategy)
        _, targs = get_config(dataset)
        diffusion = Diffusion(rng, beta_max=targs.diffusion_beta_max, schedule='linear')
        network_call = create_reverse_step(strategy, model, diffusion, targs.output_type)
        final_call = create_last_step(strategy, model, diffusion, targs.output_type)
        xshape = [batch_size] + model.inp_shape
    

    def rand_normal(class_label):
        with strategy.scope():
            xtr = diffusion.rng.normal(xshape, dtype=tf.float32)
        return xtr, class_label

    if action=='fid': labels = tf.math.floormod(tf.range(num_images), 1000)
    else: labels = tf.random.uniform([num_images,], maxval=1000, dtype=tf.int32)
    samples_xT = tf.data.Dataset.from_tensor_slices((labels)).batch(batch_size, drop_remainder=False)
    samples_xT = samples_xT.map(rand_normal)
    if strategy_type=='TPU': samples_xT = strategy.experimental_distribute_dataset(samples_xT)

    batches = []
    for X, Y in samples_xT:
        for t, t_next in tqdm.tqdm(zip(reversed(seq), reversed(seq_next))):
            bs = batch_size // strategy.num_replicas_in_sync
            index = diffusion.to_timestep(1, t)
            alpha = diffusion.alpha_set[t] * tf.ones([1, 1, 1, 1]) 
            alpha_next = diffusion.alpha_set[t_next] if t_next>=0 else diffusion.alpha_set[0]
            alpha_next = alpha_next * tf.ones([1, 1, 1, 1]) 

            if dataset=='imagenet':
                if t >= 396: X = network2_call(X, index, alpha, alpha_next, Y)
                elif t > seq[0]: X = network1_call(X, index, alpha, alpha_next, Y)
                else: X = final_call(X, index, Y)
            else:
                if t > seq[0]: X = network_call(X, index, alpha, alpha_next, tf.zeros([bs,]))
                else: X = final_call(X, index, tf.zeros([bs,]))

        if strategy_type=='TPU': X = tf.concat(X.values, axis=0)
        X = (tf.clip_by_value(X, -1.0, 1.0) + 1.0) * 127.5 
        batches.append(X)

    all_images = np.concatenate(tuple(batches), axis=0)[:10000]
    all_images = all_images.astype('uint8')
    if action=='fid':
        npz_path = save_path + '.npz'
        np.savez(save_path, all_images, labels.numpy())
    elif action=='grid':
        dims=[np.floor(np.sqrt(num_images)), np.floor(np.sqrt(num_images))]
        show_images(images, scale=15, savepath=save_path, dims=dims)
    elif action=='folder':
        if not os.path.isdir(save_path): os.mkdir(save_path)
        for i, img in enumerate(all_images):
            imgpath = os.path.join(save_path, 'images{}.png'.format(str(i)))
            imageio.imwrite(imgpath, img)
    return all_images

if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("device", type=str, help="'colabtpu' for colab TPU, 'cpu' for CPU, otherwise will use GPU (single gpu only)")
    parser.add_argument("action", type=str, help="'fid' to write to an npz file, 'grid' to create a grid of images, 'folder' to write individual images")
    parser.add_argument("dataset", type=str, help="Which dataset's images to write")
    parser.add_argument("num_images", type=int, help="How many images to write")
    parser.add_argument("save_path", type=str, help="save location, will add a .npz extension if action='FID', .png if 'grid', directory of .png if 'folder'")
    parser.add_argument("--batch_size", type=int, default=16, help="Global batch size")
    parser.add_argument("--timesteps", type=str, default=1000, help="How many timesteps to use")
    parser.add_argument("--modified_stride", type=bool, default=False, help="Whether to use the LSUN timestep spacing from ADM, this sets timesteps=250 automatically")
    parser.add_argument("--guidance", type=float, default=1.5, help="classifier-free guidance weights, 1.0 for no guidance, only applies to imagenet")
    parser.add_argument("--policy", type=str, default='fp32', help="one of fp32, fp16, bfloat16")
    parser.add_argument("--seed", type=int, default='random', help="input an integer to fix the seed")
    
    
    args = parser.parse_args()
    main(args.device, args.action, args.dataset, args.num_images, args.save_path, args.batch_size, args.timesteps, args.modified_stride, args.guidance, args.policy, args.seed)
