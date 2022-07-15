from ml_collections.config_dict import ConfigDict

model_args_lsun_p2 = {
    'class_cond': False,
    'droprate': 0.0,
    'res': 256, 
    'c': 128,
    'n_blocks': 2,
    'attn_resos': [16, 8], 
    'c_mult': [1, 2, 2, 4, 4], 
    'policy': 'mixed_bfloat16',
    'patch_size': 2,
    'se': False
}
training_args_lsun_p2 = {
    'start_snr': None,
    'end_snr': None,
    'diffusion_beta_max': 0.02,

    'dataset': 'lsun',
    'model_dir': './models/lsun_p2',
    'data_dir': 'gs://patchdpmdata/church/tfrecord',

    'output_type': 'x0',
    'schedule': 'linear',
    'T': 1000,

    'batch_size': 128,
    'iterations': 335000,

    'print_every': 500,
    'save_every': 5000,
    'device': "Colab TPU",

    'optimizer_args': ConfigDict({
        'lr': 1e-4,
        'beta_1': 0.9,
        'beta_2': 0.99,
        'warmup_steps': 5000, 
        'ema_rate': 0.99, 
        'ema_freq': 100,
    })
}

#-----------------------------------------

model_args_lsun_p4 = {
    'class_cond': False,
    'droprate': 0.0,
    'res': 256, 
    'c': 256,
    'n_blocks': 2,
    'attn_resos': [16, 8], 
    'c_mult': [1, 1, 2, 2], 
    'policy': 'mixed_bfloat16',
    'patch_size': 4,
    'se': False
}
training_args_lsun_p4 = {
    'start_snr': None,
    'end_snr': None,
    'diffusion_beta_max': 0.02,

    'dataset': 'lsun',
    'model_dir': './models/lsun_p4',
    'data_dir': 'gs://patchdpmdata/church/tfrecord',

    'output_type': 'x0',
    'schedule': 'linear',
    'T': 1000,

    'batch_size': 256,
    'iterations': 275000,

    'print_every': 500,
    'save_every': 5000,
    'device': "Colab TPU",

    'optimizer_args': ConfigDict({
        'lr': 1e-4,
        'beta_1': 0.9,
        'beta_2': 0.99,
        'warmup_steps': 5000, 
        'ema_rate': 0.99,  
        'ema_freq': 100,
    })
}

#-----------------------------------------

model_args_lsun_p8 = {
    'class_cond': False,
    'droprate': 0.0,
    'res': 256,
    'c': 256,
    'n_blocks': 3,
    'attn_resos': [16, 8], 
    'c_mult': [1, 1.5, 2], 
    'policy': 'mixed_bfloat16',
    'patch_size': 8,
    'se': False
}
training_args_lsun_p8 = {
    'start_snr': None,
    'end_snr': None,
    'diffusion_beta_max': 0.02,

    'dataset': 'lsun',
    'model_dir': './models/lsun_p8',
    'data_dir': 'gs://patchdpmdata/church/tfrecord',

    'output_type': 'x0',
    'schedule': 'linear',
    'T': 1000,

    'batch_size': 512,
    'iterations': 215000,

    'print_every': 500,
    'save_every': 5000,
    'device': "Colab TPU",

    'optimizer_args': ConfigDict({
        'lr': 1e-4,
        'beta_1': 0.9,
        'beta_2': 0.99,
        'warmup_steps': 5000, 
        'ema_rate': 0.99, 
        'ema_freq': 100,
    })
}

#-----------------------------------------

model_args_imagenet = {
    'class_cond': True,
    'res': 256,
    'c': 256,
    'n_blocks': 3,
    'droprate': 0.0,
    'attn_resos': [16, 8],
    'c_mult': [1, 2, 2, 2],
    'policy': 'mixed_bfloat16',
    'patch_size': 4,
    'se': True
}
training_args_imagenet_top = {
    'start_snr': 0.25,
    'end_snr': None,
    'beta_max': 0.02,

    'dataset': 'imagenet256',
    'model_dir': './models/i256_model0',
    'data_dir': 'gs://patchdpmdata/imagenet256/tfrecord',
    'prev_model_dir': None,

    'output_type': 'x0',
    'schedule': 'linear',
    'T': 1000,

    'batch_size': 256,
    'iterations': 5e5,
    'print_every': 250,
    'save_every': 2500,
    'device': "Colab TPU",

    'optimizer_args': ConfigDict({
        'lr': 1e-4,
        'beta_1': 0.9,
        'beta_2': 0.98,
        'warmup_steps': 5000, 
        'ema_rate': 0.99, 
        'ema_freq': 100,
    })
}
training_args_imagenet_bottom = {
    'start_snr': None,
    'end_snr': 0.25,
    'beta_max': 0.02,

    'dataset': 'imagenet256',
    'model_dir': './models/i256_model1',
    'data_dir': 'gs://patchdpmdata/imagenet256/tfrecord',  
    'prev_model_dir': './models/i256_model0',

    'output_type': 'x0',
    'schedule': 'linear',
    'T': 1000,

    'batch_size': 256,
    'iterations': 5e5,
    'print_every': 250,
    'save_every': 2500,
    'device': "Colab TPU",

    'optimizer_args': ConfigDict({
        'lr': 5e-5,
        'beta_1': 0.9,
        'beta_2': 0.999,
        'warmup_steps': 10000, 
        'ema_rate': 0.99, 
        'ema_freq': 100,
    })
}


#-----------------------------------------

model_args_ffhq = {
    'class_cond': False,
    'droprate': 0.0,
    'res': 1024, 
    'c': 128,
    'n_blocks': 2,
    'attn_resos': [16, 8], 
    'c_mult': [1, 2, 2, 4, 4, 4], 
    'policy': 'mixed_bfloat16',
    'patch_size': 4,
    'se': False
}
training_args_ffhq = {
    'start_snr': None,
    'end_snr': None,
    'diffusion_beta_max': 0.025,

    'dataset': 'ffhq',
    'model_dir': './models/ffhq',
    'data_dir': 'gs://patchdpmdata/ffhq/tfrecord',

    'output_type': 'x0',
    'schedule': 'linear',
    'T': 1000,

    'batch_size': 32,
    'iterations': 780000,

    'print_every': 1250,
    'save_every': 5000,
    'device': "Colab TPU",

    'optimizer_args': ConfigDict({
        'lr': 1e-4,
        'beta_1': 0.9,
        'beta_2': 0.98,
        'warmup_steps': 5000, 

        'ema_freq': 100,
        'ema_rate': 0.99 
    })
}

def get_config(expm_name):
    expm_name = expm_name.lower() 
    model_args_dict = {
        'church_p2': model_args_lsun_p2,
        'church_p4': model_args_lsun_p4,
        'church_p8': model_args_lsun_p8,
        'imagenet_top': model_args_imagenet,
        'imagenet_bottom': model_args_imagenet,
        'ffhq': model_args_ffhq
    } 
    training_args_dict = {
        'church_p2': training_args_lsun_p2,
        'church_p4': training_args_lsun_p4,
        'church_p8': training_args_lsun_p8,
        'imagenet_top': training_args_imagenet_top,
        'imagenet_bottom': training_args_imagenet_bottom,
        'ffhq': training_args_ffhq
    } 
    try:
        model_args = model_args_dict[expm_name]
        training_args = training_args_dict[expm_name]
    except KeyError:
        print("expm_name must be one of 'church_p2', 'church_p4', 'church_p8', \
            'imagenet_top', 'imagenet_bottom', 'ffhq'")
        return 1
    
    model_args = ConfigDict(model_args)
    training_args = ConfigDict(training_args)
    return model_args, training_args


