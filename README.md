# Patch Diffusion

Code for the paper "Improving Diffusion Model Efficiency Through Patching". The core idea of the paper is to insert a ViT-style patching operation at the beginning of the U-Net, letting it operate on data with smaller height and width. We show in our paper that the optimal prediction for **x** is quite blurry for most timesteps, and therefore convolutions at the original resolution are usually not necessary. This causes a considerable reduction in compute cost: For example, when using a patch size of 4 (P = 4), generating 256x256 images costs only as much as generating 64x64 images normally (with P = 1). 

# Pretrained Models

We include our models for ImageNet 256x256 and FFHQ 1024x1024, as well as 3 LSUN models with P=2, P=4, and P=8. 

You can download them from Google Drive:

 * ImageNet 256x256, Split #0: [ema_weights.h5](https://drive.google.com/file/d/1k8nLu6dvUylwNegmIMOAtqE9sGebNG68/view?usp=sharing)
 * ImageNet 256x256, Split #1: [ema_weights.h5](https://drive.google.com/file/d/1-8H16U50C_LqVKHLnh0sVCjFjHp5GHWp/view?usp=sharing)
 * FFHQ 1024x1024: [ema_weights.h5](https://drive.google.com/file/d/1hFD9jxxd3QBDuKBRobI4zGg9LCExVkhx/view?usp=sharing)
 * LSUN 256x256, P=2: [ema_weights.h5](https://drive.google.com/file/d/1piyyBza7A_xLR2bbStO7_5h6Hyu3PS_v/view?usp=sharing)
 * LSUN 256x256, P=4: [ema_weights.h5](https://drive.google.com/file/d/1-1ZGwkMWjQryELmayjmbkTjv-aqpcaHQ/view?usp=sharing)
 * LSUN 256x256, P=8: [ema_weights.h5](https://drive.google.com/file/d/1-4CPg5XwZLSEBPoO3lLAeRtRdmW6W8Rz/view?usp=sharing)

# Sampling Instructions

First, clone our repository and change directory into it; then install requirements.txt. For sampling, you'll also want to download our pretrained models, and put them in their corresponding directory (e.g. ./models/ffhq/ema_weights.h5 for ffhq).

Assuming you have downloaded the relevant models in ./models, run sampling_script.py to sample from our models. 

Example usage:
python sampling_script.py gpu grid ffhq 2 ./ffhq_ims --batch_size 2 --timesteps 250


We trained our models for a relatively short duration: our ImageNet models trained for a combined 32 V-100 days (approximately), while our FFHQ model trained for roughly 14 V-100 days. Our LSUN models trained for about 5 V-100 days each. In general, longer training is recommended if you have the budget for it - it improves results.


# Citation:

If you find this work helpful to your research, please cite us:

@misc{https://doi.org/10.48550/arxiv.2207.04316,
  doi = {10.48550/ARXIV.2207.04316},
  
  url = {https://arxiv.org/abs/2207.04316},
  
  author = {Luhman, Troy and Luhman, Eric},
  
  keywords = {Machine Learning (cs.LG), Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Improving Diffusion Model Efficiency Through Patching},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}
