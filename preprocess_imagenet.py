#script for downloading imagenet and converting to tfrecord

#first run the following commands
# mkdir ./datasets/imagenet256/
# wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar -P ./datasets/imagenet256
# mkdir ./datasets/imagenet256/zipfiles
# tar -xvf ./datasets/imagenet256/ILSVRC2012_img_train.tar -C ./datasets/imagenet256/zipfiles
# rm ./datasets/imagenet256/ILSVRC2012_img_train.tar

import tensorflow as tf
import tarfile
import os
import shutil
import numpy as np
import cv2
from time import time
import gc
import random



data_dir = './datasets/imagenet256'
zipfile_dir = os.path.join(data_dir, 'zipfiles')
png_dir = './png_resized' #will contain all 1.2m images
if not os.path.isdir(png_dir): os.mkdir(png_dir)


def center_crop(x, crop_h, resolution=64):
    h, w = x.shape[:2]
    crop_h = min(h, w)  
    crop_w = crop_h

    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    x = x[j:j+crop_h, i:i+crop_w]
    return cv2.resize(x, (resolution, resolution))
    
def get_image(image_path, image_size, resize_w=256):
    x = cv2.imread(image_path)[:, :, ::-1]  #cv2 reads BGR, so we use ::-1
    out = center_crop(x, image_size, resize_w)
    return out

def convert_jpeg_to_png(dir, png_dir, classlabel):
    for imname in os.listdir(dir):
        impath = os.path.join(dir, imname)
        im = get_image(impath, image_size=256, resize_w=256)
        im = tf.convert_to_tensor(im)
        png_path = '{}/{}_label{}_png'.format(png_dir, imname[:-5], classlabel)  
        png_im = tf.io.encode_png(im)
        tf.io.write_file(png_path, png_im)

for i, name in enumerate(os.listdir(zipfile_dir)):
    tmp_dir = './tmp_jpeg_images'
    if not os.path.isdir(tmp_dir): os.mkdir(tmp_dir)

    zipfile_path = os.path.join(zipfile_dir, name)
    file = tarfile.open(zipfile_path)
    file.extractall(tmp_dir)
    file.close()

    convert_jpeg_to_png(tmp_dir, png_dir, classlabel=i)
    shutil.rmtree(tmp_dir)
    print(i, zipfile_path)
print(len(os.listdir(png_dir)))



filenames = os.listdir(png_dir)
random.shuffle(filenames)



shardsize = 2560
shardnum = 0
shard_x, shard_y = [], []
for filename in filenames:    
    png_im = tf.io.read_file(os.path.join(png_dir, filename))

    shard_x.append(im)
    label = int(filename[filename.index('label')+5  :  filename.index('_png')])
    shard_y.append(label)

    if len(shard_x)==shardsize or filename==filenames[-1]:
        shard_x = np.array(shard_x)
        shard_y = np.array(shard_y)
        print(shardnum, shard_x.shape, shard_y.shape)
        np.savez('{}/shard{}.npz'.format(data_dir, shardnum), x=shard_x, y=shard_y)

        shardnum += 1
        shard_x, shard_y = [], []

shutil.rmtree(png_dir)

datadir = './drive/My Drive/datasets/imagenet256'
tfrecord_dir = './drive/My Drive/datasets/imagenet256/tfrecord'
shardlist = os.listdir(datadir)[2:]

shard_counter = 0
shardsize = 3200

def write_feature(file_writer, x, y):
    #x and y are single images/labels
    xfeature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[x]))
    yfeature = tf.train.Feature(int64_list=tf.train.Int64List(value=[y]))
    features_for_example = {'x': xfeature, 'y': yfeature}
    example_proto = tf.train.Example(features=tf.train.Features(feature=features_for_example))
    file_writer.write(example_proto.SerializeToString())

for shard in shardlist:
    shardname = os.path.join(datadir, shard)
    shard_values = np.load(shardname)
    shardx = shard_values["x"]
    shardy = shard_values["y"]
    s = time()
    gc.collect()

    for i in range(3): #WHY 3??
        tfrecord_path = os.path.join(tfrecord_dir, "example{}.tfrecords".format(shard_counter*3+i))
        x = tf.convert_to_tensor(shardx[i*SHARDSIZE : i*SHARDSIZE+SHARDSIZE])
        x_png = tf.vectorized_map(tf.io.encode_png, x)
        y = shardy[i*SHARDSIZE : i*SHARDSIZE+SHARDSIZE]
        print(shard, shard_counter*3+i, i, time()-s)

        with tf.io.TFRecordWriter(tfrecord_path) as file_writer:
            for j in range(SHARDSIZE):
                write_feature(file_writer, x_png[j].numpy(), y[j])
        gc.collect()
    
    shard_counter += 1