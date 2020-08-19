import pickle
import numpy as np
import os
from urllib.request import urlretrieve
import tarfile
import sys
import time
import tensorflow as tf
import math


DATASET_DIRECTORY = './dataset/'
CIFAR10_DIRECTORY = 'cifar-10-batches-py/'
URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
META_FILE = DATASET_DIRECTORY+CIFAR10_DIRECTORY+'batches.meta'
NUM_CLASSES = 10

def augment(images, labels,
            resize=None, # (width, height) tuple or None
            horizontal_flip=False,
            vertical_flip=False,
            rotate=0, # Maximum rotation angle in degrees
            crop_probability=0, # How often we do crops
            crop_min_percent=0.6, # Minimum linear dimension of a crop
            crop_max_percent=1.,  # Maximum linear dimension of a crop
            mixup=0):  # Mixup coeffecient, see https://arxiv.org/abs/1710.09412.pdf
  if resize is not None:
    images = tf.image.resize_bilinear(images, resize)
  
  # My experiments showed that casting on GPU improves training performance
  images = tf.image.convert_image_dtype(images, dtype=tf.float32)
  images = tf.subtract(images, 0.5)
  images = tf.multiply(images, 2.0)
  labels = tf.to_float(labels)
  

  with tf.name_scope('augmentation'):
    shp = tf.shape(images)
    batch_size, height, width = shp[0], shp[1], shp[2]
    width = tf.cast(width, tf.float32)
    height = tf.cast(height, tf.float32)

    # The list of affine transformations that our image will go under.
    # Every element is Nx8 tensor, where N is a batch size.
    transforms = []
    identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
    if horizontal_flip:
      coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
      flip_transform = tf.convert_to_tensor(
          [-1., 0., width, 0., 1., 0., 0., 0.], dtype=tf.float32)
      transforms.append(
          tf.where(coin,
                   tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                   tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

    if vertical_flip:
      coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
      flip_transform = tf.convert_to_tensor(
          [1, 0, 0, 0, -1, height, 0, 0], dtype=tf.float32)
      transforms.append(
          tf.where(coin,
                   tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                   tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

    if rotate > 0:
      angle_rad = rotate / 180 * math.pi
      angles = tf.random_uniform([batch_size], -angle_rad, angle_rad)
      transforms.append(
          tf.contrib.image.angles_to_projective_transforms(
              angles, height, width))

    if crop_probability > 0:
      crop_pct = tf.random_uniform([batch_size], crop_min_percent,
                                   crop_max_percent)
      left = tf.random_uniform([batch_size], 0, width * (1 - crop_pct))
      top = tf.random_uniform([batch_size], 0, height * (1 - crop_pct))
      crop_transform = tf.stack([
          crop_pct,
          tf.zeros([batch_size]), top,
          tf.zeros([batch_size]), crop_pct, left,
          tf.zeros([batch_size]),
          tf.zeros([batch_size])
      ], 1)

      coin = tf.less(
          tf.random_uniform([batch_size], 0, 1.0), crop_probability)
      transforms.append(
          tf.where(coin, crop_transform,
                   tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

    if transforms:
      images = tf.contrib.image.transform(
          images,
          tf.contrib.image.compose_transforms(*transforms),
          interpolation='BILINEAR') # or 'NEAREST'

    def cshift(values): # Circular shift in batch dimension
      return tf.concat([values[-1:, ...], values[:-1, ...]], 0)

    if mixup > 0:
      mixup = 1.0 * mixup # Convert to float, as tf.distributions.Beta requires floats.
      beta = tf.distributions.Beta(mixup, mixup)
      lam = beta.sample(batch_size)
      ll = tf.expand_dims(tf.expand_dims(tf.expand_dims(lam, -1), -1), -1)
      images = ll * images + (1 - ll) * cshift(images)
      labels = lam * labels + (1 - lam) * cshift(labels)

  return images, labels

def get_data(file):

    f = open(file, 'rb')
    data_dict = pickle.load(f, encoding='latin1')
    f.close()

    X = np.array(data_dict['data'], dtype=float)
    Y = np.eye(NUM_CLASSES)[np.array(data_dict['labels'])]

    X /= 255.0
    X = X.reshape([-1, 3, 32, 32])
    X = X.transpose([0, 2, 3, 1])
    X -= (0.4914, 0.4822, 0.4465)
    X /= (0.2023, 0.1994, 0.2010)
    X = X.reshape(-1, 32*32*3)

    return X,Y

def get_train_batch():
    maybe_download_and_extract()
    f = open(META_FILE, 'rb')
    f.close()
    BATCH_FILE = DATASET_DIRECTORY+CIFAR10_DIRECTORY+'/data_batch_'
    x, y = get_data(BATCH_FILE+'1')
    for i in range(4):
        xx, yy = get_data(BATCH_FILE+str(i+2))
        x = np.concatenate((x ,xx), axis=0)
        y = np.concatenate((y, yy), axis = 0)
    return x, y

def get_test_batch():
    maybe_download_and_extract()
    f = open(META_FILE, 'rb')
    f.close()
    BATCH_FILE = DATASET_DIRECTORY+CIFAR10_DIRECTORY+'/test_batch'
    x, y = get_data(BATCH_FILE)
    return x, y


def print_download_progress(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = count * block_size
    pct_complete = float(progress_size) / total_size
    speed = int(progress_size / (1024 * duration))
    msg = "\r- Download progress: {0:.1%}, {1:} MB, {2:} KB/s, {3:} seconds passed".format(pct_complete, progress_size/(1024*1024), speed, int(duration))
    sys.stdout.write(msg)
    sys.stdout.flush()


def maybe_download_and_extract():
    if not os.path.exists(DATASET_DIRECTORY):
        os.makedirs(DATASET_DIRECTORY)
        filename = URL.split('/')[-1]
        file_path = os.path.join(DATASET_DIRECTORY, filename)
        file_path, _ = urlretrieve(url=URL, filename=file_path, reporthook=print_download_progress)
        print("\nDownload finished. Extracting files.")
        tarfile.open(name=file_path, mode="r:gz").extractall(DATASET_DIRECTORY)
        print("Done.")
