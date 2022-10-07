import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sys import stderr
def eprint(s):  stderr.write(s)

@tf.function
def get_y(img, label):
    return label

@tf.function
def get_x(img, label):
    return img

    
@tf.function
def normalize(img, label):
    return tf.cast(img, tf.float32) / 255., label

@tf.function
def normalize_i(img, label):
    return tf.cast(img, tf.float32) / 255.

@tf.function
def preprocess_cif(img, label):
    gray = tf.image.rgb_to_grayscale(img)
    return tf.image.resize(gray, [28,28]), label

@tf.function
def psnr(this, other):
    return tf.image.psnr(this, other, max_val=1.0)

@tf.function
def ssim(this, other):
    return tf.image.ssim(this, other, max_val=1.0)

def avgloss(these, others, method):
    n = len(these)
    assert len(these) == len(others)

    return sum(eval(method)(t,o) for t,o in zip(these,others)) / n

def top_q_np(img, q):
    zeros = np.zeros_like(img)
    top_n = int(np.count_nonzero(img) * q)
    idx_top_n = np.argpartition(img.reshape(-1), -top_n)[-top_n:]
    zeros.reshape(-1)[idx_top_n] = img.reshape(-1)[idx_top_n]
    return zeros

@tf.function
def top_q(img, q):
    flat = tf.reshape(img,[-1])
    s = tf.math.count_nonzero(flat)
    k = tf.cast(tf.cast(s, tf.float32) * q, tf.int32)

    t = tf.math.top_k(flat,k)
    idx = tf.expand_dims(t.indices,-1)
    filtered = tf.scatter_nd(idx, t.values, flat.shape)

    return tf.reshape(filtered, img.shape)

def get_shape(ds):
    return tuple(ds['train'].element_spec[0].shape.as_list())

   
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def rescale_grads(xai):
    xai = np.absolute(xai)
    xai_min = np.min(xai, axis=(1, 2), keepdims=True)
    xai_max = np.max(xai, axis=(1, 2), keepdims=True)
    return (xai - xai_min).astype("float32") / (xai_max - xai_min).astype("float32")

def clip_mask(mask, mask_clip=0.01):
    mask_variance = np.mean(mask**2)
    #if mask_variance > mask_clip:
    mask = mask/np.sqrt(mask_variance/mask_clip)
    return mask


def get_ds_size(ds):
    return tf.data.experimental.cardinality(ds).numpy()