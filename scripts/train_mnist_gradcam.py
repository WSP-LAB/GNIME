import os
import time
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import hickle as hkl

from scripts.models import *
from scripts.plot import *
from scripts.util import * 
from scripts.xai import *

# setup directories
DATASETS_DIR = './datasets/'  
FIGURES_DIR = './figures/' 
MODELS_DIR = './models/'
AUX_DIR = './aux/'

### Prepare dataset
ds = tfds.load('mnist', data_dir=DATASETS_DIR, as_supervised=True)

def preprocess_ds(image, label, resize=(32,32)):
    image = tf.cast(image, tf.float32)
    # Normalize pixel values
    image = image / 255.0
    # Resize image
    image = tf.image.resize(image, resize)
    return image, label

# merge train and test to create custom disjoint segregations
ds = ds['train'].concatenate(ds['test']).map(preprocess_ds)

# create disjoint datasets
split_size = tf.data.experimental.cardinality(ds).numpy()//2
victim_ds = ds.take(split_size)
attacker_ds = ds.skip(split_size)

victim_ds_batched = victim_ds.shuffle(10000).batch(256)
attacker_ds_batched = attacker_ds.shuffle(10000).batch(256)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.5, beta_2=0.999)
victim_model = tf.keras.models.load_model(MODELS_DIR+'victim/mnist', compile=False)
victim_model.compile(optimizer=optimizer, loss=loss_object, metrics=['accuracy'])


ckpt_dir = MODELS_DIR+'def/mnist_gradcam/'
os.makedirs(ckpt_dir, exist_ok=True)
fig_dir = FIGURES_DIR+'def/mnist_gradcam/'
os.makedirs(fig_dir, exist_ok=True)

xai = hkl.load(AUX_DIR+'mnist_gradcam.hkl')
split_size = len(xai)//2
def_xai = xai[:split_size]

_ = ds.map(get_x, num_parallel_calls=tf.data.AUTOTUNE)
def_x = np.array(list(_))[:split_size]
def_pred = victim_model.predict(def_x)


optimizer_noiser = tf.keras.optimizers.Adam(learning_rate=0.0001)
optimizer_inverter = tf.keras.optimizers.Adam(learning_rate=0.0001)

def_model = GNIME(
    victim_model, 
    xai_shape = (16,16,1), 
    x_shape = (32,32,1), 
    num_of_classes = 10,
    mean = np.mean(def_xai),
    stddev = np.std(def_xai)
)

print("Training GNIME ...")

number_of_samples = len(def_xai)
batch_size = 1024
steps_per_epoch = number_of_samples // batch_size
epochs= 500
ckpt_interval = 50

N_loss1es = []
N_loss2es = []
I_losses = []

for epoch in range(1, epochs+1):
    N_loss1_epoch = 0
    N_loss2_epoch = 0
    I_loss_epoch = 0
    
    for step in range(1, steps_per_epoch+1):
        idx = np.random.choice(number_of_samples, batch_size, replace=False)
        xai_subset = def_xai[idx]
        x_subset = def_x[idx]
        pred_subset = def_pred[idx]
        
        (N_loss1, N_loss2), I_loss = def_model.train_step(
            xai_subset,
            x_subset,
            pred_subset,
            optimizer_noiser,
            optimizer_inverter,
            a=100
        )
        N_loss1_epoch += N_loss1
        N_loss2_epoch += N_loss2
        I_loss_epoch += I_loss
        
    N_loss1_epoch /= steps_per_epoch
    N_loss2_epoch /= steps_per_epoch
    I_loss_epoch /= steps_per_epoch
    print(f'\rEpoch {epoch:3d}/{epochs:3d}\tI_loss, (N_loss1, N_loss2) = {I_loss_epoch:.5f} ({N_loss1_epoch:.5f}, {N_loss2_epoch:.5f})', end='')
    
    N_loss1es.append(N_loss1_epoch)
    N_loss2es.append(N_loss2_epoch)
    I_losses.append(I_loss_epoch)
    
    if epoch%ckpt_interval==0:
        def_model.save_weights(ckpt_dir + f'e{epoch}.ckpt')
        save_fig_def(def_model, def_x, def_xai, def_pred, cnt=20, figname=fig_dir+f'e{epoch}.png')
        print('\t(model saved!)')


fig, ax = plt.subplots(2,2)
ax[0,0].set_title('N_loss1es')
ax[0,0].plot(N_loss1es)
ax[1,0].set_title('N_loss2es')
ax[1,0].plot(N_loss2es)
ax[1,1].set_title('I_losses')
ax[1,1].plot(I_losses)

plt.savefig(fig_dir+'defense.png')