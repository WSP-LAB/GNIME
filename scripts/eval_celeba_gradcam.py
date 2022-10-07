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

# use saved version to avoid indexes getting mixed
imgs = hkl.load(AUX_DIR+'celeba_imgs.hkl')
labels = hkl.load(AUX_DIR+'celeba_labels.hkl')
xais = hkl.load(AUX_DIR+'celeba_gradcam.hkl')

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.5, beta_2=0.999)
top5 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5')

victim_model = tf.keras.models.load_model(MODELS_DIR+'victim/celeba', compile=False)
victim_model.compile(optimizer=optimizer, loss=loss_object, metrics=['accuracy', top5])






########## Evaluate ExpMI against GNIME ##########
print('>> ExpMI against GNIME')

ckpt_dir = MODELS_DIR+'def/celeba_gradcam/'
os.makedirs(ckpt_dir, exist_ok=True)
fig_dir = FIGURES_DIR+'def/celeba_gradcam/'
os.makedirs(fig_dir, exist_ok=True)

split_size = len(imgs)//2
def_x = imgs[:split_size]
def_pred = victim_model.predict(def_x)
def_xai = xais[:split_size]

optimizer_noiser = tf.keras.optimizers.Adam(learning_rate=0.0001)
optimizer_inverter = tf.keras.optimizers.Adam(learning_rate=0.0001)

def_model = GNIME(
    victim_model, 
    xai_shape = (32,32,1), 
    x_shape = (128,128,3), 
    num_of_classes = 1000,
    mean = np.mean(def_xai),
    stddev = np.std(def_xai)
)


def_model.load_weights(ckpt_dir + f'e{500}.ckpt')

loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(0.0001)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    
img = imgs[split_size:]
pred = victim_model.predict(img)
xai = xais[split_size:]

# perturb xai using def_model
xai2 = np.empty([0]+list(xai[0].shape), dtype=np.float32)
batch_size=128
for i in tqdm(range(len(xai)//batch_size+1), ncols=100):
    xai_batch = xai[i*batch_size : (i+1)*batch_size]
    pred_batch = pred[i*batch_size : (i+1)*batch_size]
    xai2_batch,_ = def_model(xai_batch,pred_batch)
    xai2 = np.concatenate((xai2, xai2_batch))

inv_ds_x = tf.data.Dataset.from_tensor_slices((xai2, pred))
img_ds = tf.data.Dataset.from_tensor_slices(img)
inv_ds = tf.data.Dataset.zip((inv_ds_x, img_ds))

inv_split_size = get_ds_size(inv_ds) * 0.8
inv_train_ds = inv_ds.take(inv_split_size)
inv_val_ds = inv_ds.skip(inv_split_size)

inv_model = ExpMI(input_shape=(32,32,1), output_shape=(128,128,3), bottleneck_size=2048, with_conf=True, num_of_classes=1000)
inv_model.compile(optimizer=optimizer, loss=loss_object)

print("training ExpMI against GNIME ...")
inv_model.fit(inv_train_ds.batch(128),
        epochs=300,
        shuffle=True,
        validation_data=inv_val_ds.batch(128),
        verbose=0,
        callbacks=[callback])

# evaluate inversion performance
eval_size = int(split_size*0.2)
print('Preparing Dataset to Evaluate Inversion Model')
eval_img = imgs[-eval_size:]
eval_label = labels[-eval_size:]
eval_pred = victim_model.predict(eval_img)
eval_xai = xais[-eval_size:]

print('Fetching and perturbing XAI')
# perturb xai using def_model
eval_xai2 = np.empty([0]+list(eval_xai[0].shape), dtype=np.float32)
batch_size=256
for i in tqdm(range(len(eval_xai)//batch_size+1), ncols=100):
    xai_batch = eval_xai[i*batch_size : (i+1)*batch_size]
    pred_batch = eval_pred[i*batch_size : (i+1)*batch_size]
    xai2_batch,_ = def_model(xai_batch,pred_batch)
    eval_xai2 = np.concatenate((eval_xai2, xai2_batch))

print('Reconstruction via Inversion Model')
#eval_inv = inv_model.predict((eval_xai2, eval_pred))
eval_inv = np.empty([0]+list(eval_img[0].shape), dtype=np.float32)
batch_size=256
for i in tqdm(range(len(eval_xai)//batch_size+1), ncols=100):
    xai2_batch = eval_xai2[i*batch_size : (i+1)*batch_size]
    pred_batch = eval_pred[i*batch_size : (i+1)*batch_size]
    inv_batch = inv_model.predict((xai2_batch, pred_batch))
    eval_inv = np.concatenate((eval_inv, inv_batch))
eval_inv = np.clip(eval_inv, 0, 1)

save_plot_attack(eval_img, eval_xai, eval_xai2, eval_inv, cnt=20, figname=fig_dir+f'eval_GNIME.png')

print('Victim Model Reclassification')
eval_recls = np.argmax(victim_model.predict(eval_inv), axis=1)

print('Calculating Evaluation Metrics')
xai_dist = np.square(eval_xai - eval_xai2).mean(axis=None) 
print(f'\txai pert: {xai_dist:.4f}')
recls_acc = np.sum(eval_recls == eval_label)/len(eval_recls)
print(f'\trecl acc: {recls_acc:.4f}')
loss_mse = np.square(eval_img - eval_inv).mean(axis=None) 
print(f'\tavg mse: {loss_mse:.4f}')
loss_psnr = avgloss(eval_img, eval_inv, "psnr")
print(f'\tavg psnr: {loss_psnr:.4f}')
loss_ssim = avgloss(eval_img, eval_inv, "ssim")
print(f'\tavg ssim: {loss_ssim:.4f}')
# TCA
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.5, beta_2=0.999)
eval_model = tf.keras.models.load_model(MODELS_DIR+'eval/celeba', compile=False)
eval_model.compile(optimizer=optimizer, loss=loss_object)
_ = np.argmax(eval_model.predict(eval_inv), axis=1)
tca = np.sum(_ == eval_label)/len(_)
print(f'\ttca: {tca:.4f}')
# DeepSIM
eval_model2 = Model(inputs=victim_model.input, outputs=victim_model.layers[-3].output)
a = eval_model2.predict(eval_inv)
b = eval_model2.predict(eval_img)
deepsim = np.exp(-np.sqrt(np.sum(np.square(a-b)))/len(a))
print(f'\tDeePSiM: {deepsim:.4f}')






########## Evaluate ExpMI against no defense ##########
print(f'>> ExpMI against no defense')
img = imgs[split_size:]
pred = victim_model.predict(img)
xai = xais[split_size:]

inv_ds_x = tf.data.Dataset.from_tensor_slices((xai, pred))
img_ds = tf.data.Dataset.from_tensor_slices(img)
inv_ds = tf.data.Dataset.zip((inv_ds_x, img_ds))

inv_split_size = get_ds_size(inv_ds) * 0.8
inv_train_ds = inv_ds.take(inv_split_size)
inv_val_ds = inv_ds.skip(inv_split_size)

loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(0.0001)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

inv_model = ExpMI(input_shape=(32,32,1), output_shape=(128,128,3), bottleneck_size=2048, with_conf=True, num_of_classes=1000)
inv_model.compile(optimizer=optimizer, loss=loss_object)

print("training inv model ...")
inv_model.fit(inv_train_ds.batch(128),
        epochs=300,
        shuffle=True,
        validation_data=inv_val_ds.batch(128),
        verbose=0,
        callbacks=[callback])

# evaluate inversion performance
eval_size = int(split_size*0.2)
print('Preparing Dataset to Evaluate Inversion Model')
eval_img = imgs[-eval_size:]
eval_label = labels[-eval_size:]
eval_pred = victim_model.predict(eval_img)
eval_xai = xais[-eval_size:]

print('Reconstruction via Inversion Model')
#eval_inv = inv_model.predict((eval_xai2, eval_pred))
eval_inv = np.empty([0]+list(eval_img[0].shape), dtype=np.float32)
batch_size=256
for i in tqdm(range(len(eval_xai)//batch_size+1), ncols=100):
    xai2_batch = eval_xai[i*batch_size : (i+1)*batch_size]
    pred_batch = eval_pred[i*batch_size : (i+1)*batch_size]
    inv_batch = inv_model.predict((xai2_batch, pred_batch))
    eval_inv = np.concatenate((eval_inv, inv_batch))
eval_inv = np.clip(eval_inv, 0, 1)

save_plot_attack(eval_img, eval_xai, eval_xai2, eval_inv, cnt=20, figname=fig_dir+f'eval_ExpMI.png')

print('Victim Model Reclassification')
eval_recls = np.argmax(victim_model.predict(eval_inv), axis=1)

print('Calculating Evaluation Metrics')
recls_acc = np.sum(eval_recls == eval_label)/len(eval_recls)
print(f'\trecl acc: {recls_acc:.4f}')
loss_mse = np.square(eval_img - eval_inv).mean(axis=None) 
print(f'\tavg mse: {loss_mse:.4f}')
loss_psnr = avgloss(eval_img, eval_inv, "psnr")
print(f'\tavg psnr: {loss_psnr:.4f}')
loss_ssim = avgloss(eval_img, eval_inv, "ssim")
print(f'\tavg ssim: {loss_ssim:.4f}')
# TCA
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.5, beta_2=0.999)
eval_model = tf.keras.models.load_model(MODELS_DIR+'eval/celeba', compile=False)
eval_model.compile(optimizer=optimizer, loss=loss_object)
_ = np.argmax(eval_model.predict(eval_inv), axis=1)
tca = np.sum(_ == eval_label)/len(_)
print(f'\ttca: {tca:.4f}')
# DeepSIM
eval_model2 = Model(inputs=victim_model.input, outputs=victim_model.layers[-3].output)
a = eval_model2.predict(eval_inv)
b = eval_model2.predict(eval_img)
deepsim = np.exp(-np.sqrt(np.sum(np.square(a-b)))/len(a))
print(f'\tDeePSiM: {deepsim:.4f}')











########## Evaluate ExpMI against RND ##########
print('>> ExpMI against RND')
img = imgs[split_size:]
pred = victim_model.predict(img)
xai = xais[split_size:]

# perturb xai with Gaussian noise
noise = np.random.normal(0, math.sqrt(xai_dist), size=xai.shape)
xai2 = np.add(xai, noise)

inv_ds_x = tf.data.Dataset.from_tensor_slices((xai2, pred))
img_ds = tf.data.Dataset.from_tensor_slices(img)
inv_ds = tf.data.Dataset.zip((inv_ds_x, img_ds))

inv_split_size = get_ds_size(inv_ds) * 0.8
inv_train_ds = inv_ds.take(inv_split_size)
inv_val_ds = inv_ds.skip(inv_split_size)

loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(0.0001)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

inv_model = ExpMI(input_shape=(32,32,1), output_shape=(128,128,3), bottleneck_size=2048, with_conf=True, num_of_classes=1000)
inv_model.compile(optimizer=optimizer, loss=loss_object)

print("training inv model ...")
inv_model.fit(inv_train_ds.batch(128),
        epochs=300,
        shuffle=True,
        validation_data=inv_val_ds.batch(128),
        verbose=0,
        callbacks=[callback])


# evaluate inversion performance
eval_size = int(split_size*0.2)
print('Preparing Dataset to Evaluate Inversion Model')
eval_img = imgs[-eval_size:]
eval_label = labels[-eval_size:]
eval_pred = victim_model.predict(eval_img)
eval_xai = xais[-eval_size:]

print('Fetching and perturbing XAI')
noise = np.random.normal(0,math.sqrt(xai_dist),size=eval_xai.shape)
eval_xai2 = np.add(eval_xai, noise)

print('Reconstruction via Inversion Model')
#eval_inv = inv_model.predict((eval_xai2, eval_pred))
eval_inv = np.empty([0]+list(eval_img[0].shape), dtype=np.float32)
batch_size=256
for i in tqdm(range(len(eval_xai)//batch_size+1), ncols=100):
    xai2_batch = eval_xai2[i*batch_size : (i+1)*batch_size]
    pred_batch = eval_pred[i*batch_size : (i+1)*batch_size]
    inv_batch = inv_model.predict((xai2_batch, pred_batch))
    eval_inv = np.concatenate((eval_inv, inv_batch))
eval_inv = np.clip(eval_inv, 0, 1)

save_plot_attack(eval_img, eval_xai, eval_xai2, eval_inv, cnt=20, figname=fig_dir+f'eval_RND.png')

print('Victim Model Reclassification')
eval_recls = np.argmax(victim_model.predict(eval_inv), axis=1)

print('Calculating Evaluation Metrics')
_ = np.square(eval_xai - eval_xai2).mean(axis=None) 
print(f'\txai pert: {_:.4f}')
recls_acc = np.sum(eval_recls == eval_label)/len(eval_recls)
print(f'\trecl acc: {recls_acc:.4f}')
loss_mse = np.square(eval_img - eval_inv).mean(axis=None) 
print(f'\tavg mse: {loss_mse:.4f}')
loss_psnr = avgloss(eval_img, eval_inv, "psnr")
print(f'\tavg psnr: {loss_psnr:.4f}')
loss_ssim = avgloss(eval_img, eval_inv, "ssim")
print(f'\tavg ssim: {loss_ssim:.4f}')
# TCA
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.5, beta_2=0.999)
eval_model = tf.keras.models.load_model(MODELS_DIR+'eval/celeba', compile=False)
eval_model.compile(optimizer=optimizer, loss=loss_object)
_ = np.argmax(eval_model.predict(eval_inv), axis=1)
tca = np.sum(_ == eval_label)/len(_)
print(f'\ttca: {tca:.4f}')
# DeepSIM
eval_model2 = Model(inputs=victim_model.input, outputs=victim_model.layers[-3].output)
a = eval_model2.predict(eval_inv)
b = eval_model2.predict(eval_img)
deepsim = np.exp(-np.sqrt(np.sum(np.square(a-b)))/len(a))
print(f'\tDeePSiM: {deepsim:.4f}')





########## Evaluate PredMI ##########
print(f'>> PredMI')
### prepare dataset to train inversion model
img = imgs[split_size:]
pred = victim_model.predict(img)

inv_ds_x = tf.data.Dataset.from_tensor_slices(pred)
img_ds = tf.data.Dataset.from_tensor_slices(img)
inv_ds = tf.data.Dataset.zip((inv_ds_x, img_ds))

inv_train_ds_size = get_ds_size(inv_ds) * 0.8
inv_train_ds = inv_ds.take(inv_train_ds_size)
inv_val_ds = inv_ds.skip(inv_train_ds_size)

loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.5, beta_2=0.999)

print("building inversion model ... ", end='')
path_to_model = MODELS_DIR+'inv/celeba_predmi'

if os.path.exists(path_to_model):
    print("pretrained model exists!")
    print("loading pretrained model")
    inv_model = tf.keras.models.load_model(path_to_model, compile=False)
    inv_model.compile(optimizer=optimizer, loss=loss_object)
    
else:
    print("no existing pretrained model!")
    # create an instance of the model
    inv_model = predmi_celeba(num_of_classes=1000)
    inv_model.compile(optimizer=optimizer, loss=loss_object)
    
    print("training target model")
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    inv_model.fit(inv_train_ds.batch(128),
            epochs=300,
            shuffle=True,
            validation_data=inv_val_ds.batch(128),
            verbose=0,
            callbacks=[callback])
    inv_model.save(path_to_model)
    
# evaluate inversion performance
eval_size = int(split_size*0.2)
print('Preparing Dataset to Evaluate Inversion Model')
eval_img = imgs[-eval_size:]
eval_label = labels[-eval_size:]
eval_pred = victim_model.predict(eval_img)
eval_xai = xais[-eval_size:]

print('Reconstruction via Inversion Model')
eval_inv = inv_model.predict(eval_pred)
eval_inv = np.clip(eval_inv, 0, 1)

save_plot2(eval_img, eval_inv, cnt=20, figname=fig_dir+f'eval_PredMI.png')

print('Victim Model Reclassification')
eval_recls = np.argmax(victim_model.predict(eval_inv), axis=1)

print('Calculating Evaluation Metrics')
recls_acc = np.sum(eval_recls == eval_label)/len(eval_recls)
print(f'\trecl acc: {recls_acc:.4f}')
loss_mse = np.square(eval_img - eval_inv).mean(axis=None) 
print(f'\tavg mse: {loss_mse:.4f}')
loss_psnr = avgloss(eval_img, eval_inv, "psnr")
print(f'\tavg psnr: {loss_psnr:.4f}')
loss_ssim = avgloss(eval_img, eval_inv, "ssim")
print(f'\tavg ssim: {loss_ssim:.4f}')
# TCA
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.5, beta_2=0.999)
eval_model = tf.keras.models.load_model(MODELS_DIR+'eval/celeba', compile=False)
eval_model.compile(optimizer=optimizer, loss=loss_object)
_ = np.argmax(eval_model.predict(eval_inv), axis=1)
tca = np.sum(_ == eval_label)/len(_)
print(f'\ttca: {tca:.4f}')
# DeepSIM
eval_model2 = Model(inputs=victim_model.input, outputs=victim_model.layers[-3].output)
a = eval_model2.predict(eval_inv)
b = eval_model2.predict(eval_img)
deepsim = np.exp(-np.sqrt(np.sum(np.square(a-b)))/len(a))
print(f'\tDeePSiM: {deepsim:.4f}')