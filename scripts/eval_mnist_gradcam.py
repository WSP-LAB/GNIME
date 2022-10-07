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


########## Evaluate ExpMI against GNIME ##########
print('>> ExpMI against GNIME')

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

def_model.load_weights(ckpt_dir + f'e{500}.ckpt')

img_ds = attacker_ds.map(get_x, num_parallel_calls=tf.data.AUTOTUNE)
img = np.array(list(img_ds)) 
pred = victim_model.predict(img)
xai = hkl.load(AUX_DIR+'mnist_gradcam.hkl')[split_size:]

# perturb xai using def_model
xai2 = np.empty([0]+list(xai[0].shape), dtype=np.float32)
batch_size=1024
for i in tqdm(range(len(xai)//batch_size+1), ncols=100):
    xai_batch = xai[i*batch_size : (i+1)*batch_size]
    pred_batch = pred[i*batch_size : (i+1)*batch_size]
    xai2_batch,_ = def_model(xai_batch,pred_batch)

    xai2 = np.concatenate((xai2, xai2_batch))

inv_ds_x = tf.data.Dataset.from_tensor_slices((xai2, pred))
inv_ds = tf.data.Dataset.zip((inv_ds_x, img_ds))
inv_split_size = get_ds_size(inv_ds) * 0.8
inv_train_ds = inv_ds.take(inv_split_size)
inv_val_ds = inv_ds.skip(inv_split_size)

loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(0.0001)

inv_model = ExpMI(input_shape=(16,16,1), output_shape=(32,32,1), bottleneck_size=2048, with_conf=True, num_of_classes=10)
inv_model.compile(optimizer=optimizer, loss=loss_object)

print("training inv model ...")
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
inv_model.fit(inv_train_ds.batch(128),
        epochs=300,
        shuffle=True,
        validation_data=inv_val_ds.batch(128),
        verbose=0,
        callbacks=[callback])


### prepare dataset to evaluate inversion model
print('Preparing Dataset to Evaluate Inversion Model')
eval_img_ds = attacker_ds.map(get_x, num_parallel_calls=tf.data.AUTOTUNE)
eval_img = np.array(list(eval_img_ds))[-7000:]
eval_label_ds = attacker_ds.map(get_y, num_parallel_calls=tf.data.AUTOTUNE)
eval_label = np.array(list(eval_label_ds))[-7000:]
eval_pred = victim_model.predict(eval_img)


print('Fetching and perturbing XAI')
eval_xai = hkl.load(AUX_DIR+'mnist_gradcam.hkl')[-7000:]
# perturb xai using def_model
eval_xai2 = np.empty([0]+list(eval_xai[0].shape), dtype=np.float32)
batch_size=1024
for i in tqdm(range(len(eval_xai)//batch_size+1), ncols=100):
    xai_batch = eval_xai[i*batch_size : (i+1)*batch_size]
    pred_batch = eval_pred[i*batch_size : (i+1)*batch_size]
    xai2_batch,_ = def_model(xai_batch,pred_batch)
    eval_xai2 = np.concatenate((eval_xai2, xai2_batch))

print('Reconstruction via Inversion Model')
#eval_inv = inv_model.predict((eval_xai2, eval_pred))
eval_inv = np.empty([0]+list(eval_img[0].shape), dtype=np.float32)
batch_size=1024
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
eval_model = tf.keras.models.load_model(MODELS_DIR+'eval/mnist', compile=False)
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
print('>> ExpMI against no defense')

### prepare dataset to train inversion model
attacker_ds_x = attacker_ds.map(get_x, num_parallel_calls=tf.data.AUTOTUNE)
imgs = np.array(list(attacker_ds_x))
xai = hkl.load(AUX_DIR+'mnist_gradcam.hkl')[split_size:]

preds = victim_model.predict(imgs)
inv_ds_x = tf.data.Dataset.from_tensor_slices((xai, preds))
inv_ds = tf.data.Dataset.zip((inv_ds_x, attacker_ds_x))

inv_train_ds_size = get_ds_size(inv_ds) * 0.8
inv_train_ds = inv_ds.take(inv_train_ds_size)
inv_val_ds = inv_ds.skip(inv_train_ds_size)

##### training inversion model #####
loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.5, beta_2=0.999)

model_path = MODELS_DIR+'inv/mnist_expmi_gradcam'

if os.path.exists(model_path):
    print("pretrained model exists!")
    print("loading pretrained model ...")
    inv_model = tf.keras.models.load_model(model_path, compile=False)
    inv_model.compile(optimizer=optimizer, loss=loss_object)
    
else:
    print("no existing pretrained model!")
    # create an instance of the model
    inv_model = ExpMI(input_shape=(16,16,1), output_shape=(32,32,1), bottleneck_size=2048, with_conf=True, num_of_classes=10)
    inv_model.compile(optimizer=optimizer, loss=loss_object)
    
    print("training inv model ...")
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    inv_model.fit(inv_train_ds.batch(256),
            epochs=300,
            shuffle=True,
            validation_data=inv_val_ds.batch(256),
            verbose=0,
            callbacks=[callback])
    inv_model.save(model_path)


### prepare dataset to evaluate inversion model
print('Preparing Dataset to Evaluate Inversion Model')
eval_img_ds = attacker_ds.map(get_x, num_parallel_calls=tf.data.AUTOTUNE)
eval_img = np.array(list(eval_img_ds))[-7000:]
eval_label_ds = attacker_ds.map(get_y, num_parallel_calls=tf.data.AUTOTUNE)
eval_label = np.array(list(eval_label_ds))[-7000:]
eval_pred = victim_model.predict(eval_img)
eval_xai = hkl.load(AUX_DIR+'mnist_gradcam.hkl')[-7000:]

print('Reconstruction via Inversion Model')
#eval_inv = inv_model.predict((eval_xai2, eval_pred))
eval_inv = np.empty([0]+list(eval_img[0].shape), dtype=np.float32)
batch_size=1024
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
eval_model = tf.keras.models.load_model(MODELS_DIR+'eval/mnist', compile=False)
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
### prepare dataset to train inversion model
attacker_ds_x = attacker_ds.map(get_x, num_parallel_calls=tf.data.AUTOTUNE)
imgs = np.array(list(attacker_ds_x))
xai = hkl.load(AUX_DIR+'mnist_gradcam.hkl')[split_size:]

# perturb xai with Gaussian noise
noise = np.random.normal(0, math.sqrt(xai_dist), size=xai.shape)
xai2 = np.add(xai, noise)
                                           
preds = victim_model.predict(imgs)
inv_ds_x = tf.data.Dataset.from_tensor_slices((xai2, preds))
inv_ds = tf.data.Dataset.zip((inv_ds_x, attacker_ds_x))

inv_train_ds_size = get_ds_size(inv_ds) * 0.8
inv_train_ds = inv_ds.take(inv_train_ds_size)
inv_val_ds = inv_ds.skip(inv_train_ds_size)

##### training inversion model #####
loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.5, beta_2=0.999)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)                    
# create an instance of the model
inv_model = ExpMI(input_shape=(16,16,1), output_shape=(32,32,1), bottleneck_size=2048, with_conf=True, num_of_classes=10)
inv_model.compile(optimizer=optimizer, loss=loss_object)

print("training inv model ...")
inv_model.fit(inv_train_ds.batch(256),
        epochs=300,
        shuffle=True,
        validation_data=inv_val_ds.batch(256),
        verbose=0,
        callbacks=[callback])

### prepare dataset to evaluate inversion model
print('Preparing Dataset to Evaluate Inversion Model')
eval_img_ds = attacker_ds.map(get_x, num_parallel_calls=tf.data.AUTOTUNE)
eval_img = np.array(list(eval_img_ds))[-7000:]
eval_label_ds = attacker_ds.map(get_y, num_parallel_calls=tf.data.AUTOTUNE)
eval_label = np.array(list(eval_label_ds))[-7000:]
eval_pred = victim_model.predict(eval_img)
eval_xai = hkl.load(AUX_DIR+'mnist_gradcam.hkl')[-7000:]

noise = np.random.normal(0,math.sqrt(xai_dist),size=eval_xai.shape)
eval_xai2 = np.add(eval_xai, noise)
                                           
print('Reconstruction via Inversion Model')
#eval_inv = inv_model.predict((eval_xai2, eval_pred))
eval_inv = np.empty([0]+list(eval_img[0].shape), dtype=np.float32)
batch_size=1024
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
eval_model = tf.keras.models.load_model(MODELS_DIR+'eval/mnist', compile=False)
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
print('>> PredMI')
### prepare dataset to train inversion model
attacker_ds_x = attacker_ds.map(get_x, num_parallel_calls=tf.data.AUTOTUNE)
imgs = np.array(list(attacker_ds_x))
preds = victim_model.predict(imgs)
inv_ds_x = tf.data.Dataset.from_tensor_slices(preds)
inv_ds = tf.data.Dataset.zip((inv_ds_x, attacker_ds_x))

inv_train_ds_size = get_ds_size(inv_ds) * 0.8
inv_train_ds = inv_ds.take(inv_train_ds_size)
inv_val_ds = inv_ds.skip(inv_train_ds_size)
                                           
##### training inversion model #####
loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.5, beta_2=0.999)

print("building PredMI model ... ", end='')
path_to_model = MODELS_DIR+'inv/mnist_predmi'

if os.path.exists(path_to_model):
    print("(pretrained model exists!)")
    print("... loading pretrained model ...")
    inv_model = tf.keras.models.load_model(path_to_model, compile=False)
    inv_model.compile(optimizer=optimizer, loss=loss_object)
    
else:
    print("(no existing pretrained model!)")
    # create an instance of the model
    inv_model = predmi_mnist(num_of_classes=10)
    inv_model.compile(optimizer=optimizer, loss=loss_object)
    
    print("training target model ...")
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    inv_model.fit(inv_train_ds.batch(256),
            epochs=300,
            shuffle=True,
            validation_data=inv_val_ds.batch(256),
            verbose=0,
            callbacks=[callback])
    inv_model.save(path_to_model)
                                           
### prepare dataset to evaluate inversion model
print('Preparing Dataset to Evaluate Inversion Model')
eval_img_ds = attacker_ds.map(get_x, num_parallel_calls=tf.data.AUTOTUNE)
eval_img = np.array(list(eval_img_ds))[-7000:]
eval_label_ds = attacker_ds.map(get_y, num_parallel_calls=tf.data.AUTOTUNE)
eval_label = np.array(list(eval_label_ds))[-7000:]

print('Reconstruction via Inversion Model')
eval_pred = victim_model.predict(eval_img)
eval_inv = inv_model.predict(eval_pred)

save_plot2(eval_img, eval_inv, cnt=20, figname=fig_dir+'eval_PredMI.png')

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
eval_model = tf.keras.models.load_model(MODELS_DIR+'eval/mnist', compile=False)
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