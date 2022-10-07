import os
import tensorflow as tf
import hickle as hkl

from scripts.models import *
from scripts.plot import *
from scripts.util import * 
from scripts.xai import *

# configure directories
DATASETS_DIR = './datasets/'  
FIGURES_DIR = './figures/' 
MODELS_DIR = './models/'
AUX_DIR = './aux/'

### Prepare dataset
print("Preparing dataset ... ", end='')
if os.path.exists(AUX_DIR+'celeba_imgs.hkl') and os.path.exists(AUX_DIR+'celeba_imgs.hkl'):
    print("(pre-generated dataset exists!)")
    print("... loading pre-generated dataset ...")
    imgs = hkl.load(AUX_DIR+'celeba_imgs.hkl')
    labels = hkl.load(AUX_DIR+'celeba_labels.hkl')
else:
    print("(no existing pre-generated dataset!)")
    print("... preparing dataset ...")
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATASETS_DIR + 'celeba_cropped_1000',
        labels='inferred',
        label_mode='int',
        color_mode='rgb',
        batch_size=1,
        image_size=(128,128),
        shuffle=False,
    )
    # save in current order to prevent shuffle
    ds = ds.unbatch()
    imgs = ds.map(normalize_i, num_parallel_calls=tf.data.AUTOTUNE)
    labels = ds.map(get_y, num_parallel_calls=tf.data.AUTOTUNE)
    imgs = np.array(list(imgs))
    labels = np.array(list(labels))
    imgs,labels = unison_shuffled_copies(imgs,labels)
    hkl.dump(imgs, AUX_DIR+'celeba_imgs.hkl', mode='w')
    hkl.dump(labels, AUX_DIR+'celeba_labels.hkl', mode='w')


### Train victim model
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.5, beta_2=0.999)
top5 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5')

print("Building target model ... ", end='')
path_to_model = MODELS_DIR+'victim/celeba'

if os.path.exists(path_to_model):
    print("(pretrained model exists!)")
    print("... loading pretrained model ...")
    victim_model = tf.keras.models.load_model(path_to_model, compile=False)
    victim_model.compile(optimizer=optimizer, loss=loss_object, metrics=['accuracy', top5])
else:
    print("(no existing pretrained model!)")
    victim_model = celeba_cnn(input_shape=(128,128,3), num_of_classes=1000)
    victim_model.compile(optimizer=optimizer, loss=loss_object, metrics=['accuracy', top5])
    
    cnt = len(imgs)
    train_size = cnt//2
    
    ds = tf.data.Dataset.from_tensor_slices((imgs,labels))
    victim_ds = ds.take(train_size)
    attacker_ds = ds.skip(train_size)
    victim_ds_batched = victim_ds.shuffle(1000).batch(128)
    attacker_ds_batched = attacker_ds.shuffle(1000).batch(128)
    
    print("... training target model ...")
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1)
    victim_model.fit(victim_ds_batched,
            epochs=100,
            shuffle=True,
            validation_data=attacker_ds_batched,
            verbose=0,
            callbacks=[callback])
    victim_model.save(path_to_model)
    

### Generate XAI
print("Generating model explanations ... ", end='')

if os.path.exists(AUX_DIR+'celeba_grad.hkl'):
    print("(pre-generated xai exists!)")
    print("... loading pre-generated xai ...")
    grads = hkl.load(AUX_DIR+'celeba_grad.hkl')
else:
    print("(no existing pre-generated xai!)")
    print("... generating xai ...")
    xai = generate_grad_batched(victim_model, imgs, batch_size=64, return_as_ds=False)
    hkl.dump(xai, AUX_DIR+'celeba_grad.hkl', mode='w')
    
    
### Train eval model
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.5, beta_2=0.999)
top5 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5')
callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1)

print("Building eval model ... ", end='')
path_to_model = MODELS_DIR+'eval/celeba'

if os.path.exists(path_to_model):
    print("(pretrained model exists!)")
    print("... loading pretrained model ...")
    victim_model = tf.keras.models.load_model(path_to_model, compile=False)
    victim_model.compile(optimizer=optimizer, loss=loss_object, metrics=['accuracy', top5])
else:
    print("(no existing pretrained model!)")
    victim_model = celeba_cnn(input_shape=(128,128,3), num_of_classes=1000)
    victim_model.compile(optimizer=optimizer, loss=loss_object, metrics=['accuracy', top5])
    
    cnt = len(imgs)
    train_size = cnt//2
    
    ds = tf.data.Dataset.from_tensor_slices((imgs,labels))
    victim_ds = ds.take(train_size)
    attacker_ds = ds.skip(train_size)
    victim_ds_batched = victim_ds.shuffle(1000).batch(128)
    attacker_ds_batched = attacker_ds.shuffle(1000).batch(128)
    
    print("... training eval model ...")
    victim_model.fit(victim_ds_batched,
            epochs=100,
            shuffle=True,
            validation_data=attacker_ds_batched,
            verbose=0,
            callbacks=[callback])
    victim_model.save(path_to_model)