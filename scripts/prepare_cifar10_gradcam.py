import os
import tensorflow as tf
import tensorflow_datasets as tfds
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
ds = tfds.load('cifar10', data_dir=DATASETS_DIR, as_supervised=True)

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


### Train victim model
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.5, beta_2=0.999)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

print("Building target model ... ", end='')
path_to_model = MODELS_DIR+'victim/cifar10'

if os.path.exists(path_to_model):
    print("(pretrained model exists!)")
    print("... loading pretrained model ...")
    victim_model = tf.keras.models.load_model(path_to_model, compile=False)
    victim_model.compile(optimizer=optimizer, loss=loss_object, metrics=['accuracy'])
else:
    print("(no existing pretrained model!)")
    victim_model = mnist_cnn(input_shape=(32,32,3), num_of_classes=10)
    victim_model.compile(optimizer=optimizer, loss=loss_object, metrics=['accuracy'])

    print("... training target model ...")
    victim_model.fit(victim_ds_batched,
            epochs=100,
            shuffle=True,
            validation_data=attacker_ds_batched,
            verbose=0,
            callbacks=[callback])
    victim_model.save(path_to_model)
    
    
### Generate XAI
print("Generating model explanations ... ", end='')

if os.path.exists(AUX_DIR+'cifar10_gradcam.hkl'):
    print("(pre-generated xai exists!)")
    print("... loading pre-generated xai ...")
    xai = hkl.load(AUX_DIR+'cifar10_gradcam.hkl')
else:
    print("(no existing pre-generated xai!)")
    print("... generating xai ...")
    
    ds_x = ds.map(get_x, num_parallel_calls=tf.data.AUTOTUNE)
    imgs = np.array(list(ds_x))
    xai = GradCAM(victim_model).generate_gradcam_batched(imgs, return_as_ds=False, batch_size=256)
    hkl.dump(xai, AUX_DIR+'cifar10_gradcam.hkl', mode='w')
    
    
### Train eval model
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.5, beta_2=0.999)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

print("Building eval model ... ", end='')
path_to_model = MODELS_DIR+'eval/cifar10'

if os.path.exists(path_to_model):
    print("(pretrained model exists!)")
    print("... loading pretrained model ...")
    victim_model = tf.keras.models.load_model(path_to_model, compile=False)
    victim_model.compile(optimizer=optimizer, loss=loss_object, metrics=['accuracy'])
else:
    print("(no existing pretrained model!)")
    victim_model = mnist_cnn(input_shape=(32,32,3), num_of_classes=10)
    victim_model.compile(optimizer=optimizer, loss=loss_object, metrics=['accuracy'])

    print("... training eval model ...")
    victim_model.fit(victim_ds_batched,
            epochs=100,
            shuffle=True,
            validation_data=attacker_ds_batched,
            verbose=0,
            callbacks=[callback])
    victim_model.save(path_to_model)