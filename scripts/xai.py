import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
from tqdm import tqdm
import tensorflow.keras.backend as K
from tensorflow.python.ops import gen_nn_ops


class GradCAM:
    def __init__(self, model, target_layer_name=None):
        self.model = model
        if target_layer_name is None:
            target_layer_name = self.find_target_layer_name()
        self.target_layer = self.model.get_layer(target_layer_name)
        self.target_layer_output_shape = self.target_layer.output.shape.as_list()
        self.heatmap_model = tf.keras.Model(
                inputs = [self.model.inputs],
                outputs = [self.target_layer.output, self.model.outputs]
        )
        
    def find_target_layer_name(self):
        for layer in reversed(self.model.layers):
            if type(layer) == Conv2D:
                return layer.name
        raise ValueError("Could not find Conv2D layer. Cannot apply GradCAM.")

    def generate_gradcam(self, imgs):
        imgs = tf.convert_to_tensor(imgs)
        with tf.GradientTape() as tape:
            conv_output, predictions = self.heatmap_model(imgs)
            prediction = tf.reduce_max(predictions[0], axis=1)
            grads = tape.gradient(prediction, conv_output)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
        return heatmap
 
    def generate_gradcam_batched(self, imgs_ds, batch_size=256, return_as_ds=True):
        imgs = np.array(list(imgs_ds))
        
        heatmaps = np.empty([0]+self.target_layer_output_shape[1:3], dtype=np.float32)
        
        for i in tqdm(range(len(imgs)//batch_size+1), desc='gradcam', ncols=100):
            imgs_batch = imgs[i*batch_size : (i+1)*batch_size]
            heatmaps_batch = self.generate_gradcam(imgs_batch)
            heatmaps = np.concatenate((heatmaps, heatmaps_batch))
            
        heatmaps = np.absolute(heatmaps)
        heatmaps_min = np.min(heatmaps, axis=(1, 2), keepdims=True)
        heatmaps_max = np.max(heatmaps, axis=(1, 2), keepdims=True)
        heatmaps = (heatmaps - heatmaps_min) / (heatmaps_max - heatmaps_min)
        
        if return_as_ds:
            heatmaps = np.expand_dims(heatmaps, -1)
            return tf.data.Dataset.from_tensor_slices(heatmaps)
        else:
            return np.expand_dims(heatmaps, -1)

def generate_grad(model, img_ds, return_as_ds=True):
    imgs = tf.convert_to_tensor(list(img_ds))
    
    with tf.GradientTape() as tape:
        tape.watch(imgs)
        pred = model(imgs)
        pred_trim = [pred[i][np.argmax(pred[i])] for i in range(len(pred))]
        pred_trim = tf.convert_to_tensor(pred_trim)
        gradients = np.array(tape.gradient(pred_trim, imgs))
    
    gradients = np.absolute(gradients)
    gradients_min = np.min(gradients, axis=(1, 2), keepdims=True)
    gradients_max = np.max(gradients, axis=(1, 2), keepdims=True)
    gradients = (gradients - gradients_min) / (gradients_max - gradients_min)
    
    if return_as_ds:
        return tf.data.Dataset.from_tensor_slices(gradients)
    else:
        return gradients

def generate_grad_batched(model, img_ds, batch_size=256, return_as_ds=True):
    imgs = np.array(list(img_ds))

    gradients=np.empty([0]+list(imgs[0].shape), dtype=np.float32)
    for i in tqdm(range(len(imgs)//batch_size+1), desc='grad', ncols=100):
        imgs_batch = imgs[i*batch_size : (i+1)*batch_size]
        imgs_batch = tf.convert_to_tensor(imgs_batch)
        with tf.GradientTape() as tape:
            tape.watch(imgs_batch)
            pred = model(imgs_batch)
            pred_trim = [pred[i][np.argmax(pred[i])] for i in range(len(pred))]
            pred_trim = tf.convert_to_tensor(pred_trim)
            gradient_batch = np.array(tape.gradient(pred_trim, imgs_batch))
        gradients=np.concatenate((gradients, gradient_batch))

    gradients = np.absolute(gradients)
    gradients_min = np.min(gradients, axis=(1, 2), keepdims=True)
    gradients_max = np.max(gradients, axis=(1, 2), keepdims=True)
    gradients = (gradients - gradients_min) / (gradients_max - gradients_min)
    
    if return_as_ds:
        return tf.data.Dataset.from_tensor_slices(gradients)
    else:
        return gradients



class LRP(object):
    def __init__(self, model, eps=1e-9, isRGB=False, pool_type='avg'):
        self.model = model
        self.eps = eps
        self.isRGB = isRGB
        self.pooling_type = pool_type # 'avg' or 'max'
            
        self.weights = {weight.name.split('/')[0]: weight for weight in self.model.trainable_weights
                        if 'bias' not in weight.name}

        # Extract activation layers (in reverse order)
        self.activations = [layer.output for layer in self.model.layers][::-1]

        # Extract the model's layers name (in reverse order)
        self.layer_names = [layer.name for layer in self.model.layers][::-1]

        # Build relevance graph
        self.relevance = self.relevance_propagation()

    def generate_batched(self, imgs, batch_size=256, return_as_ds=True):        
        f = K.function(inputs=self.model.input, outputs=self.relevance)
        
        relevance_scores = np.empty([0]+list(imgs[0].shape), dtype=np.float32)
        for i in tqdm(range(len(imgs)//batch_size+1), desc='lrp', ncols=100):
            imgs_batch = imgs[i*batch_size : (i+1)*batch_size]
            relevance_scores_batch = f(inputs=imgs_batch)
            relevance_scores = np.concatenate((relevance_scores, relevance_scores_batch))
        
        relevance_scores = self.postprocess(relevance_scores)
        
        if return_as_ds:
            return tf.data.Dataset.from_tensor_slices(relevance_scores)
        else:
            return relevance_scores
    

    def relevance_propagation(self):
        relevance = self.model.output
        for i, layer_name in enumerate(self.layer_names):
            if 'prediction' in layer_name:
                relevance = self.relprop_dense(self.activations[i+1], self.weights[layer_name], relevance)
            elif 'dense' in layer_name:
                relevance = self.relprop_dense(self.activations[i+1], self.weights[layer_name], relevance)
            elif 'flatten' in layer_name:
                relevance = self.relprop_flatten(self.activations[i+1], relevance)
            elif 'pool' in layer_name:
                relevance = self.relprop_pool(self.activations[i+1], relevance)
            elif 'conv' in layer_name:
                relevance = self.relprop_conv(self.activations[i+1], self.weights[layer_name], relevance, layer_name)
            elif 'input' in layer_name:
                pass
            elif 'dropout' in layer_name:
                pass
            else:
                raise Exception("Error: layer type not recognized.")
        return relevance

    def relprop_dense(self, x, w, r):
        w_pos = tf.maximum(w, 0.0)
        z = tf.matmul(x, w_pos) + self.eps
        s = r / z
        c = tf.matmul(s, tf.transpose(w_pos))
        return c * x
        
    def relprop_flatten(self, x, r):
        return tf.reshape(r, tf.shape(x))

    def relprop_pool(self, x, r, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME'):
        if self.pooling_type == "avg":
            z = tf.nn.avg_pool(x, ksize, strides, padding) + self.eps
            s = r / z
            c = gen_nn_ops.avg_pool_grad(tf.shape(x), s, ksize, strides, padding)
        elif self.pooling_type == "max":
            z = tf.nn.max_pool(x, ksize, strides, padding) + self.eps
            s = r / z
            c = gen_nn_ops.max_pool_grad_v2(x, z, s, ksize, strides, padding)
        else:
            raise Exception("Error: no such unpooling operation implemented.")
        return c * x

    def relprop_conv(self, x, w, r, name, strides=(1, 1, 1, 1), padding='SAME'):
        if name == 'conv2d':
            x = tf.ones_like(x)     # only for input

        w_pos = tf.maximum(w, 0.0)
        z = tf.nn.conv2d(x, w_pos, strides, padding) + self.eps
        s = r / z
        c = tf.compat.v1.nn.conv2d_backprop_input(tf.shape(x), w_pos, s, strides, padding)
        return c * x
        
    @staticmethod
    def rescale(x, eps):
        x_min = np.min(x, axis=(1, 2), keepdims=True)
        x_max = np.max(x, axis=(1, 2), keepdims=True)
        return (x - x_min).astype("float32") / ((x_max - x_min).astype("float32") + eps)

    def postprocess(self, x):
        x = np.absolute(x)
        x = self.rescale(x, self.eps)
        return x
    
    
