import tensorflow as tf
from tensorflow.keras import Model, Sequential, losses
from tensorflow.keras.layers import *
import math

def mnist_cnn(input_shape=(32,32,1), num_of_classes=10):
    x = Input(shape=input_shape)
    _ = Conv2D(128, (3,3), padding='same', activation='relu')(x)
    _ = MaxPooling2D((2,2))(_)
    _ = Conv2D(256, (3,3), padding='same', activation='relu')(_)
    _ = MaxPooling2D((2,2))(_)
    _ = Flatten()(_)
    _ = Dense(512, activation='relu')(_)
    y = Dense(num_of_classes)(_)
    return Model(inputs=[x], outputs=[y])

def celeba_cnn(input_shape=(128,128,3), num_of_classes=1000):
    x = Input(shape=input_shape)
    _ = Conv2D(128, (3,3), padding='same', activation='relu')(x)
    _ = MaxPooling2D((2,2))(_)
    _ = Conv2D(256, (3,3), padding='same', activation='relu')(_)
    _ = MaxPooling2D((2,2))(_)
    _ = Conv2D(512, (3,3), padding='same', activation='relu')(_)
    _ = MaxPooling2D((2,2))(_)
    _ = Flatten()(_)
    _ = Dropout(rate=0.5)(_)
    _ = Dense(512, activation='relu')(_)
    _ = Dropout(rate=0.5)(_)
    y = Dense(num_of_classes)(_)
    return Model(inputs=[x], outputs=[y])



def encoder_block(inputs, channels):
    conv = Conv2D(channels, (3,3), padding='same', activation='relu')(inputs)
    pool = MaxPool2D((2,2))(conv)
    return conv, pool
    
def decoder_block(inputs, channels, conv_output=None, isFinal=False):
    if conv_output is not None:
        inputs = concatenate([inputs, conv_output], axis=3)
    
    if isFinal:
        return Conv2D(channels, (3,3), padding='same')(inputs)
    else:
        conv = Conv2D(channels, (3,3), padding='same', activation='relu')(inputs)
        return UpSampling2D((2,2))(conv)
    
def ExpMI(input_shape=(32,32,1), output_shape=(32,32,1), bottleneck_size=2048, with_conf=True, num_of_classes=10):
    xai = x = Input(shape=input_shape)
    
    # encoding layers
    kernel_size=input_shape[0]
    conv_output_stack = []
    while kernel_size > 2:
        channels = bottleneck_size // kernel_size
        conv, x = encoder_block(x, channels)
        conv_output_stack.append(conv)
        kernel_size /= 2
    
    # bottleneck layer
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    img = Flatten()(xai)
    x = concatenate([x, img], axis=1)
    if with_conf:
        pred = Input(shape=(num_of_classes))
        x = concatenate([x, pred], axis=1)
        
    x = Dense(bottleneck_size//4, activation='relu')(x)
    x = Reshape(target_shape=(1,1,bottleneck_size//4))(x)
    x = UpSampling2D((4,4))(x)
    
    # decoding layers
    while kernel_size < output_shape[0]:
        kernel_size *= 2
        channels = (bottleneck_size//2) // kernel_size
        if conv_output_stack:
            conv_output = conv_output_stack.pop()
        else:
            conv_output = None
            
        if kernel_size==output_shape[0]:
            x = decoder_block(x, output_shape[2], conv_output, isFinal=True)
        else:
            x = decoder_block(x, channels, conv_output, isFinal=False)
        
    if with_conf:
        return Model(inputs=[xai, pred], outputs=[x])
    return Model(inputs=[xai], outputs=[x])



def predmi_mnist(num_of_classes=10):
    conf = Input(shape=(num_of_classes))
    _ = Dense(1024, activation='relu')(conf)
    _ = Reshape(target_shape=(1,1,1024))(_)
    _ = UpSampling2D(4)(_) # -> 4*4*1024
    _ = Conv2D(1024, (3,3), padding='same', activation='relu')(_)      
    _ = UpSampling2D((2,2))(_) # -> 8*8*1024
    _ = Conv2D(512, (3,3), padding='same', activation='relu')(_)
    _ = UpSampling2D((2,2))(_) # -> 16*16*512
    _ = Conv2D(256, (3,3), padding='same', activation='relu')(_)
    _ = UpSampling2D((2,2))(_) # -> 32*32*256
    img = Conv2D(1, (3,3), padding='same')(_)
    return Model(inputs=[conf], outputs=[img])

def predmi_cifar10(num_of_classes=10):
    conf = Input(shape=(num_of_classes))
    _ = Dense(1024, activation='relu')(conf)
    _ = Reshape(target_shape=(1,1,1024))(_)
    _ = UpSampling2D(4)(_) # -> 4*4*1024    
    _ = Conv2D(1024, (3,3), padding='same', activation='relu')(_)      
    _ = UpSampling2D((2,2))(_) # -> 8*8*1024
    _ = Conv2D(512, (3,3), padding='same', activation='relu')(_)
    _ = UpSampling2D((2,2))(_) # -> 16*16*512
    _ = Conv2D(256, (3,3), padding='same', activation='relu')(_)
    _ = UpSampling2D((2,2))(_) # -> 32*32*256
    img = Conv2D(3, (3,3), padding='same')(_)
    return Model(inputs=[conf], outputs=[img])

def predmi_celeba(num_of_classes=1000, output_channels=3):
    conf = Input(shape=(num_of_classes))
    _ = Dense(1024, activation='relu')(conf)
    _ = Reshape(target_shape=(1,1,1024))(_)
    _ = UpSampling2D(4)(_) # -> 4*4*1024
    _ = Conv2D(1024, (3,3), padding='same', activation='relu')(_)      
    _ = UpSampling2D((2,2))(_) # -> 8*8*1024
    _ = Conv2D(512, (3,3), padding='same', activation='relu')(_)
    _ = UpSampling2D((2,2))(_) # -> 16*16*512
    _ = Conv2D(256, (3,3), padding='same', activation='relu')(_)
    _ = UpSampling2D((2,2))(_) # -> 32*32*256
    _ = Conv2D(128, (3,3), padding='same', activation='relu')(_)
    _ = UpSampling2D((2,2))(_) # -> 64*64*128
    _ = Conv2D(64, (3,3), padding='same', activation='relu')(_)
    _ = UpSampling2D((2,2))(_) # -> 128*128*64
    img = Conv2D(output_channels, (3,3), padding='same')(_)
    return Model(inputs=[conf], outputs=[img])


class GNIME(Model):
    def __init__(self, victim_model, xai_shape=(32,32,1), x_shape=(32,32,1),
                 num_of_classes=10, mean=0.01, stddev=0.01):
        super(GNIME, self).__init__()
        self.victim_model = victim_model
        self.mean = mean
        self.stddev = stddev
        
        self.noiser = ExpMI(
            input_shape=xai_shape,
            output_shape=xai_shape,
            with_conf=True,
            num_of_classes=num_of_classes
        )
        self.inverter = ExpMI(
            input_shape=xai_shape,
            output_shape=x_shape,
            with_conf=True,
            num_of_classes=num_of_classes
        )

    @staticmethod
    def standardize(img, mean, std):
        return (img-mean)/std
    
    @staticmethod
    def destandardize(img, mean, std):
        return img*std + mean
        
    def loss(self, xai, x, pred):
        xai = self.standardize(xai, self.mean, self.stddev)
        xai2 = self.noiser((xai, pred))
        N_loss1 = tf.math.reduce_mean((xai2 - xai)**2)
        xai2 = self.destandardize(xai2, self.mean, self.stddev)
        xai2 = tf.clip_by_value(xai2, clip_value_min=0, clip_value_max=1)
        
        x_inv1 = self.inverter((xai, pred))
        x_inv2 = self.inverter((xai2, pred))
        
        N_loss2 = I_loss = tf.math.reduce_mean((x_inv2 - x)**2)
        I_loss += tf.math.reduce_mean((x_inv1 - x)**2)
        
        return N_loss1, N_loss2, I_loss

    def call(self, xai, pred):
        xai = self.standardize(xai, self.mean, self.stddev)
        xai2 = self.noiser((xai, pred))
        xai2 = self.destandardize(xai2, self.mean, self.stddev)
        xai2 = tf.clip_by_value(xai2, clip_value_min=0, clip_value_max=1)
        x_ = self.inverter((xai2, pred))
        return xai2, x_

    @tf.function
    def train_step(self, xai, x, pred, optimizer_noiser, optimizer_inverter, a=1e3):
        with tf.GradientTape() as noiser_tape, tf.GradientTape() as inverter_tape:
            N_loss1, N_loss2, I_loss = self.loss(xai, x, pred)
            N_loss = N_loss1 - N_loss2*a

            grads_inverter_loss = inverter_tape.gradient(
                target=I_loss, sources=self.inverter.trainable_variables
            )
            grads_noiser_loss = noiser_tape.gradient(
                target=N_loss, sources=self.noiser.trainable_variables
            )
            
            optimizer_inverter.apply_gradients(
                zip(grads_inverter_loss, self.inverter.trainable_variables)
            )
            optimizer_noiser.apply_gradients(
                zip(grads_noiser_loss, self.noiser.trainable_variables)
            )

        return (N_loss1, N_loss2), I_loss