import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



def save_plot3(x, xai, inv, cnt=5, offset=0, figname='./plot.png', titles=['x','xai','reconstructed x']):
    scale=2
    plt.gray()
    fig, ax = plt.subplots(cnt,3, figsize=(scale*3,scale*cnt))
    
    for i in range(cnt):
        x_i = x[i+offset]
        xai_i = xai[i+offset]
        inv_i = inv[i+offset]
        
        if i==0:
            ax[i,0].set_title(titles[0])
            ax[i,1].set_title(titles[1])
            ax[i,2].set_title(titles[2])
        
        ax[i,0].imshow(x_i)
        ax[i,0].axis('off')
        ax[i,1].imshow(xai_i)
        ax[i,1].axis('off')
        ax[i,2].imshow(np.clip(inv_i,0,1))
        ax[i,2].axis('off')
        
        ax[i,2].set_title(f'{np.square(x_i - inv_i).mean(axis=None):.5f}')

    plt.savefig(figname)
    

def save_plot2(x, inv, cnt=5, offset=0, figname='./plot.png', titles=['x','reconstructed x']):
    scale=2
    plt.gray()
    fig, ax = plt.subplots(cnt,2, figsize=(scale*2,scale*cnt))
    
    for i in range(cnt):
        x_i = x[i+offset]
        inv_i = inv[i+offset]
        
        if i==0:
            ax[i,0].set_title(titles[0])
            ax[i,1].set_title(titles[1])
        
        ax[i,0].imshow(x_i)
        ax[i,0].axis('off')
        ax[i,1].imshow(np.clip(inv_i,0,1))
        ax[i,1].axis('off')
        
        ax[i,1].set_title(f'{np.square(x_i - inv_i).mean(axis=None):.5f}')

    plt.savefig(figname)
    
    
def save_plot4(x, xai1, xai2, xai3, cnt=5, offset=0, figname='./plot.png', titles=['x','xai1','xai2','xai3']):
    scale=2
    plt.gray()
    fig, ax = plt.subplots(cnt,4, figsize=(scale*3,scale*cnt))
    
    for i in range(cnt):
        x_i = x[i+offset]
        xai1_i = xai1[i+offset]
        xai2_i = xai2[i+offset]
        xai3_i = xai3[i+offset]
        
        if i==0:
            ax[i,0].set_title(titles[0])
            ax[i,1].set_title(titles[1])
            ax[i,2].set_title(titles[2])
            ax[i,3].set_title(titles[3])
        
        ax[i,0].imshow(x_i)
        ax[i,0].axis('off')
        ax[i,1].imshow(xai1_i)
        ax[i,1].axis('off')
        ax[i,2].imshow(np.clip(xai2_i,0,1))
        ax[i,2].axis('off')
        ax[i,3].imshow(np.clip(xai3_i,0,1))
        ax[i,3].axis('off')

    plt.savefig(figname)


def save_fig_def(def_model, x, xai, pred, cnt=5, figname='./newfigures/default.png', grayscale=False):
    mse = tf.keras.losses.MeanSquaredError()
    if grayscale:
        plt.gray()
    scale=2
    fig, ax = plt.subplots(cnt,4, figsize=(scale*4,scale*cnt))
    for i in range(cnt):
        x_i = x[i]
        xai_i = xai[i]
        xai_i_ = np.expand_dims(xai_i, axis=0)
        pred_i = pred[i]
        pred_i = np.expand_dims(pred_i, axis=0)
        xai_masked_i, inv_xai_masked_i = def_model(xai_i_, pred_i)
        xai_masked_i = tf.clip_by_value(xai_masked_i, clip_value_min=0, clip_value_max=1)
        inv_xai_masked_i = tf.clip_by_value(inv_xai_masked_i, clip_value_min=0, clip_value_max=1)
        
        if i==0:
            ax[i,0].set_title('x')
            ax[i,1].set_title('xai')
            ax[i,2].set_title('masked xai')
            ax[i,3].set_title('masked xai inv')
        
        ax[i,0].imshow(x_i)
        ax[i,0].axis('off')
        ax[i,1].imshow(xai_i)
        ax[i,1].axis('off')
        
        ax[i,2].set_title(f'{mse(xai_i,xai_masked_i):.4f}')
        ax[i,2].imshow(xai_masked_i[-1,...])
        ax[i,2].axis('off')
        ax[i,3].set_title(f'{mse(x_i,inv_xai_masked_i):.4f}')
        ax[i,3].imshow(inv_xai_masked_i[-1,...])
        ax[i,3].axis('off')

    plt.savefig(figname)
    

def save_plot_attack(x, xai1, xai2, inv, cnt=5, offset=0, figname='./plot.png', titles=['x','xai','xai_','inv']):
    scale=2
    plt.gray()
    fig, ax = plt.subplots(cnt,4, figsize=(scale*3,scale*cnt))
    
    for i in range(cnt):
        x_i = x[i+offset]
        xai1_i = xai1[i+offset]
        xai2_i = xai2[i+offset]
        inv_i = inv[i+offset]
        
        if i==0:
            ax[i,0].set_title(titles[0])
            ax[i,1].set_title(titles[1])
            ax[i,2].set_title(titles[2])
            ax[i,3].set_title(titles[3])
        
        ax[i,0].imshow(x_i)
        ax[i,0].axis('off')
        ax[i,1].imshow(xai1_i)
        ax[i,1].axis('off')
        
        xai2_i = np.clip(xai2_i, 0, 1)
        ax[i,2].imshow(xai2_i)
        ax[i,2].axis('off')
        ax[i,2].set_title(f'{np.square(xai1_i - xai2_i).mean(axis=None):.4f}')
        
        inv_i = np.clip(inv_i, 0, 1)
        ax[i,3].imshow(inv_i)
        ax[i,3].axis('off')
        ax[i,3].set_title(f'{np.square(x_i - inv_i).mean(axis=None):.5f}')

    plt.savefig(figname)