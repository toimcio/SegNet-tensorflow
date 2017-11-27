# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 10:09:45 2017

@author: s161488
"""
import matplotlib.pyplot as plt
import numpy as np


NUM_SAMPLE = []
for i in range(30):
    NUM_SAMPLE.append(2*i+1)
    
acc_tot = np.load("/zhome/1c/2/114196/Documents/SegNet-tensorflow/segnet_bayes_30000/Data/acc_final.npy")
iu_final_mean = np.load("/zhome/1c/2/114196/Documents/SegNet-tensorflow/segnet_bayes_30000/Data/iu_mean_final.npy")




def PLOT_ACC_IU():
    #PLOT THE ACCURACY
    plt.figure(1)
    plt.subplot(211)
    plt.plot(NUM_SAMPLE,acc_tot)
    plt.xlabel('number of samples simulated')
    plt.ylabel('global accuracy')
    plt.title('The relation between number of samples generated and global accuracy')
    plt.subplot(212)
    plt.plot(NUM_SAMPLE,iu_final_mean)
    plt.xlabel('number of samples simulated')
    plt.ylabel('mean MoI')
    plt.title('The relation between number of samples generated and MoI')
    plt.show()

var_tot = np.load("/zhome/1c/2/114196/Documents/SegNet-tensorflow/segnet_bayes_30000/Data/var_tot.npy")
prob_variance = np.load("/zhome/1c/2/114196/Documents/SegNet-tensorflow/segnet_bayes_30000/Data/prob_variance.npy")    
pred_tot = np.load("/zhome/1c/2/114196/Documents/SegNet-tensorflow/segnet_bayes_30000/Data/pred_tot.npy")
image_tot = np.load("/zhome/1c/2/114196/Documents/SegNet-tensorflow/test_image.npy")
label_tot = np.load("/zhome/1c/2/114196/Documents/SegNet-tensorflow/test_label.npy")
def PLOT_VARIANCE():
    plt.figure(2)
    plt.subplot(221)
    plt.imshow(image_tot[-1])
    plt.title('image')
    plt.subplot(222)
    plt.imshow(pred_tot[-1])
    plt.title('predicted label')
    plt.subplot(223)
    plt.imshow(var_tot[-1],cmap = 'Greys')
    plt.title('Dropout Uncertainty for all class')
    plt.subplot(224)
    plt.imshow(prob_variance[:,:,8],cmap = 'Greys')
    plt.title('Dropout uncertainty for Car class')
    plt.show()
    
    
    
    
    


