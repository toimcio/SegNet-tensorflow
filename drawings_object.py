import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

def writeImage(image):
    """ store label data to colored image """
    Sky = [128,128,128]
    Building = [128,0,0]
    Pole = [192,192,128]
    Road_marking = [255,69,0]
    Road = [128,64,128]
    Pavement = [60,40,222]
    Tree = [128,128,0]
    SignSymbol = [192,128,128]
    Fence = [64,64,128]
    Car = [64,0,128]
    Pedestrian = [64,64,0]
    Bicyclist = [0,128,192]
    Unlabelled = [0,0,0]
    r = image.copy()
    g = image.copy()
    b = image.copy()
    label_colours = np.array([Sky, Building, Pole, Road_marking, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])
    for l in range(0,12):
        r[image==l] = label_colours[l,0]
        g[image==l] = label_colours[l,1]
        b[image==l] = label_colours[l,2]
    rgb = np.zeros((image.shape[0], image.shape[1], 3))
    rgb[:,:,0] = r/1.0
    rgb[:,:,1] = g/1.0
    rgb[:,:,2] = b/1.0
    im = Image.fromarray(np.uint8(rgb))
    plt.imshow(im)
    
def display_color_legend():
    
    Sky          = np.array([128,128,128])/256
    Building     = np.array([128,0,0])/256
    Pole         = np.array([192,192,128])/256
    Road         = np.array([255,69,0])/256
    Pavement     = np.array([128,64,128])/256
    Tree         = np.array([60,40,222])/256
    SignSymbol   = np.array([128,128,0])/256
    Fence        = np.array([192,128,128])/256
    Car          = np.array([64,64,128])/256
    Pedestrian   = np.array([64,0,128])/256
    Bicyclist    = np.array([64,64,0])/256
    Unlabelled   = np.array([0,128,192])/256

    patches = [mpatches.Patch(color=Sky, label='Sky'), mpatches.Patch(color=Building, label='Building'), 
               mpatches.Patch(color=Pole, label='Pole'), mpatches.Patch(color=Road, label='Road'), 
               mpatches.Patch(color=Pavement, label='Pavement'), mpatches.Patch(color=Tree, label='Tree'),
               mpatches.Patch(color=SignSymbol, label='SignSymbol'), mpatches.Patch(color=Fence, label='Fence'),
               mpatches.Patch(color=Car, label='Car'), mpatches.Patch(color=Pedestrian, label='Pedestrian'),
               mpatches.Patch(color=Bicyclist, label='Bicyclist'), mpatches.Patch(color=Unlabelled, label='Unlabelled')]
    
    plt.figure(figsize=(0.2,0.2))
    plt.legend(handles=patches, ncol=12)
    plt.axis('off')
    plt.show()
    
def draw_plots_bayes(images, labels, predicted_labels, uncertainty):
    
    num_images = len(images)
    
    cols = ['Input', 'Ground truth', 'Output', 'Uncertainty']
    rows = ['Image {}'.format(row) for row in range(1,num_images+1)]
    #rows = ['Worst', 'Average', 'Best']

    fig, axes = plt.subplots(nrows=num_images, ncols=4, figsize=(20,num_images*4))
    

    for i in range(num_images):

        plt.subplot(num_images, 4, (4*i+1))
        plt.imshow(images[i])
        #plt.ylabel("Image %d" % (i+1), size='18')
        plt.ylabel(rows[i], size='22')
        plt.xticks([])
        plt.yticks([])

        if (i==0): 
            plt.title(cols[0], size='22', va='bottom')

        plt.subplot(num_images, 4, (4*i+2))
        writeImage(labels[i])
        plt.xticks([])
        plt.yticks([])

        if (i==0): 
            plt.title(cols[1], size='22', va='bottom')

        plt.subplot(num_images, 4, (4*i+3))
        writeImage(predicted_labels[i])
        plt.xticks([])
        plt.yticks([])

        if (i==0): 
            plt.title(cols[2], size='22', va='bottom')
            
        plt.subplot(num_images, 4, (4*i+4))
        plt.imshow(uncertainty[i], cmap = 'Greys')
        plt.xticks([])
        plt.yticks([])

        if (i==0): 
            plt.title(cols[3], size='22', va='bottom')

    plt.show()
    
    
def draw_plots_bayes_external(images, predicted_labels, uncertainty):
    
    num_images = len(images)
    
    cols = ['Input', 'Output', 'Uncertainty']
    rows = ['Image {}'.format(row) for row in range(1,num_images+1)]
    #rows = ['Worst', 'Average', 'Best']

    fig, axes = plt.subplots(nrows=num_images, ncols=3, figsize=(16,num_images*4))
    

    for i in range(num_images):

        plt.subplot(num_images, 3, (3*i+1))
        plt.imshow(images[i])
        #plt.ylabel("Image %d" % (i+1), size='18')
        plt.ylabel(rows[i], size='18')
        plt.xticks([])
        plt.yticks([])

        if (i==0): 
            plt.title(cols[0], size='18', va='bottom')

        plt.subplot(num_images, 3, (3*i+2))
        writeImage(predicted_labels[i])
        plt.xticks([])
        plt.yticks([])

        if (i==0): 
            plt.title(cols[1], size='18', va='bottom')
            
        plt.subplot(num_images, 3, (3*i+3))
        plt.imshow(uncertainty[i], cmap = 'Greys')
        plt.xticks([])
        plt.yticks([])

        if (i==0): 
            plt.title(cols[2], size='18', va='bottom')

    plt.show()