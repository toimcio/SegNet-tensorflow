import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

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
    
def draw_plots(images, labels, predicted_labels):
    
    num_images = len(images)
    
    cols = ['Input', 'Ground truth', 'Output']
    rows = ['Image {}'.format(row) for row in range(1,num_images+1)]

    fig, axes = plt.subplots(nrows=num_images, ncols=3, figsize=(20,num_images*28/5))

    for i in range(num_images):

        plt.subplot(num_images, 3, (3*i+1))
        plt.imshow(images[i])
        plt.ylabel("Image %d" % (i+1), size='18')

        if (i==0): 
            plt.title(cols[0], size='18', va='bottom')

        plt.subplot(num_images, 3, (3*i+2))
        writeImage(labels[i])

        if (i==0): 
            plt.title(cols[1], size='18', va='bottom')

        plt.subplot(num_images, 3, (3*i+3))
        writeImage(predicted_labels[i])

        if (i==0): 
            plt.title(cols[2], size='18', va='bottom')

    plt.show()