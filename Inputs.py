import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import os, sys
import numpy as np
import math
#import skimage
#import skimage.io

IMAGE_HEIGHT = 360
IMAGE_WIDTH = 480
IMAGE_DEPTH = 3

NUM_CLASSES = 11
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 367
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 101
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 3-D Tensor of [height, width, 1] type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 3D tensor of [batch_size, height, width ,1] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 1
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + (1+num_preprocess_threads) * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + (1+num_preprocess_threads) * batch_size)

  # Display the training images in the visualizer.
  # tf.image_summary('images', images)

  return images, label_batch

def CamVid_reader_seq(filename_queue, seq_length):
  image_seq_filenames = tf.split(axis=0, num_or_size_splits=seq_length, value=filename_queue[0])
  label_seq_filenames = tf.split(axis=0, num_or_size_splits=seq_length, value=filename_queue[1])

  image_seq = []
  label_seq = []
  for im ,la in zip(image_seq_filenames, label_seq_filenames):
    imageValue = tf.read_file(tf.squeeze(im))
    labelValue = tf.read_file(tf.squeeze(la))
    image_bytes = tf.image.decode_png(imageValue)
    label_bytes = tf.image.decode_png(labelValue)
    image = tf.cast(tf.reshape(image_bytes, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH)), tf.float32)
    label = tf.cast(tf.reshape(label_bytes, (IMAGE_HEIGHT, IMAGE_WIDTH, 1)), tf.int64)
    image_seq.append(image)
    label_seq.append(label)
  return image_seq, label_seq

def CamVid_reader(filename_queue):

  image_filename = filename_queue[0]
  label_filename = filename_queue[1]

  imageValue = tf.read_file(image_filename)
  labelValue = tf.read_file(label_filename)

  image_bytes = tf.image.decode_png(imageValue)
  label_bytes = tf.image.decode_png(labelValue)

  image = tf.reshape(image_bytes, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))
  label = tf.reshape(label_bytes, (IMAGE_HEIGHT, IMAGE_WIDTH, 1))

  return image, label

def get_filename_list(path):
  fd = open(path)
  image_filenames = []
  label_filenames = []
  filenames = []
  for i in fd:
    i = i.strip().split(" ")
    image_filenames.append(i[0])
    label_filenames.append(i[1])
    
  #image_filenames = ["/zhome/1c/2/114196/Documents" + name for name in image_filenames]
  #label_filenames = ["/zhome/1c/2/114196/Documents" + name for name in label_filenames]
  image_filenames = ["." + name for name in image_filenames]
  label_filenames = ["." + name for name in label_filenames]
  return image_filenames, label_filenames

def CamVidInputs(image_filenames, label_filenames, batch_size,min_queue_examples):

  images = ops.convert_to_tensor(image_filenames, dtype=dtypes.string)
  labels = ops.convert_to_tensor(label_filenames, dtype=dtypes.string)

  filename_queue = tf.train.slice_input_producer([images, labels], shuffle=True)

  image, label = CamVid_reader(filename_queue)
  reshaped_image = tf.cast(image, tf.float32)

  #min_fraction_of_examples_in_queue = 0.4
  #min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
  #                         min_fraction_of_examples_in_queue)
  print ('Filling queue with %d CamVid images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)
 
  

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(reshaped_image, label,
                                         min_queue_examples, batch_size,
                                         shuffle=False)
#def get_all_test_data(im_list, la_list):
#  images = []
#  labels = []
#  index = 0
#  for im_filename, la_filename in zip(im_list, la_list):
#    im = np.array(skimage.io.imread(im_filename), np.float32)
#    im = im[np.newaxis]
#    la = skimage.io.imread(la_filename)
#    la = la[np.newaxis]
#    la = la[...,np.newaxis]
#    images.append(im)
#    labels.append(la)
#  return images, label
                                         
def TestLoading(image,label):
    """
    Input: 
    image: the tensor, size is NumBatch, height, width, channel
    label: tne tensor, size is NumBatch, height, width, channel
    Output:
    image_batch
    label_batch
    """
    max_step = 6
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)
        image_tot = []
        label_tot = []
        for step in range(max_step):
            image_batch, label_batch = sess.run([image,label])
            image_tot.append(image_batch)
            label_tot.append(label_batch)
        
        
        coord.request_stop()
        coord.join(threads)
        
        return image_tot, label_tot

import matplotlib.pyplot as plt

def ImageShowing(image):
    fake = []
    for diff in image:
        for sep in diff:
            fake.append(sep)
    for i in range(18):
        plt.imshow(fake[i])
        plt.figure(i+1)
    plt.show()
                
                
    
    
def ShortImageFilename(image_filename,label_filename):
    """
    Input: the full image_filename, label_filename
    Output: the short image_filename, label_filename
    """
    index = [0,64,109,122,145,166,250,288,300]
    image_filename_short = []
    label_filename_short = []
    for i in index:
        image_filename_short.append(image_filename[i])
        label_filename_short.append(label_filename[i])
        
    return image_filename_short, label_filename_short
    
def Test(path):
    """
    To test if the CamVidInputs can really give us random mini-batch training image.
    To see the result, we choose only 9 images, the index in function ShortImageFilename represent the images that we choose, 
    the reason to choose those specific image is that the images in the Training dataset are so similiar, to see the difference
    between the images better, I only choose these 9 images. 
    Then the max_step is 6, and the batch_size is 3, so it will give us 18 images. Since we set the min_queue_examples to be
    8 really close to total image size 9, which means that the images which are dequeued should be uniformed choosed from 
    these 9 images, and the capacity is 8+2*3=14, so it doesn't matter that the capacity larger than 9. The most ideal situation is
    that for these 18 images that we get, each image should appear for 2 times, and all the image in the dataset(9) images should
    at least appear once, 2*9. The practical result is that these 9 images all appeared, and some image appears two times(2), some 
    images appear 3 times(3), some image appear 1 time(3), so although not all the images appear 2 times, but still all the images 
    are appeard. For one batch, images generated are different if set shuffle=False tf.train.batch. 
    """
    image_filename,label_filename = get_filename_list(path)
    image_filename_short,label_filename_short = ShortImageFilename(image_filename,label_filename)
    min_queue_examples = 8
    batch_size = 3
    print("The number of images that we are going to test:",np.shape(image_filename_short))
    print("Batch Size:",batch_size)
    image,label = CamVidInputs(image_filename_short,label_filename_short,batch_size,min_queue_examples)
    image_batch,label_batch = TestLoading(image,label)
    print("The shape of the output training image for all the iterations:",np.shape(image_batch))
    ImageShowing(image_batch)
    