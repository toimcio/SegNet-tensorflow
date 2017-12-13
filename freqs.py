from scipy import misc
import numpy as np
import operator


with open('SegNet/sun3d_dataset/train.txt') as fd:
    image_filenames = []
    label_filenames = []
    for i in fd:
        i = i.strip().split(" ")
        image_filenames.append(i[0])
        label_filenames.append(i[1])

with open('SegNet/sun3d_dataset/val.txt') as fd:
    for i in fd:
        i = i.strip().split(" ")
        image_filenames.append(i[0])
        label_filenames.append(i[1])

with open('SegNet/sun3d_dataset/test.txt') as fd:
    for i in fd:
        i = i.strip().split(" ")
        image_filenames.append(i[0])
        label_filenames.append(i[1])

    image_filenames = ['.' + name for name in image_filenames]
    label_filenames = ['.' + name for name in label_filenames]

dict_labels = dict()


for image_path in label_filenames:
    image = misc.imread(image_path)
    for r in image:
        for k in r:
            if k in dict_labels:
                dict_labels[k] += 1
            else:
                dict_labels[k] = 1

sorted = sorted(dict_labels.items(), key=operator.itemgetter(0))
print(sorted)
values = list(dict_labels.values())
median_freq = np.median([v/sum(values) for v in values])
freqs = {k: v/sum(values) for k, v in dict_labels.items()}
weights = {k: median_freq/v for k, v in freqs.items()}
print(sorted)
print(weights)