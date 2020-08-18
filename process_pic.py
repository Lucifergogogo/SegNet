from PIL import Image
import os
import numpy as np

txt_path = 'dataset/CamVid/train.txt'
label_path = 'dataset/CamVid/train_labels_pre/'
save_path = 'dataset/CamVid/train_labels/'
size_w = 480
size_h = 360
dictionary = {
    (128, 128, 128): 0,
    (128, 0, 0): 1,
    (192, 192, 128): 2,
    (128, 64, 128): 3,
    (0, 0, 192): 4,
    (128, 128, 0): 5,
    (192, 128, 128): 6,
    (64, 64, 128): 7,
    (64, 0, 128): 8,
    (64, 64, 0): 9,
    (0, 128, 192): 10,
    (0, 0, 0): 11
}

ims = []
fh = open(txt_path, 'r')
imgs = []
for image_names in fh:
    image_names = image_names.rstrip()
    names = image_names.split('/')
    imgs.append(names[1])

for name in imgs:
    image = Image.open(label_path + name).convert('RGB')
    arr_pre = np.array(image)
    arr_tar = np.zeros((size_h, size_w), dtype=int)
    for h in range(1, size_h):
        for w in range(1, size_w):
            var = arr_pre[h - 1, w - 1, :]
            arr_tar[h - 1, w - 1] = dictionary.get((var[0], var[1], var[2]))
    image = Image.fromarray(arr_tar)
    image.save(save_path + name)

print('end')