from __future__ import print_function
import os
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
from SegNet import Segnet
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage

transform = transforms.Compose(
    [transforms.CenterCrop(320),
     transforms.ToTensor()])

dictionary = {
    0: (128, 128, 128),
    1: (128, 0, 0),
    2: (192, 192, 128),
    3: (128, 64, 128),
    4: (0, 0, 192),
    5: (128, 128, 0),
    6: (192, 128, 128),
    7: (64, 64, 128),
    8: (64, 0, 128),
    9: (64, 64, 0),
    10: (0, 128, 192),
    11: (0, 0, 0)
}

model = Segnet(3, 12)
model_path = 'model/segnet.pth'

model.load_state_dict(torch.load(model_path, map_location='cpu'))

test_image_path = 'dataset/CamVid/test/0001TP_009810.png'
test_image = Image.open(test_image_path).convert('RGB')
print('Operating...')
img = transform(test_image)
img = img.unsqueeze(0)
img = Variable(img)
label_image = model(img)
arr = np.transpose(label_image.squeeze(0).data.numpy(), [1, 2, 0])
arr_result = np.zeros((320, 320, 3))
for h in range(0, 319):
    for w in range(0, 319):
        var = arr[h, w, :].tolist()
        index = var.index(max(var))
        arr_result[h, w, 0] = dictionary.get(index)[0]
        arr_result[h, w, 1] = dictionary.get(index)[1]
        arr_result[h, w, 2] = dictionary.get(index)[2]

img = Image.fromarray(arr_result.astype('uint8')).convert('RGB')
img.show()