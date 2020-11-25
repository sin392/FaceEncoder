import collections
import os
import torch
from glob import glob
from PIL import Image
import time

from model import FaceEncoder


def extract_lab(path):
    # data/test/{label}/image.jpg
    return os.path.basename(os.path.dirname(path))


img_lst = sorted(glob('data/test/*/*.jpg'))  # noqa
lab_lst = [extract_lab(x) for x in img_lst]

print("images :", len(set(img_lst)))
print("people :", len(set(lab_lst)))

print(img_lst[:10])
print(lab_lst[:10])

# print(collections.Counter(sorted(lab_lst)))
print(lab_lst.count("n00016"))
