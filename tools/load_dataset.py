from os import listdir
import cv2
import numpy as np

def load_imgs(path):
    imgs = []
    path_list = listdir(path)
    for img_path in path_list:
        img = cv2.imread(path+img_path, 0)
        imgs.append(img)
    return imgs

def load_dataset(path):
    fakePath = 'fake/'
    realPath = 'real/'

    fake_imgs = load_imgs(path+fakePath)


    fake_labels = np.zeros((len(fake_imgs), 1), dtype=bool)
    real_imgs = load_imgs(path+realPath)
    real_labels = np.ones((len(real_imgs), 1), dtype=bool)
    # print(fake_labels)

    imgs = np.concatenate((fake_imgs, real_imgs))
    labels = np.concatenate((fake_labels, real_labels))
    return imgs, labels
