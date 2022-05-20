from os import listdir
import cv2
import numpy as np

def load_imgs(path):
    imgs = []
    path_list = sort_by_number(listdir(path))
    for img_path in path_list:
        img = cv2.imread(path+img_path, 0)
        # img = img.flatten()
        imgs.append(img)
    return imgs

def sort_by_number(path_list):
    n = []
    for item in path_list:
        num = item.split('.')[0]
        if '_' in num:
            n.append(int(num.split('_')[1]))
        else:
            n.append(int(num))
    idx = np.argsort(n)
    return [path_list[i] for i in idx]

def load_dataset(path, test=False):
    if test:
        
        imgs = np.array(load_imgs(path+"images/"))
        
        labels = np.zeros((len(imgs), 1), dtype=bool)
    else:
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
