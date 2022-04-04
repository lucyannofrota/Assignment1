import cv2
import numpy as np

# Img Serialized 28x28 = 784
def feature_SerializedImg(img):
    return img.flatten()

# Color Histogram 256
def feature_colorHistogram(img):
    return cv2.calcHist([img], [0], None, [256], [0, 256])


# Color Mean 1
def feature_colorMean(img):
    return img.mean()

# [Sum noise per row 28 | Sum noise per col 28 | Sum noise all 1] = 57
def feature_noise(img):
    dimg = cv2.fastNlMeansDenoising(img, None, 8)
    ResImg = img - dimg
    return np.concatenate((ResImg.sum(axis=1), ResImg.sum(axis=0), [ResImg.sum()]))


# Serialized Bluried Img 28x28 = 784
def feature_gBlur(img):
    return feature_SerializedImg(cv2.GaussianBlur(img, (3, 3),0))


# Serialized Median Img 28x28 = 784
def feature_medianBlur(img):
    return feature_SerializedImg(cv2.medianBlur(img, 3))


# Serialized Canny Img 28x28 = 784
def feature_cannyEdge(img):
    return feature_SerializedImg(cv2.cv2.Canny(img, 100, 200))
