import cv2
import numpy as np
img1 = cv2.imread('/Users/pranoyr/PycharmProjects/Pytorch/lstm-object-detection-new/data/JPEGImages/000005.jpg')
img2 = cv2.imread('/Users/pranoyr/PycharmProjects/Pytorch/lstm-object-detection-new/data/JPEGImages/000007.jpg')
img3 = cv2.imread('/Users/pranoyr/PycharmProjects/Pytorch/lstm-object-detection-new/data/JPEGImages/000009.jpg')
img4 = cv2.imread('/Users/pranoyr/PycharmProjects/Pytorch/lstm-object-detection-new/data/JPEGImages/0000012.jpg')
img5 = cv2.imread('/Users/pranoyr/PycharmProjects/Pytorch/lstm-object-detection-new/data/JPEGImages/0000016.jpg')


print(img1.shape)
video = np.array([img1,img2,img3,img4,img5])


for i in video:
	print(i.shape)