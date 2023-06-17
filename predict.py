import tensorflow as tf
from keras.models import load_model
import cv2
import numpy as np

lis = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
model = load_model('model.h5')
img_test = cv2.imread("Chu_viet_tay.jpg")   #chữ j
# img_test = cv2.imread("img011-00009.png")   # Chữ A
# img_test = cv2.imread('img012-00004.png') #chu B
# img_test = cv2.imread('img018-00003.png') #chu H
#img_test = cv2.imread('img020-00003.png') #chu J
img_test = cv2.resize(img_test, (64, 64))
img_test = img_test.reshape(1, 64, 64, 3)
print(model.predict(img_test))
class_labels = np.argmax(model.predict(img_test))    # lấy vị trí có giá trị lớn nhất trong mảng 
print('kết quả dự đoán là :',lis[class_labels])