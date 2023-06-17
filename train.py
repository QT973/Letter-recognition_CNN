import cv2
import tensorflow as tf
from keras.utils import np_utils
import numpy as np
from imutils import paths
import os
from sklearn import preprocessing

data = [] 
labels = []

# Đưa tất cả đường dẫn trong thư mục data thành 1 list
imagePaths = list(paths.list_images('./data'))
# Tách lấy , append labels và data
for imagePath in imagePaths:
    # Tách lấy labels
    # split là như kiểu xóa bỏ còn os.path.sep nó là dấu gạch chéo . -2 là để lấy giá trị thứ 2 từ ngoài vào
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath, 1)
    image = cv2.resize(image, (64, 64))
    data.append(image)
    labels.append(label)
# Chuyển labels từ dạng chữ thành dạng số
le = preprocessing.LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
# print(labels)

data = np.array(data)
x_train = data.astype('float32')

x_train /= 255
y_train = np_utils.to_categorical(labels, 10)
# print(y_train)
img_width, img_height = 64, 64

# train_data_dir = 'data/train'
# validation_data_dir = 'data/test'
# nb_train_samples = 5000
# nb_validation_samples = 150
epochs = 70
batch_size = 16

input_shape = (img_width, img_height, 3)


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=(
        3, 3), input_shape=input_shape, activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(
        3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(300, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax),
])

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=epochs)
model.save('model.h5')
