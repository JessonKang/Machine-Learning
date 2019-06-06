'''
    搭建神经网络，实现手写数字识别
'''

import numpy as np
np.random.seed(1337)
from keras.datasets import mnist #手写数字数据集
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt

#数据加载
'''
X_train shape (60000,28*28)，y shape(60000)，即训练集一共6w张28*28的图片
X_test shape (10000,),测试集1w张图片
'''
(X_train,y_train),(X_test,y_test) = mnist.load_data()

print(X_train.shape[0])
print(X_test.shape[0])

#数据处理
X_train = X_train.reshape(X_train.shape[0],-1)/255. #归一化，这里是把每个像素点都除255吗？
X_test = X_test.reshape(X_test.shape[0],-1)/255.
y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)

#建立神经网络模型，不同于之前的regressor
model = Sequential([
    Dense(32,input_dim=784), #784为输入的神经元个数（即28*28），32为输出层神经元个数，可指定
    Activation('relu'), #激活函数
    Dense(10),
    Activation('softmax'),]
)

#定义优化器
rmsprop = RMSprop(lr=0.001,rho=0.9,epsilon=1e-08,decay=0.0)

#设置想要显示的度量
model.compile(optimizer=rmsprop,loss='categorical_crossentropy',metrics=['accuracy'])

#训练
print('Training-------------')
history = model.fit(X_train,y_train,epochs=2,batch_size=32)

#测试
print('Testing-------------')
loss,accuracy = model.evaluate(X_test,y_test)

print('test loss:',loss)
print('test accuracy:',accuracy)





