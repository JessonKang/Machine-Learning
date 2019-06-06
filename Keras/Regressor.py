import tensorflow as tf
import numpy as np
np.random.seed(1337)
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

#create some data
X = np.linspace(-1,1,200) #生成两百个在[-1,1]之间的数据，注意这些数据是按照有规律的大小排列的
np.random.shuffle(X) #打乱这些数据
Y = 0.5*X + 2 + np.random.normal(0,0.05,(200,))

#plot data
plt.scatter(X,Y)
plt.show()

#划分数据
X_train,Y_train = X[:160],Y[:160] #前160个样本当训练集
X_test,Y_test = X[160:],Y[160:] #测试集

#建立一个神经网络模型
model = Sequential()
model.add(Dense(units=1,input_dim=1))

#选择损失函数
model.compile(loss='mse',optimizer='sgd')

#训练
print('Traning-------------------------')
for step in range(301): #训练301次
    cost = model.train_on_batch(X_train,Y_train)
    if step % 100 == 0: #每100次输出一次loss值
        print('train cost:',cost)

#测试
print('\nTesting-------------------------')
cost = model.evaluate(X_test,Y_test,batch_size=40) #
print('test cost:',cost)

W,b = model.layers[0].get_weights() #返回神经网络第1层的参数
print('Weights=',W,'\nbiases=',b)

#根据学习到的参数给测试数据画图
Y_pred = model.predict(X_test) #根据样本获取对应的预测值
plt.scatter(X_test,Y_test)
plt.plot(X_test,Y_pred)
plt.show()










