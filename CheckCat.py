# function: tran a model to check whether the pictures have cats
# author: AloneMasterMan
# name: test
# platform: jupyter
#导入文件
import h5py
train_data = h5py.File(r'\Users\datasets\train_catvnoncat.h5','r')
test_data = h5py.File(r'\Users\datasets\test_catvnoncat.h5','r')

for key in train_data:
    print(key)

train_data['train_set_x'].shape

train_data['train_set_y'].shape

for key in test_data:
    print(key)

test_data['test_set_x'].shape

test_data['test_set_y'].shape

train_data_org =train_data['train_set_x'][:]
train_labels_org =train_data['train_set_y'][:]
test_data_org =test_data['test_set_x'][:]
test_labels_org =test_data['test_set_y'][:]

#查看图片
import matplotlib.pyplot as plt
%matplotlib inline
plt.imshow(train_data_org[176])

m_train = train_data_org.shape[0]
m_test = test_data_org.shape[0]
train_data_tran = train_data_org.reshape(m_train,-1).T
test_data_tran = test_data_org.reshape(m_test,-1).T

print(train_data_tran.shape,test_data_tran.shape)

import numpy as np#train_labels_org
train_labels = train_labels_org[np.newaxis,:]
test_labels = test_labels_org[np.newaxis,:]

#标注化数据
train_data_sta = train_data_tran/255
test_data_sta = test_data_tran/255
#print(train_data_sta[:9,:9])

#定义sigmiod函数
def sigmoid(z):
    s=1/(1+np.exp(-z))
    return s

#初始化函数
# n_dim = train_data_sta.shape[0]#求特征值
# # print(n_dim)
# w = np.zeros((n_dim,1))#定义0矩阵
# b = 0
# # n_dim = test_data_sta[0].shape
def initialize_with_zeros(dim):
    w=np.zeros((dim,1))
    b=0

    assert(w.shape==(dim,1))
    assert(isinstance(b,float) or isinstance(b,int))

    return w,b

#定义前向传播函数、代价函数、以及梯度下降
def propagate(w,b,X,Y):
    
    #定义前向传播函数
    A=sigmoid(np.dot(w.T,X)+b)
    
    #损失&代价函数lost&cost_function
    m=X.shape[1]
    cost=-np.sum(np.multiply(Y,np.log(A))+np.multiply(1-Y,np.log(1-A)))/m

    #梯度下降
    dw=1/m*np.dot(X,(A-Y).T)
    db=1/m*np.sum(A-Y)

    #断言区
    assert(dw.shape==w.shape)
    assert(db.dtype==float)
    cost = np.squeeze(cost)#删除数组中为1的那个维度
    assert(cost.shape == ())#cost为实数
    grads={'dw':dw,
           'db':db}
    return grads,cost

#优化部分
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs=[]
    for i in range(num_iterations):
        grads,cost=propagate(w,b,X,Y)
        dw=grads['dw']
        db=grads['db']
        w=w-learning_rate*dw
        b=b-learning_rate*db
        if i%100==0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    params={'w':w,
            'b':b}
    grads = {"dw": dw,
             "db": db}
    return params, grads, costs

#预测部分
def predict(w,b,X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        if A[0,i]>=0.5:
            Y_prediction[0,i]=1
        else:
            Y_prediction[0,i]=0
    assert(Y_prediction.shape == (1, m))
    return Y_prediction

#模型整合
def model(X_train,Y_train,X_test,Y_test,num_iterations=2000,learning_rate=0.005, print_cost = False):
    w,b=np.zeros((X_train.shape[0],1)),0
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = parameters["w"]
    b = parameters["b"]
    Y_prediction_test=predict(w,b,X_test)
    Y_prediction_train=predict(w,b,X_train)
    print('train accuracy: {} %'.format(100-np.mean(np.abs(Y_prediction_train-Y_train))*100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    d={"costs": costs,
        "Y_prediction_test": Y_prediction_test, 
        "Y_prediction_train" : Y_prediction_train, 
#         "w" : w, 
#         "b" : b,
        "learning_rate" : learning_rate,
        "num_iterations": num_iterations
    }
    return d

d = model(train_data_sta, train_labels_org, test_data_sta, test_labels_org, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
