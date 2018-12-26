# 利用反向传播算法训练神经网络参数
import math
from numpy import *
import numpy as np
import random

def createDataset():
    X = mat([[0,0],[1,0],[0,1.9],[1,1],[2,2]])
    Y = mat([[1,0],[0,1],[0,1],[0,1],[0,1]])
    return X,Y

def sigmoid(x):
    return 1/(1+np.exp(-x))

def Predict(X,q,Weight_iy,Weight_yo,Theta_iy,Theta_yo):
    m = X.shape[0]  # 样本个数
    l = Y.shape[1]  # Y属性个数
    B = mat(zeros((m, q)))
    Yp = mat(zeros((m, l)))
    for k in range(m):
        ## Step2: 计算预测输出
        Alfa = X[k, :] * Weight_iy
        # 隐层输入矩阵
        for h in range(q):
            B[k, h] = sigmoid((Alfa - Theta_iy)[:, h])
        # 输出层输入矩阵
        Beta = B[k, :] * Weight_yo
        # 预测值
        for j in range(l):
            Yp[k, j] = sigmoid((Beta - Theta_yo)[:, j])
    return Yp

# neta:步长  q：隐层个数
def BPNN(X,Y,neta,q):
    d = X.shape[1]           # X属性个数
    l = Y.shape[1]           # Y属性个数
    m = X.shape[0]           # 样本个数
    B = mat(zeros((m,q)))
    g = mat(zeros((1,l)))
    e = mat(zeros((1,q)))
    Yp = mat(zeros((m,l)))
    Error = 100
    ErrorVec = mat(zeros((m,1)))
    flag = 0
    # Step1: 在（0，1）内随机初始化参数
    Weight_iy = mat(np.random.rand(d, q))
    Weight_yo = mat(np.random.rand(q, l))
    Theta_iy = mat(np.random.rand(1, q))
    Theta_yo = mat(np.random.rand(1, l))
    # print(Weight_iy)
    # print(Weight_yo)
    while(Error>0.001):
        flag += 1
        for k in range(m):
            ## Step2: 计算预测输出
            Alfa = X[k,:]*Weight_iy
            # 隐层输入矩阵
            for h in range(q):
                B[k,h] = sigmoid((Alfa - Theta_iy)[:,h])
            # 输出层输入矩阵
            Beta = B[k,:]*Weight_yo
            # 预测值
            for j in range(l):
                Yp[k,j] = sigmoid((Beta - Theta_yo)[:,j])
                ## Step3: 计算输出层梯度项 gj
                g[:,j] = Yp[k,j]*(1-Yp[k,j])*(Y[k,j]-Yp[k,j])
            ## Step4：计算隐层梯度项 eh
            for h in range(q):
                e[:,h] = B[k,h]*(1-B[k,h])*Weight_yo[h,:]*g.T
            ## Step5： 更新权值与阈值
            # 输出权值和阈值
            for h in range(q):
                for j in range(l):
                    Weight_yo[h,j] += neta*g[:,j]*B[k,h]
                    Theta_yo[:,j] -= neta*g[:,j]
            # 输入权值和阈值
            for i in range(d):
                for h in range(q):
                    Weight_iy[i, h] += neta * e[:,h] * X[k,i]
                    Theta_iy[:, h] -= neta * e[:,h]
        Yp = Predict(X, q, Weight_iy, Weight_yo, Theta_iy, Theta_yo)
        for k in range(m):
            ErrorVec[k,:] = 1/2*(Yp-Y)[k,:]*(Yp-Y)[k,:].T
        Error = mean(ErrorVec, axis=0)
    print(flag)
    return Weight_iy,Weight_yo,Theta_iy,Theta_yo

# main
if __name__ == '__main__':
    X,Y = createDataset()
    neta = 1
    q = 2
    Weight_iy, Weight_yo, Theta_iy, Theta_yo = BPNN(X, Y, neta, q)
    Xtest = mat([[0.5, 0.0], [1.8, 0.9]])
    Yp = Predict(Xtest, q, Weight_iy, Weight_yo, Theta_iy, Theta_yo)
    print(Yp)
    # print(Weight_iy)
    # print(Weight_yo)