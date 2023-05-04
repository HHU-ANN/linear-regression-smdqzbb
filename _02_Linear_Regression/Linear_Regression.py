# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    X,y = read_data()
    yMean = numpy.mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = numpy.mean(xMat, 0)
    xVar = numpy.var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    xTx = xMat.T * xMat
    denom = xTx + numpy.eye(numpy.shape(xMat)[1]) * numpy.exp(-9)
    if numpy.linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws.T
    except Exception:
    return numpy.nan
def calculate(ml_data, ml_label, calculate_data):
    ws = ridge_regres(ml_data, ml_label)
    if (isinstance(ws, float) or isinstance(ws, numpy.float64)) and (numpy.isnan(ws) 
    or numpy.isnan(ws[0][0])):
        return numpy.nan
    calculateMat = numpy.mat(calculate_data)
    xMat = numpy.mat(ml_data)
    xMeans = numpy.mean(xMat, 0)
    xVar = numpy.var(xMat, 0)
    calculateMat = (calculateMat - xMeans) / xVar
    calculate_result = calculateMat * numpy.mat(ws).T + numpy.mean(ml_label)
    print (calculate_result)
def lasso(data):
    X,y = read_data()
    weight = np.matmul(np.linalg.inv(np.matmul(X.T,X)),np.matmul(X.T,y))
    return weight @ data

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
